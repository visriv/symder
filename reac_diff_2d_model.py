import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

import haiku as hk
import optax

import os.path
from functools import partial
from tqdm.auto import tqdm
import argparse

from data.utils import get_dataset
from data.reac_diff_2d import generate_dataset

from symder.encoder_utils import concat_visible, append_dzdt
from symder.sym_models import SymModel, Quadratic, FFT2, rescale_z
from symder.symder import get_symder_apply, get_model_apply

from utils import loss_fn, save_pytree, load_pytree


def get_model(num_visible, num_hidden, num_der, mesh, dx, dt, scale, get_dzdt=False):

    # Define encoder
    hidden_size = 128
    pad = 2
    def encoder(x):
        return hk.Sequential(
                [lambda x: jnp.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad), (0, 0)), 'wrap'),
                 hk.Conv3D(hidden_size, kernel_shape=5, padding='VALID'), jax.nn.relu,
                 hk.Conv3D(hidden_size, kernel_shape=1), jax.nn.relu,
                 hk.Conv3D(num_hidden, kernel_shape=1)
                ])(x)

    encoder = hk.without_apply_rng(hk.transform(encoder))
    encoder_apply = append_dzdt(encoder.apply) if get_dzdt else encoder.apply
    encoder_apply = concat_visible(encoder_apply, visible_transform=lambda x: x[:, pad:-pad])

    # Define symbolic model
    n_dims = num_visible + num_hidden
    scale_vec = jnp.concatenate((scale[:,0], jnp.ones(num_hidden)))
    @partial(rescale_z, scale_vec=scale_vec)
    def sym_model(z, t):
        return SymModel(dt, 
                (FFT2(mesh, dx, init=jnp.zeros),
                 hk.Linear(n_dims, w_init=jnp.zeros, b_init=jnp.zeros),
                 Quadratic(n_dims, init=jnp.zeros)
                ))(z, t)

    sym_model = hk.without_apply_rng(hk.transform(sym_model))

    # Define SymDer function which automatically computes 
    # higher order time derivatives of symbolic model
    symder_apply = get_symder_apply(sym_model.apply, num_der=num_der, 
                    transform=lambda z: z[..., :num_visible], get_dzdt=get_dzdt)

    # Define full model, combining encoder and symbolic model
    model_apply = get_model_apply(encoder_apply, symder_apply, 
                    encoder_name='encoder', sym_model_name='sym_model', 
                    hidden_transform=lambda z: z[..., -num_hidden:], get_dzdt=get_dzdt)
    model_init = {'encoder': encoder.init, 'sym_model': sym_model.init}

    return model_apply, model_init, {'pad': pad}


def train(n_steps, model_apply, params, scaled_data, 
            loss_fn_args={}, data_args={}, optimizers={},
            sparse_thres=None, sparse_interval=None, 
            key_seq=hk.PRNGSequence(42), multi_gpu=False):
    
    # JIT compile gradient function
    loss_fn_apply = partial(loss_fn, model_apply, **loss_fn_args)
    if multi_gpu:
        def grad_loss(params, batch, target):
            grad_out = jax.grad(loss_fn_apply, has_aux=True)(params, batch, target)
            return lax.pmean(grad_out, axis_name='devices')
        grad_loss = jax.pmap(grad_loss, axis_name='devices')
    else:
        grad_loss = jax.jit(jax.grad(loss_fn_apply, has_aux=True))

    # Initialize optimizers
    opt_init, opt_update, opt_state = {}, {}, {}
    for name in params.keys():
        opt_init[name], opt_update[name] = optimizers[name]
        if multi_gpu:
            opt_state[name] = jax.pmap(opt_init[name])(params[name])
        else:
            opt_state[name] = opt_init[name](params[name])

    # Define update function
    def update_params(grads, opt_state, params, sparse_mask):
        if sparsify:
            grads['sym_model'] = jax.tree_multimap(jnp.multiply, sparse_mask, grads['sym_model'])

        updates = {}
        for name in params.keys():
            updates[name], opt_state[name] = opt_update[name](grads[name], opt_state[name], params[name])
        params = optax.apply_updates(params, updates)

        if sparsify:
            params['sym_model'] = jax.tree_multimap(jnp.multiply, sparse_mask, params['sym_model'])

        if multi_gpu:
            # Ensure params, opt_state, sparse_mask are the same across all devices
            params = lax.pmean(params, axis_name='devices')
            opt_state, sparse_mask = lax.pmax((opt_state, sparse_mask), axis_name='devices')

        return params, opt_state, sparse_mask

    if multi_gpu:
        update_params = jax.pmap(update_params, axis_name='devices')
    else:
        update_params = jax.jit(update_params)

    # Get batch and target
    # TODO: replace this with call to a data generator/data loader
    if multi_gpu:
        n_devices = jax.device_count()
        pad = data_args['pad']
        time_size = (scaled_data.shape[0] - 2*pad) // n_devices
        batch = []
        target = []
        for i in range(n_devices):
            start, end = i*time_size, (i+1)*time_size + 2*pad
            if loss_fn_args['reg_dzdt'] is not None:
                batch.append(scaled_data[ None, start:end, :, :, :, :2]) # batch, time, mesh, mesh, num_visible, 2
            else:
                batch.append(scaled_data[None, start:end, :, :, :]) # batch, time, mesh, mesh, num_visible
            target.append(scaled_data[None, start+pad:end-pad, :, :, :, 1:]) # batch, time, mesh, mesh, num_visible, num_der

        batch = jax.device_put_sharded(batch, jax.devices())
        target = jax.device_put_sharded(target, jax.devices())

    else:
        if loss_fn_args['reg_dzdt'] is not None:
            batch = scaled_data[None, :, :, :, :, :2] # batch, time, mesh, mesh, num_visible, 2
        else:
            batch = scaled_data[None, :, :, :, :] # batch, time, mesh, mesh, num_visible
        pad = data_args['pad']
        target = scaled_data[None, pad:-pad, :, :, :, 1:] # batch, time, mesh, mesh, num_visible, num_der

        batch = jnp.asarray(batch)
        target = jnp.asarray(target)

    # Initialize sparse mask
    sparsify = sparse_thres is not None and sparse_interval is not None
    if sparsify:
        if multi_gpu:
            sparse_mask = jax.tree_map(jax.pmap(lambda x: jnp.ones_like(x, dtype=bool)), params['sym_model'])
        else:
            sparse_mask = jax.tree_map(lambda x: jnp.ones_like(x, dtype=bool), params['sym_model'])

    # Training loop
    if multi_gpu:
        print(f"Training for {n_steps} steps on {n_devices} devices...")
    else:
        print(f"Training for {n_steps} steps...")

    best_loss = np.float('inf')
    best_params = None

    thres_fn = lambda x: jnp.abs(x) > sparse_thres
    if multi_gpu:
        thres_fn = jax.pmap(thres_fn)

    for step in range(n_steps):
        # Compute gradients and losses
        grads, loss_list = grad_loss(params, batch, target)

        # Save best params if loss is lower than best_loss
        loss = loss_list[0][0] if multi_gpu else loss_list[0]
        if loss < best_loss:
            best_loss = loss
            best_params = jax.tree_map(lambda x: x.copy(), params)

        # Update sparse_mask based on a threshold
        if step > 0 and step % sparse_interval == 0:
            sparse_mask = jax.tree_map(thres_fn, best_params['sym_model'])
            
        # Update params based on optimizers
        params, opt_state, sparse_mask = update_params(grads, opt_state, params, sparse_mask)
        
        # Print loss
        if step % 100 == 0:
            loss, mse, reg_dzdt, reg_l1_sparse = loss_list
            if multi_gpu:
                loss, mse, reg_dzdt, reg_l1_sparse = loss[0], mse[0], reg_dzdt[0], reg_l1_sparse[0]
            print(f'Loss[{step}] = {loss}, MSE = {mse}, Reg. dz/dt = {reg_dzdt}, Reg. L1 Sparse = {reg_l1_sparse}')
            if multi_gpu:
                print(jax.tree_map(lambda x: x[0], params['sym_model']))
            else:
                print(params['sym_model'])

    if multi_gpu:
        best_params = jax.tree_map(lambda x: x[0], best_params)
        sparse_mask = jax.tree_map(lambda x: x[0], sparse_mask)

    print('\nBest loss:', best_loss)
    print('Best sym_model params:', best_params['sym_model'])
    return best_loss, best_params, sparse_mask
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run SymDer model on 2D reaction-diffusion data.")
    parser.add_argument('-o', '--output', type=str, default="./reac_diff_2d_run0/", 
        help="Output folder path. Default: ./reac_diff_2d_run0/")
    parser.add_argument('-d', '--dataset', type=str, default="./data/reac_diff_2d.npz", 
        help="Path to 2D reaction-diffusion dataset (generated and saved if it does not exist). Default: ./data/reac_diff_2d.npz")
    args = parser.parse_args()

    # Seed random number generator
    key_seq = hk.PRNGSequence(42)

    # Set dataset parameters and load/generate dataset
    l = 64
    mesh = 64
    dt = 5e-2
    tspan = (0, 50 + 4*dt)
    scaled_data, scale = get_dataset(args.dataset, generate_dataset, 
                                     l=l, mesh=mesh, dt=dt, tspan=tspan)

    # Set SymDer parameters
    num_visible = 1
    num_hidden = 1
    num_der = 2

    # Set training hyperparameters
    n_steps = 50000
    sparse_thres = 2e-3
    sparse_interval = 1000
    multi_gpu = True

    # Define optimizers
    optimizers = {'encoder': optax.adabelief(1e-3, eps=1e-16), #optax.adamw(1e-3, weight_decay=1e-2), 
                  'sym_model': optax.adabelief(1e-3, eps=1e-16) #optax.adam(1e-3)
                 }

    # Set loss function hyperparameters
    loss_fn_args = {'scale': jnp.array(scale), 
                    'deriv_weight': jnp.array([1., 1.]), 
                    'reg_dzdt': 0, 
                    'reg_l1_sparse': 0}
    get_dzdt = loss_fn_args['reg_dzdt'] is not None

    # Check dataset shapes
    assert scaled_data.shape[-2] == num_visible
    assert scaled_data.shape[-1] == num_der + 1
    assert scale.shape[0] == num_visible
    assert scale.shape[1] == num_der + 1

    # Define model
    model_apply, model_init, model_args = get_model(num_visible, num_hidden, num_der, 
                                                    mesh, np.sqrt(1e1)*l/mesh,
                                                    1e1*dt, scale, get_dzdt=get_dzdt)

    # Initialize parameters
    params = {}
    params['encoder'] = model_init['encoder'](next(key_seq), 
                            jnp.ones([1, scaled_data.shape[1], mesh, mesh, num_visible]))
    params['sym_model'] = model_init['sym_model'](next(key_seq), 
                            jnp.ones([1, 1, mesh, mesh, num_visible + num_hidden]), 0.)
    if multi_gpu:
        for name in params.keys():
            params[name] = jax.device_put_replicated(params[name], jax.devices())

    # Train
    best_loss, best_params, sparse_mask = train(n_steps, 
                                                model_apply, 
                                                params, 
                                                scaled_data, 
                                                loss_fn_args=loss_fn_args, 
                                                data_args={'pad': model_args['pad']},
                                                optimizers=optimizers, 
                                                sparse_thres=sparse_thres, 
                                                sparse_interval=sparse_interval, 
                                                key_seq=key_seq,
                                                multi_gpu=multi_gpu)

    # Save model parameters and sparse mask
    save_pytree(os.path.join(args.output, 'best_params.pickle'), best_params)
    save_pytree(os.path.join(args.output, 'sparse_mask.pickle'), sparse_mask)




