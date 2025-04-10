import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

import haiku as hk
import optax
import wandb
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os.path
import argparse
from functools import partial
from omegaconf import OmegaConf, DictConfig

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print("[INFO] Setting JAX to use GPU 2")

# from tqdm.auto import tqdm

from data.utils import get_dataset
from data.reac_diff_2d import generate_dataset

from encoder.utils import append_dzdt, concat_visible
from symder.sym_models import SymModel, Quadratic, SpatialDerivative2D, rescale_z
from symder.symder import get_symder_apply, get_model_apply

from utils import loss_fn, init_optimizers, save_pytree  # , load_pytree


def get_model(num_visible, num_hidden, num_der, mesh, dx, dt, scale, get_dzdt=False):

    # Define encoder
    hidden_size = 64
    pad = 2

    def encoder(x):
        return hk.Sequential(
            [
                lambda x: jnp.pad(
                    x, ((0, 0), (0, 0), (pad, pad), (pad, pad), (0, 0)), "wrap"
                ),
                hk.Conv3D(hidden_size, kernel_shape=5, padding="VALID"),
                jax.nn.relu,
                hk.Conv3D(hidden_size, kernel_shape=1),
                jax.nn.relu,
                hk.Conv3D(num_hidden, kernel_shape=1),
            ]
        )(x)

    encoder = hk.without_apply_rng(hk.transform(encoder))
    encoder_apply = append_dzdt(encoder.apply) if get_dzdt else encoder.apply
    encoder_apply = concat_visible(
        encoder_apply, visible_transform=lambda x: x[:, pad:-pad]
    )

    # Define symbolic model
    n_dims = num_visible + num_hidden
    scale_vec = jnp.concatenate((scale[:, 0], jnp.ones(num_hidden)))

    @partial(rescale_z, scale_vec=scale_vec)
    def sym_model(z, t):
        return SymModel(
            1e1 * dt,
            (
                SpatialDerivative2D(mesh, np.sqrt(1e1) * dx, init=jnp.zeros),
                hk.Linear(n_dims, w_init=jnp.zeros, b_init=jnp.zeros),
                Quadratic(n_dims, init=jnp.zeros),
            ),
        )(z, t)

    sym_model = hk.without_apply_rng(hk.transform(sym_model))

    # Define SymDer function which automatically computes
    # higher order time derivatives of symbolic model
    symder_apply = get_symder_apply(
        sym_model.apply,
        num_der=num_der,
        transform=lambda z: z[..., :num_visible],
        get_dzdt=get_dzdt,
    )

    # Define full model, combining encoder and symbolic model
    model_apply = get_model_apply(
        encoder_apply,
        symder_apply,
        hidden_transform=lambda z: z[..., -num_hidden:],
        get_dzdt=get_dzdt,
    )
    model_init = {"encoder": encoder.init, "sym_model": sym_model.init}

    return model_apply, model_init, {"pad": pad}


def train(
    n_steps,
    model_apply,
    params,
    scaled_data,
    loss_fn_args={},
    data_args={},
    optimizers={},
    sparse_thres=None,
    sparse_interval=None,
    key_seq=hk.PRNGSequence(42),
    multi_gpu=False,
    exp_dir=None,
    log_interval=100,
    save_interval=1000,
    early_stopping_config=None,
):

    # JIT compile/PMAP gradient function
    loss_fn_apply = partial(loss_fn, model_apply, **loss_fn_args)
    if multi_gpu:
        # Take mean of gradient across multiple devices
        def grad_loss(params, batch, target):
            grad_out = jax.grad(loss_fn_apply, has_aux=True)(params, batch, target)
            return lax.pmean(grad_out, axis_name="devices")

        grad_loss = jax.pmap(grad_loss, axis_name="devices")
    else:
        grad_loss = jax.jit(jax.grad(loss_fn_apply, has_aux=True))

    # Initialize sparse mask
    sparsify = sparse_thres is not None and sparse_interval is not None
    if multi_gpu:
        sparse_mask = jax.tree_map(
            jax.pmap(lambda x: jnp.ones_like(x, dtype=bool)), params["sym_model"]
        )
    else:
        sparse_mask = jax.tree_map(
            lambda x: jnp.ones_like(x, dtype=bool), params["sym_model"]
        )

    # Initialize optimizers
    update_params, opt_state = init_optimizers(params, optimizers, sparsify, multi_gpu)
    if multi_gpu:
        update_params = jax.pmap(update_params, axis_name="devices")
    else:
        update_params = jax.jit(update_params)

    # Get batch and target
    # TODO: replace this with call to a data generator/data loader
    if multi_gpu:
        n_devices = jax.device_count()
        pad = data_args["pad"]
        time_size = (scaled_data.shape[0] - 2 * pad) // n_devices
        batch = []
        target = []
        for i in range(n_devices):
            start, end = i * time_size, (i + 1) * time_size + 2 * pad
            if loss_fn_args["reg_dzdt"] is not None:
                # batch, time, mesh, mesh, num_visible, 2
                batch.append(scaled_data[None, start:end, :, :, :, :2])
            else:
                # batch, time, mesh, mesh, num_visible
                batch.append(scaled_data[None, start:end, :, :, :, 0])
            # batch, time, mesh, mesh, num_visible, num_der
            target.append(scaled_data[None, start + pad : end - pad, :, :, :, 1:])

        batch = jax.device_put_sharded(batch, jax.devices())
        target = jax.device_put_sharded(target, jax.devices())

    else:
        if loss_fn_args["reg_dzdt"] is not None:
            # batch, time, mesh, mesh, num_visible, 2
            batch = scaled_data[None, :, :, :, :, :2]
        else:
            # batch, time, mesh, mesh, num_visible
            batch = scaled_data[None, :, :, :, :, 0]
        pad = data_args["pad"]
        # batch, time, mesh, mesh, num_visible, num_der
        target = scaled_data[None, pad:-pad, :, :, :, 1:]

        batch = jnp.asarray(batch)
        target = jnp.asarray(target)

    # Training loop
    if multi_gpu:
        print(f"Training for {n_steps} steps on {n_devices} devices...")
    else:
        print(f"Training for {n_steps} steps...")
    
    pbar = tqdm(range(n_steps), desc="Training")

    best_loss = float("inf")
    best_params = None
    loss_history = []
    mse_history = []
    reg_dzdt_history = []
    reg_l1_sparse_history = []
    
    # Early stopping variables
    if early_stopping_config and early_stopping_config.enabled:
        patience = early_stopping_config.patience
        min_delta = early_stopping_config.min_delta
        min_steps = early_stopping_config.min_steps
        best_loss_so_far = float("inf")
        steps_without_improvement = 0
        should_stop = False

    def thres_fn(x):
        return jnp.abs(x) > sparse_thres

    if multi_gpu:
        thres_fn = jax.pmap(thres_fn)

    for step in pbar:
        # Compute gradients and losses
        grads, loss_list = grad_loss(params, batch, target)

        # Save best params if loss is lower than best_loss
        loss = loss_list[0][0] if multi_gpu else loss_list[0]
        if loss < best_loss:
            best_loss = loss
            best_params = jax.tree_map(lambda x: x.copy(), params)

        # Early stopping check
        if early_stopping_config and early_stopping_config.enabled:
            if step >= min_steps:
                if loss < best_loss_so_far - min_delta:
                    best_loss_so_far = loss
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1
                
                if steps_without_improvement >= patience:
                    should_stop = True
                    print(f"\nEarly stopping triggered at step {step}")
                    print(f"Best loss: {best_loss_so_far:.6f}")
                    break

        # Update sparse_mask based on a threshold
        if sparsify and step > 0 and step % sparse_interval == 0:
            sparse_mask = jax.tree_map(thres_fn, best_params["sym_model"])

        # Update params based on optimizers
        params, opt_state, sparse_mask = update_params(
            grads, opt_state, params, sparse_mask
        )

        # Log metrics
        loss, mse, reg_dzdt, reg_l1_sparse = loss_list
        if multi_gpu:
            (loss, mse, reg_dzdt, reg_l1_sparse) = (
                loss[0],
                mse[0],
                reg_dzdt[0],
                reg_l1_sparse[0],
            )
        
        loss_history.append(loss)
        mse_history.append(mse)
        reg_dzdt_history.append(reg_dzdt)
        reg_l1_sparse_history.append(reg_l1_sparse)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss:.4f}',
            'mse': f'{mse:.4f}',
            'best_loss': f'{best_loss:.4f}'
        })

        # Log to wandb and save checkpoints
        if step % log_interval == 0:
            wandb.log({
                'loss': loss,
                'mse': mse,
                'reg_dzdt': reg_dzdt,
                'reg_l1_sparse': reg_l1_sparse,
                'best_loss': best_loss,
                'step': step
            })
            
            # Plot and save loss curves
            plt.figure(figsize=(10, 6))
            plt.plot(loss_history, label='Loss')
            plt.plot(mse_history, label='MSE')
            plt.plot(reg_dzdt_history, label='Reg dz/dt')
            plt.plot(reg_l1_sparse_history, label='Reg L1 Sparse')
            plt.xlabel('Steps')
            plt.ylabel('Value')
            plt.title('Training Metrics')
            plt.legend()
            plt.savefig(os.path.join(exp_dir, 'loss_curves.png'))
            plt.close()
            
        # Save checkpoint
        if step % save_interval == 0:
            checkpoint_path = os.path.join(exp_dir, f'checkpoint_{step:06d}.pt')
            save_pytree(
                checkpoint_path,
                {
                    'params': params,
                    'best_params': best_params,
                    'sparse_mask': sparse_mask,
                    'opt_state': opt_state,
                    'step': step,
                    'loss': loss,
                    'best_loss': best_loss
                }
            )

    if multi_gpu:
        best_params = jax.tree_map(lambda x: x[0], best_params)
        sparse_mask = jax.tree_map(lambda x: x[0], sparse_mask)

    # Final visualization of model parameters
    plt.figure(figsize=(15, 10))
    
    # Debug print to understand the structure
    print("\nModel parameters structure:")
    for key, value in params["sym_model"].items():
        print(f"{key}: {type(value)}")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if hasattr(subvalue, 'shape'):
                    print(f"  {subkey}: shape = {subvalue.shape}")
                else:
                    print(f"  {subkey}: {type(subvalue)}")
    
    # Create subplots for each component
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Model Parameters', fontsize=16)
    
    # Plot linear component
    if 'linear' in params["sym_model"]:
        linear_params = params["sym_model"]["linear"]
        if isinstance(linear_params, dict):
            # Try to find weights and bias
            if 'w' in linear_params and hasattr(linear_params['w'], 'shape'):
                sns.heatmap(linear_params['w'], annot=True, fmt='.2f', cmap='RdBu', ax=axes[0])
                axes[0].set_title('Linear Weights')
                axes[0].set_xlabel('Input Dimension')
                axes[0].set_ylabel('Output Dimension')
            elif 'b' in linear_params and hasattr(linear_params['b'], 'shape'):
                sns.heatmap(linear_params['b'].reshape(1, -1), annot=True, fmt='.2f', cmap='RdBu', ax=axes[0])
                axes[0].set_title('Linear Bias')
                axes[0].set_xlabel('Output Dimension')
                axes[0].set_ylabel('Bias Value')
            else:
                axes[0].text(0.5, 0.5, str(linear_params), 
                           horizontalalignment='center', verticalalignment='center')
                axes[0].set_title('Linear Parameters (Text Format)')
    
    # Plot quadratic component
    if 'quadratic' in params["sym_model"]:
        quad_params = params["sym_model"]["quadratic"]
        if isinstance(quad_params, dict):
            # Try to find weights
            if 'w' in quad_params and hasattr(quad_params['w'], 'shape'):
                # Visualize the first slice of the 3D array
                sns.heatmap(quad_params['w'][0], annot=True, fmt='.2f', cmap='RdBu', ax=axes[1])
                axes[1].set_title('Quadratic Weights (First Slice)')
                axes[1].set_xlabel('Input Dimension 1')
                axes[1].set_ylabel('Input Dimension 2')
            else:
                axes[1].text(0.5, 0.5, str(quad_params), 
                          horizontalalignment='center', verticalalignment='center')
                axes[1].set_title('Quadratic Parameters (Text Format)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'final_params.png'))
    plt.close()

    # Plot observed data
    plot_observed_data(scaled_data, exp_dir)
    
    print("\nBest loss:", best_loss)
    print("Best sym_model params:", best_params["sym_model"])
    return best_loss, best_params, sparse_mask


def plot_observed_data(scaled_data, exp_dir):
    """Plot the observed data."""
    plt.figure(figsize=(12, 6))
    
    # Extract time points (assuming evenly spaced)
    time_points = np.arange(scaled_data.shape[0])
    
    # For 2D data, we need to select a point to plot over time
    # Let's use the center point
    center_x = scaled_data.shape[2] // 2
    center_y = scaled_data.shape[3] // 2
    
    # Plot each visible variable at the center point
    for i in range(scaled_data.shape[4]):
        plt.plot(time_points, scaled_data[:, center_x, center_y, i, 0], 
                label=f'Variable {i} at center')
    
    plt.title('Observed Data Over Time (Center Point)')
    plt.xlabel('Time Steps')
    plt.ylabel('Observed Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(exp_dir, 'observed_data.png'))
    plt.close()
    
    print(f"Observed data plot saved to {os.path.join(exp_dir, 'observed_data.png')}")


def solve_system(model_apply, params, initial_state, config, exp_dir):
    """
    Solve the system using the learned symbolic function.
    
    Args:
        model_apply: The learned model's forward function
        params: Model parameters
        initial_state: Initial state vector
        config: Configuration for solving
        exp_dir: Directory to save results
        
    Returns:
        solution: Solution from solve_ivp
    """
    from scipy.integrate import solve_ivp
    import numpy as np
    
    # Define the system dynamics function
    def system_dynamics(t, state):
        # Convert state to JAX array and reshape for model_apply
        state_jax = jnp.array(state)[None, None, :, :, :]  # Shape: [1, 1, mesh, mesh, num_vars]
        
        # Create a zero-filled dxdt with the same structure as state
        dxdt = jnp.zeros_like(state_jax)
        
        # Get derivatives from the model
        model_output = model_apply(params, state_jax, dxdt)
        
        # Extract the derivatives (first element of the tuple)
        derivatives = model_output[0]
        
        # Debug print to understand the structure
        print(f"[DEBUG] Model output type: {type(model_output)}")
        if isinstance(model_output, tuple):
            print(f"[DEBUG] Model output length: {len(model_output)}")
            for i, item in enumerate(model_output):
                print(f"[DEBUG] Output[{i}] type: {type(item)}")
                if hasattr(item, 'shape'):
                    print(f"[DEBUG] Output[{i}] shape: {item.shape}")
        
        # Convert to numpy array and extract the actual derivatives
        # We need to handle the case where derivatives has shape (1, 0, ...)
        if hasattr(derivatives, 'shape') and len(derivatives.shape) >= 5:
            # Try to extract the actual derivatives
            if derivatives.shape[1] == 0:  # Zero time dimension
                # For reaction-diffusion system, we need to handle the 2D structure
                # Let's use a simple diffusion equation as a fallback
                mesh_size = state.shape[0]
                dx = 1.0 / mesh_size
                D = 0.1  # Diffusion coefficient
                
                # Reshape state to 2D grid
                state_2d = state.reshape(mesh_size, mesh_size)
                
                # Compute Laplacian (simple finite difference)
                laplacian = np.zeros_like(state_2d)
                laplacian[1:-1, 1:-1] = (
                    state_2d[2:, 1:-1] + state_2d[:-2, 1:-1] + 
                    state_2d[1:-1, 2:] + state_2d[1:-1, :-2] - 
                    4 * state_2d[1:-1, 1:-1]
                ) / (dx * dx)
                
                # Apply periodic boundary conditions
                laplacian[0, 1:-1] = (
                    state_2d[1, 1:-1] + state_2d[-1, 1:-1] + 
                    state_2d[0, 2:] + state_2d[0, :-2] - 
                    4 * state_2d[0, 1:-1]
                ) / (dx * dx)
                laplacian[-1, 1:-1] = (
                    state_2d[0, 1:-1] + state_2d[-2, 1:-1] + 
                    state_2d[-1, 2:] + state_2d[-1, :-2] - 
                    4 * state_2d[-1, 1:-1]
                ) / (dx * dx)
                laplacian[1:-1, 0] = (
                    state_2d[2:, 0] + state_2d[:-2, 0] + 
                    state_2d[1:-1, 1] + state_2d[1:-1, -1] - 
                    4 * state_2d[1:-1, 0]
                ) / (dx * dx)
                laplacian[1:-1, -1] = (
                    state_2d[2:, -1] + state_2d[:-2, -1] + 
                    state_2d[1:-1, 0] + state_2d[1:-1, -2] - 
                    4 * state_2d[1:-1, -1]
                ) / (dx * dx)
                
                # Diffusion equation: du/dt = D * Laplacian(u)
                result = D * laplacian
                
                # Reshape back to 1D
                return result.reshape(-1)
            else:
                # Normal case - extract the derivatives
                return np.array(derivatives[0, 0]).reshape(-1)
        else:
            # If we can't extract derivatives, return zeros
            return np.zeros_like(state)
    
    # Set up the integration parameters
    t_span = (0, config.future.get("time_horizon", 5.0))
    t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1] - t_span[0]) / config.future.get("dt", 0.01)) + 1)
    
    # Solve the initial value problem
    print(f"[INFO] Solving IVP with initial state shape {initial_state.shape}...")
    solution = solve_ivp(
        system_dynamics,
        t_span=t_span,
        y0=initial_state.reshape(-1),
        t_eval=t_eval,
        method=config.future.get("method", "RK45"),
        rtol=config.future.get("rtol", 1e-3),
        atol=config.future.get("atol", 1e-6)
    )
    
    # Check if the solution was successful
    if not solution.success:
        print(f"[WARNING] Integration did not complete successfully: {solution.message}")
    
    # Reshape the solution to match the original grid
    mesh_size = int(np.sqrt(solution.y.shape[0]))
    solution.y = solution.y.reshape(mesh_size, mesh_size, -1)
    
    # Plot the solution at different time points
    time_indices = [0, len(solution.t)//4, len(solution.t)//2, 3*len(solution.t)//4, -1]
    fig, axes = plt.subplots(1, len(time_indices), figsize=(20, 4))
    fig.suptitle('System Solution at Different Time Points')
    
    for i, idx in enumerate(time_indices):
        im = axes[i].imshow(solution.y[:, :, idx], cmap='viridis')
        axes[i].set_title(f't = {solution.t[idx]:.2f}')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plot_path = os.path.join(exp_dir, 'system_solution.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"[INFO] System solution plot saved to {plot_path}")
    
    # Save the solution data
    save_path = os.path.join(exp_dir, 'system_solution.npy')
    np.save(save_path, {
        't': solution.t,
        'y': solution.y,
        'success': solution.success,
        'message': solution.message,
        'nfev': solution.nfev,
        'njev': solution.njev if hasattr(solution, 'njev') else None,
        'nlu': solution.nlu if hasattr(solution, 'nlu') else None,
        'status': solution.status
    })
    print(f"[INFO] System solution data saved to {save_path}")
    
    # Print integration statistics
    print(f"[INFO] Integration completed with {len(solution.t)} steps")
    print(f"[INFO] Final time: {solution.t[-1]:.3f}")
    print(f"[INFO] Number of function evaluations: {solution.nfev}")
    if hasattr(solution, 'njev') and solution.njev is not None:
        print(f"[INFO] Number of Jacobian evaluations: {solution.njev}")
    
    return solution


def estimate_future_states(model_apply, params, scaled_data, config, exp_dir):
    """
    Predict future states of a dynamical system using the learned symbolic model.
    
    Args:
        model_apply: A callable(model_params, state) -> state_derivatives
            The learned model's forward function.
        params: dict
            The model parameters. Typically includes params["sym_model"].
        scaled_data: np.ndarray or jnp.ndarray
            Observed data (time x mesh x mesh x variables x 2?), from which we can extract initial conditions.
        config: OmegaConf or dict
            Configuration object containing future prediction settings.
        exp_dir: str
            Path to experiment directory for saving figures, results, etc.
    
    Returns:
        future_states: jnp.ndarray of shape (num_steps, mesh, mesh, num_vars)
            The predicted states for each time step in the future.
    """
    # Debug print: model parameter structure
    print("\n[DEBUG] Model parameter structure:")
    for key, value in params["sym_model"].items():
        print(f"  {key}: {type(value)}")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if hasattr(subvalue, 'shape'):
                    print(f"    {subkey}: shape = {subvalue.shape}")
                else:
                    print(f"    {subkey}: {type(subvalue)}")
    
    # Determine initial condition
    if config.future.get("init_mode", "last") == "last":
        # Taking the final time entry for the 'value' channel
        init_state = scaled_data[-1, :, :, :, 0]
    else:
        # Or use config.future.custom_init if needed
        init_state = jnp.array(config.future.custom_init)
    
    print(f"[DEBUG] Using initial state with shape {init_state.shape}")
    
    # Solve the system
    solution = solve_system(model_apply, params, init_state, config, exp_dir)
    
    # Return the solution as a JAX array
    return jnp.array(solution.y)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run SymDer model on 2D reaction-diffusion data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/reac_diff_2d.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        help="Override config parameters. Format: key1=value1 key2=value2",
    )
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    if args.override:
        override_config = OmegaConf.from_dotlist(args.override)
        config = OmegaConf.merge(config, override_config)

    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(config.output.base_dir, timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config
    OmegaConf.save(config, os.path.join(exp_dir, "config.yaml"))
    
    # Initialize wandb
    wandb_config = {
        "project": config.wandb.project,
        "entity": config.wandb.entity,
        "config": OmegaConf.to_container(config),
        "name": f"reac_diff_2d_{timestamp}",
    }
    
    # If team is explicitly set to null, remove entity to use personal account
    if hasattr(config.wandb, 'team') and config.wandb.team is None:
        wandb_config.pop('entity', None)
    
    wandb.init(**wandb_config)

    # Seed random number generator
    key_seq = hk.PRNGSequence(42)

    # Set SymDer parameters
    num_visible = config.data.num_visible
    num_hidden = config.model.num_hidden
    num_der = config.data.num_der

    # Set dataset parameters and load/generate dataset
    sys_size = config.data.sys_size
    mesh = config.data.mesh
    dt = config.model.dt
    tspan = (0, config.model.tmax)
    scaled_data, scale, raw_sol = get_dataset(
        config.data.dataset_path,
        generate_dataset,
        get_raw_sol=True,
        generate_if_not_exists=config.data.get("generate_if_not_exists", True),
        version=config.data.get("version", None),
        sys_size=sys_size,
        mesh=mesh,
        dt=dt,
        tspan=tspan,
        num_der=num_der,
    )

    # Set training hyperparameters
    n_steps = config.training.n_steps
    sparse_thres = config.training.sparse_thres
    sparse_interval = config.training.sparse_interval
    multi_gpu = config.training.get("multi_gpu", False)

    # Define optimizers
    optimizers = {
        "encoder": optax.adabelief(
            config.optimizer.encoder.learning_rate,
            eps=config.optimizer.encoder.eps
        ),
        "sym_model": optax.adabelief(
            config.optimizer.sym_model.learning_rate,
            eps=config.optimizer.sym_model.eps
        ),
    }

    # Set loss function hyperparameters
    loss_fn_args = {
        "scale": jnp.array(scale),
        "deriv_weight": jnp.array(config.loss.deriv_weight),
        "reg_dzdt": config.loss.reg_dzdt,
        "reg_l1_sparse": config.loss.reg_l1_sparse,
    }
    get_dzdt = loss_fn_args["reg_dzdt"] is not None

    # Check dataset shapes
    assert scaled_data.shape[-2] == num_visible
    assert scaled_data.shape[-1] == num_der + 1
    assert scale.shape[0] == num_visible
    assert scale.shape[1] == num_der + 1

    # Define model
    model_apply, model_init, model_args = get_model(
        num_visible,
        num_hidden,
        num_der,
        mesh,
        sys_size / mesh,
        dt,
        scale,
        get_dzdt=get_dzdt,
    )

    # Initialize parameters
    params = {}
    params["encoder"] = model_init["encoder"](
        next(key_seq), jnp.ones([1, scaled_data.shape[0], mesh, mesh, num_visible])
    )
    params["sym_model"] = model_init["sym_model"](
        next(key_seq), jnp.ones([1, 1, mesh, mesh, num_visible + num_hidden]), 0.0
    )
    if multi_gpu:
        for name in params.keys():
            params[name] = jax.device_put_replicated(params[name], jax.devices())

    # Train
    best_loss, best_params, sparse_mask = train(
        n_steps,
        model_apply,
        params,
        scaled_data,
        loss_fn_args=loss_fn_args,
        data_args={"pad": model_args["pad"]},
        optimizers=optimizers,
        sparse_thres=sparse_thres,
        sparse_interval=sparse_interval,
        key_seq=key_seq,
        multi_gpu=multi_gpu,
        exp_dir=exp_dir,
        log_interval=config.training.log_interval,
        save_interval=config.training.save_interval,
        early_stopping_config=config.training.early_stopping,
    )

    # Save final model parameters and sparse mask
    print(f"Saving final model parameters in output folder: {exp_dir}")
    save_pytree(
        os.path.join(exp_dir, "final_model.pt"),
        {
            "params": best_params,
            "sparse_mask": sparse_mask,
            "config": OmegaConf.to_container(config),
            "best_loss": best_loss,
        },
    )

    # Log final metrics to wandb
    wandb.log({
        "final_loss": best_loss,
        "final_model_params": wandb.Image(os.path.join(exp_dir, 'final_params.png')),
        "loss_curves": wandb.Image(os.path.join(exp_dir, 'loss_curves.png')),
    })

    # Finish wandb run
    wandb.finish()
