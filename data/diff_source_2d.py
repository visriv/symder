from jax import lax
import numpy as np
from numpy.fft import fftfreq, fft2, ifft2
from sklearn.preprocessing import StandardScaler

from .utils import solve_ETDRK4, generate_diff_kernels

__all__ = ["generate_dataset"]


def generate_dataset(
    sys_size=64, mesh=64, dt=5e-2, tspan=None, num_der=2, seed=0, raw_sol=False,
):
    if tspan is None:
        tspan = (0, 50 + 2 * dt)

    kx = np.expand_dims(2 * np.pi * fftfreq(mesh, d=sys_size / mesh), axis=-1)
    ky = np.expand_dims(2 * np.pi * fftfreq(mesh, d=sys_size / mesh), axis=0)

    # Initial condition
    np.random.seed(seed)
    krange = 1
    envelope = np.exp(-1 / (2 * krange ** 2) * (kx ** 2 + ky ** 2))
    v0 = envelope * (
        np.random.normal(loc=0, scale=1.0, size=(2, mesh, mesh))
        + 1j * np.random.normal(loc=0, scale=1.0, size=(2, mesh, mesh))
    )
    u0 = np.real(ifft2(v0))
    # normalize
    u0 = u0 / np.max(np.abs(u0), axis=(-2, -1), keepdims=True)

    n_rects = 50
    u0[1] = np.zeros((1, mesh, mesh))
    rect_pos = (
        np.random.uniform(0, sys_size, size=(n_rects, 2)) * mesh / sys_size
    ).astype(int)
    rect_size = (
        np.random.uniform(0, 0.05 * sys_size, size=(n_rects, 2)) * mesh / sys_size
    ).astype(int)
    rect_value = np.random.uniform(0, 0.2, size=(n_rects,))
    for i in range(n_rects):
        rect = np.zeros((mesh, mesh), dtype=bool)
        rect[: rect_size[i, 0], : rect_size[i, 1]] = True
        rect = np.roll(np.roll(rect, rect_pos[i, 0], axis=0), rect_pos[i, 1], axis=1)
        u0[1, :, :] = u0[1, :, :] * (1 - rect) + rect_value[i] * rect

    # Differential equation definition
    D2 = -(kx ** 2 + ky ** 2)
    L = np.stack((0.2 * D2, np.zeros_like(D2)))

    def N(v):
        v2 = v[..., 1, :, :]
        dv = np.stack([1 * v2, -0.1 * v2], axis=-3)
        return dv

    # Solve using ETDRK4 method
    print("Generating 2D diffusion with source dataset...")
    sol_u = solve_ETDRK4(L, N, fft2(u0), tspan, dt, lambda v: np.real(ifft2(v)))
    data = sol_u[:, 0].reshape(sol_u.shape[0], 1 * mesh ** 2)
    data = data.T

    # Compute finite difference derivatives
    kernels = generate_diff_kernels(num_der)
    data = lax.conv(data[:, None, :], kernels[:, None, :], (1,), "VALID")
    # time, mesh**2, num_visible, num_der+1
    data = data[None, ...].transpose((3, 1, 0, 2))

    # Rescale/normalize data
    reshaped_data = data.reshape(-1, data.shape[2] * data.shape[3])
    scaler = StandardScaler(with_mean=False)
    scaler.fit(reshaped_data)
    # scaler.scale_ /= scaler.scale_[0]
    scaled_data = scaler.transform(reshaped_data)
    # time, mesh, mesh, num_visible, num_der+1
    scaled_data = scaled_data.reshape(-1, mesh, mesh, 1, num_der + 1)

    return (
        scaled_data,
        scaler.scale_.reshape(1, num_der + 1),
        sol_u if raw_sol else None,
    )
