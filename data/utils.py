# import jax.numpy as jnp
from jax import lax
import numpy as np

import os.path
from tqdm.auto import tqdm

__all__ = ["solve_ETDRK4", "generate_diff_kernels", "save_dataset", "load_dataset"]


def solve_ETDRK4(L, N, v0, tspan, dt, output_func):
    """ETDRK4 method"""
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)

    contour_radius = 1
    M = 16
    r = contour_radius * np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)

    LR = dt * L
    LR = np.expand_dims(LR, axis=-1) + r

    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1) / LR, axis=-1))
    f1 = dt * np.real(
        np.mean(
            (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3, axis=-1
        )
    )
    f2 = dt * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3, axis=-1))
    f3 = dt * np.real(
        np.mean(
            (-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3, axis=-1
        )
    )

    u = []
    v = v0
    for t in tqdm(np.arange(tspan[0], tspan[1], dt)):
        u.append(output_func(v))

        Nv = N(v)
        a = E2 * v + Q * Nv
        Na = N(a)
        b = E2 * v + Q * Na
        Nb = N(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = N(c)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3

    return np.stack(u)


def generate_diff_kernels(order):
    p = int(np.floor((order + 1) / 2))

    rev_d1 = np.array((0.5, 0.0, -0.5))
    d2 = np.array((1.0, -2.0, 1.0))

    even_kernels = [np.pad(np.array((1.0,)), (p,))]
    for i in range(order // 2):
        even_kernels.append(np.convolve(even_kernels[-1], d2, mode="same"))

    even_kernels = np.stack(even_kernels)
    odd_kernels = lax.conv(
        even_kernels[:, None, :], rev_d1[None, None, :], (1,), "SAME"
    ).squeeze(1)

    kernels = np.stack((even_kernels, odd_kernels), axis=1).reshape(-1, 2 * p + 1)
    if order % 2 == 0:
        kernels = kernels[:-1]

    return kernels


def get_dataset(filename, generate_dataset, get_raw_sol=False, generate_if_not_exists=True, version=None, **gen_kwargs):
    """
    Get a dataset, either by loading from file or generating it.
    
    Args:
        filename: Path to the dataset file
        generate_dataset: Function to generate the dataset
        get_raw_sol: Whether to return the raw solution
        generate_if_not_exists: Whether to generate the dataset if it doesn't exist
        version: Version tag for the dataset
        **gen_kwargs: Additional arguments to pass to generate_dataset
        
    Returns:
        Tuple of (scaled_data, scale, raw_sol) if get_raw_sol is True,
        otherwise (scaled_data, scale)
    """
    # Add version to filename if provided
    if version:
        base, ext = os.path.splitext(filename)
        versioned_filename = f"{base}_{version}{ext}"
    else:
        versioned_filename = filename
    
    # Check if the dataset exists
    if os.path.isfile(versioned_filename):
        print(f"Dataset found at {versioned_filename}")
        scaled_data, scale, loaded_gen_kwargs, raw_sol = load_dataset(
            versioned_filename, get_raw_sol
        )
        
        # Check if the loaded dataset matches the requested parameters
        # We'll do a simple check for the most important parameters
        for key in ['dt', 'tmax', 'num_visible', 'visible_vars', 'num_der']:
            if key in gen_kwargs and key in loaded_gen_kwargs:
                if gen_kwargs[key] != loaded_gen_kwargs[key]:
                    print(f"Warning: Parameter mismatch for {key}. Requested: {gen_kwargs[key]}, Loaded: {loaded_gen_kwargs[key]}")
                    if generate_if_not_exists:
                        print(f"Generating new dataset with requested parameters...")
                        scaled_data, scale, raw_sol = generate_dataset(
                            raw_sol=get_raw_sol, **gen_kwargs
                        )
                        save_dataset(versioned_filename, scaled_data, scale, gen_kwargs, raw_sol)
                        break
    else:
        if generate_if_not_exists:
            print(f"Dataset not found at {versioned_filename}. Generating new dataset...")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(versioned_filename), exist_ok=True)
            
            scaled_data, scale, raw_sol = generate_dataset(
                raw_sol=get_raw_sol, **gen_kwargs
            )
            save_dataset(versioned_filename, scaled_data, scale, gen_kwargs, raw_sol)
        else:
            raise FileNotFoundError(f"Dataset not found at {versioned_filename} and generate_if_not_exists is False.")
    
    return (scaled_data, scale, raw_sol) if get_raw_sol else (scaled_data, scale)


def save_dataset(filename, scaled_data, scale, gen_kwargs, raw_sol=None):
    """
    Save a dataset to a file.
    
    Args:
        filename: Path to save the dataset
        scaled_data: The scaled data
        scale: The scale factors
        gen_kwargs: The generation parameters
        raw_sol: The raw solution (optional)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    print(f"Saving dataset to file: {filename}")
    np.savez(
        filename,
        scaled_data=scaled_data,
        scale=scale,
        gen_kwargs=gen_kwargs,
        raw_sol=raw_sol,
    )
    print(f"Dataset saved successfully.")


def load_dataset(filename, get_raw_sol=False):
    """
    Load a dataset from a file.
    
    Args:
        filename: Path to the dataset file
        get_raw_sol: Whether to return the raw solution
        
    Returns:
        Tuple of (scaled_data, scale, gen_kwargs, raw_sol)
    """
    print(f"Loading dataset from file: {filename}")
    dataset = np.load(filename, allow_pickle=True)
    return (
        dataset["scaled_data"],
        dataset["scale"],
        dataset["gen_kwargs"].item(),  # Convert to dict
        dataset["raw_sol"] if get_raw_sol and "raw_sol" in dataset else None,
    )
