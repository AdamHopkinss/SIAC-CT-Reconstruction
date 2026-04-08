from itertools import product
import pandas as pd
import numpy as np
from tqdm import tqdm

def run_monte_carlo_study(
    method_fn,
    method_name,
    method_param_grid,
    noise_levels,
    n_reps,
    input_generator,
    metric_fn,
    reference,
    base_seed=0,
):
    """
    Run a Monte Carlo study for one reconstruction/postprocessing method.

    Parameters
    ----------
    method_fn : callable
        Function of the form
            recon = method_fn(input_data, **method_params)
        returning a reconstructed/postprocessed image as a NumPy array.

    method_name : str
        Name of the method, stored in the output table.

    method_param_grid : dict
        Dictionary mapping parameter names to lists of values. All combinations
        are evaluated.

    noise_levels : sequence of float
        Noise levels to test.

    n_reps : int
        Number of Monte Carlo repetitions per noise level and parameter setting.

    input_generator : callable
        Function of the form
            input_data = input_generator(noise_level, seed)
        returning the noisy input expected by `method_fn`.

    metric_fn : callable
        Function of the form
            metrics = metric_fn(recon, reference)
        returning a dictionary of scalar metrics.

    reference : numpy.ndarray
        Ground-truth image used for evaluation.

    base_seed : int, optional
        Base seed used to generate repetition-specific seeds.

    Returns
    -------
    results_df : pandas.DataFrame
        Long-form table with one row per Monte Carlo run.
    """
    param_names = list(method_param_grid.keys())
    param_lists = [method_param_grid[name] for name in param_names]
    param_combinations = list(product(*param_lists))

    rows = []

    for noise_level in tqdm(noise_levels, desc="Noise levels"):
        for rep in range(n_reps):
            seed = base_seed + rep
            input_data = input_generator(noise_level=noise_level, seed=seed)

            for values in param_combinations:
                params = dict(zip(param_names, values))

                recon = method_fn(input_data, **params)
                metrics = metric_fn(recon, reference)

                row = {
                    "method": method_name,
                    "noise_level": noise_level,
                    "rep": rep,
                    "seed": seed,
                    **params,
                    **metrics,
                }
                rows.append(row)

    return pd.DataFrame(rows)