import numpy as np

def random_modal_coeffs_1d(K, p, seed=0, scale=1.0):
    """
    Random modal coefficients for K elements in 1D.
    Shape: (K, p+1)
    """
    rng = np.random.default_rng(seed)
    return scale * rng.standard_normal((K, p + 1))

