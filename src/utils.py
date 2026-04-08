import numpy as np
import warnings

def _to_numpy(recon):
    """Accept ODL element or numpy array."""
    if hasattr(recon, "asarray"):
        return recon.asarray()
    return np.asarray(recon)

def resolve_degree(DOF_x: int, DOF_y: int, deg: int) -> tuple[int, int, int]:
    """
    Resolve an admissible polynomial degree p and element counts Kx, Ky such that

        DOF_x = (p + 1) * Kx
        DOF_y = (p + 1) * Ky
    """
    if not isinstance(DOF_x, int) or not isinstance(DOF_y, int) or not isinstance(deg, int):
        raise TypeError("DOF_x, DOF_y, and deg must all be integers.")
    if DOF_x <= 0 or DOF_y <= 0:
        raise ValueError("DOF_x and DOF_y must be positive.")
    if deg < 0:
        raise ValueError("deg must be nonnegative.")

    requested_deg = deg
    max_p = min(DOF_x, DOF_y) - 1

    for p in range(deg, max_p + 1):
        order = p + 1
        if DOF_x % order == 0 and DOF_y % order == 0:
            Kx = DOF_x // order
            Ky = DOF_y // order

            if p != requested_deg:
                warnings.warn(
                    f"Requested degree deg={requested_deg} is not admissible for "
                    f"(DOF_x, DOF_y)=({DOF_x}, {DOF_y}). Using p={p} instead, "
                    f"giving (Kx, Ky)=({Kx}, {Ky}).",
                    stacklevel=2
                )
            return p, Ky, Kx

    raise ValueError(
        f"No admissible polynomial degree p >= {deg} found such that "
        f"DOF_x = (p+1)Kx and DOF_y = (p+1)Ky for "
        f"(DOF_x, DOF_y)=({DOF_x}, {DOF_y})."
    )
    
    

# Create a function to return the ODL Fourier filters
def fbp_filters(norm_freq, filter_type, frequency_scaling):
    filter_type_in = filter_type    # Add this for error message
    if callable(filter_type):
        filt = filter_type(norm_freq)
    else:   # add else here for string-based call (e.g. "hann")
        filter_type = str(filter_type).lower()
        if filter_type == 'ram-lak':
            filt = np.copy(norm_freq)
        elif filter_type == 'shepp-logan':
            filt = norm_freq * np.sinc(norm_freq / (2 * frequency_scaling))
        elif filter_type == 'cosine':
            filt = norm_freq * np.cos(norm_freq * np.pi / (2 * frequency_scaling))
        elif filter_type == 'hamming':
            filt = norm_freq * (
                0.54 + 0.46 * np.cos(norm_freq * np.pi / (frequency_scaling)))
        elif filter_type == 'hann':
            filt = norm_freq * (
                np.cos(norm_freq * np.pi / (2 * frequency_scaling)) ** 2)
        else:
            raise ValueError('unknown `filter_type` ({})'
                            ''.format(filter_type_in))

    indicator = (norm_freq <= frequency_scaling)
    filt *= indicator
    return filt
    

import contextlib, os

def _silent_call(fn, *args, **kwargs):
    with open(os.devnull, "w") as fnull:
        with contextlib.redirect_stdout(fnull):
            return fn(*args, **kwargs)