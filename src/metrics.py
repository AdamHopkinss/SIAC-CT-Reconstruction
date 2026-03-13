import numpy as np
import pandas as pd

from skimage.metrics import structural_similarity as ssim_metric



def removed_energy(x: np.ndarray, y: np.ndarray):
    """
    Energy removed by an operator S, defined as r = x - y.

    Parameters
    ----------
    x : ndarray
        Original image (e.g. BP or FBP).
    y : ndarray
        Processed image (e.g. SIAC(x)).

    Returns
    -------
    Erem : float
        Absolute removed energy ||x - y||_2^2.
    Erel : float
        Relative removed energy ||x - y||_2^2 / ||x||_2^2.
    """
    r = x - y
    Erem = np.sum(r**2)
    Erel = Erem / np.sum(x**2)
    return Erem, Erel


def rel_l2_err(x: np.ndarray, xtrue: np.ndarray):
    """
    Relative L2 error with respect to ground truth.

    ||x - xtrue||_2 / ||xtrue||_2
    """
    return np.linalg.norm(x - xtrue) / np.linalg.norm(xtrue)


def gradient_error(x: np.ndarray, xtrue: np.ndarray, dx: float, dy: float):
    """
    Relative L2 error of the gradient (H1-seminorm error).

    ||grad(x) - grad(xtrue)||_2 / ||grad(xtrue)||_2
    """
    # Gradients (order: dy, dx)
    gx_y, gx_x = np.gradient(x, dy, dx)
    gt_y, gt_x = np.gradient(xtrue, dy, dx)

    # Gradient difference
    diff_sq = (gx_x - gt_x)**2 + (gx_y - gt_y)**2
    true_sq = gt_x**2 + gt_y**2

    num = np.sqrt(np.sum(diff_sq))
    den = np.sqrt(np.sum(true_sq))

    return num / den


import numpy as np

def highfreq_removed_energy(x: np.ndarray,
                            y: np.ndarray,
                            dx: float,
                            dy: float,
                            frac: float = 0.6,
                            use_fftshift: bool = True):
    """
    Energy removed in the *high-frequency* band.

    Parameters
    ----------
    x : ndarray (Ny, Nx)
        Original image.
    y : ndarray (Ny, Nx)
        Processed image.
    dx, dy : float
        Grid spacing in x and y.
    frac : float
        Cutoff as a fraction of the Nyquist radius in frequency space.
        frac=0.6 means "frequencies with radius > 0.6 * Nyquist_radius".
        Must be in (0, 1).
    use_fftshift : bool
        If True, compute with centered frequency grids (easier to reason about).

    Returns
    -------
    Erem_hf : float
        Absolute removed high-frequency energy = sum_{HF} |Rhat|^2.
    Erem_hf_rel_total : float
        Removed high-frequency energy relative to total energy in x: Erem_hf / sum |Xhat|^2.
    Erem_hf_rel_hf : float
        Removed high-frequency energy relative to original high-frequency energy in x:
        Erem_hf / sum_{HF} |Xhat|^2.
    """
    if not (0.0 < frac < 1.0):
        raise ValueError("frac must be in (0, 1).")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    r = x - y
    Ny, Nx = x.shape

    # Fourier transforms
    X = np.fft.fft2(x)
    R = np.fft.fft2(r)

    # Frequency grids in cycles per unit length
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)

    if use_fftshift:
        X = np.fft.fftshift(X)
        R = np.fft.fftshift(R)
        fx = np.fft.fftshift(fx)
        fy = np.fft.fftshift(fy)

    FX, FY = np.meshgrid(fx, fy)  # shapes (Ny, Nx)
    fr = np.sqrt(FX**2 + FY**2)

    # Nyquist radius (cycles per unit length)
    nyq_x = 0.5 / dx
    nyq_y = 0.5 / dy
    nyq_r = np.sqrt(nyq_x**2 + nyq_y**2)

    cutoff = frac * nyq_r
    mask_hf = fr >= cutoff

    # Energies (Parseval: scaling cancels out in ratios, so we keep plain sums)
    Erem_hf = np.sum(np.abs(R[mask_hf])**2)
    Ex_total = np.sum(np.abs(X)**2)
    Ex_hf = np.sum(np.abs(X[mask_hf])**2)

    Erem_hf_rel_total = Erem_hf / Ex_total if Ex_total != 0 else np.nan
    Erem_hf_rel_hf = Erem_hf / Ex_hf if Ex_hf != 0 else np.nan

    return Erem_hf, Erem_hf_rel_total, Erem_hf_rel_hf


def ssim(x: np.ndarray, xtrue: np.ndarray, data_range=None):
    """
    Structural Similarity Index (SSIM).

    Parameters
    ----------
    x : ndarray
        Reconstructed image.
    xtrue : ndarray
        Ground truth image.
    data_range : float or None
        If None, inferred from xtrue.

    Returns
    -------
    float
        SSIM index in [-1, 1], typically [0, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    xtrue = np.asarray(xtrue, dtype=np.float64)

    if data_range is None:
        data_range = xtrue.max() - xtrue.min()

    return ssim_metric(
        xtrue, x,
        data_range=data_range,
        channel_axis=None
    )

def gradient_weighted_ssim(
    image: np.ndarray,
    truth: np.ndarray,
    data_range=None,
    alpha: float = 0.75,
    sigma: float = 1.0,
    eps: float = 1e-12,
    return_maps: bool = False,
):
    """
    Gradient-weighted SSIM using gradient magnitude of the ground truth.

    If return_maps=True, also returns (ssim_map, weight_map).
    """
    image = np.asarray(image, dtype=float)
    truth = np.asarray(truth, dtype=float)
    
    if image.shape != truth.shape:
        raise ValueError("image and truth must have same shape")

    if data_range is None:
        data_range = float(truth.max() - truth.min())
        if data_range == 0:
            data_range = 1.0

    # SSIM map
    _, ssim_map = ssim_metric(image, truth, data_range=data_range, full=True)

    # Ground-truth gradient weights
    if sigma is not None and sigma > 0:
        from scipy.ndimage import gaussian_filter
        truth_for_grad = gaussian_filter(truth, sigma=sigma)
    else:
        truth_for_grad = truth

    gy, gx = np.gradient(truth_for_grad)
    weights = np.sqrt(gx**2 + gy**2) ** alpha

    wsum = weights.sum()
    if wsum < eps:
        gw_ssim = float(np.mean(ssim_map))
        if return_maps:
            return gw_ssim, ssim_map, weights
        return gw_ssim

    weights = weights / (wsum + eps)
    gw_ssim = float(np.sum(weights * ssim_map))

    if return_maps:
        return gw_ssim, ssim_map, weights

    return gw_ssim


def eval_metrics(
    image: np.ndarray,
    truth: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    dx: float | None = None,
    dy: float | None = None,
    hf_frac: float = 0.6,
    data_range=None,
    extra: dict | None = None,
):
    """
    Evaluate metrics for one image.

    Parameters
    ----------
    image : ndarray
        Main image to evaluate.
    truth : ndarray or None
        Ground-truth image. If given, compute truth-based metrics.
    reference : ndarray or None
        Optional baseline/reference image. If given, compute reference-based metrics.
    dx, dy : float or None
        Grid spacings. Needed for gradient_error and highfreq_removed_energy.
    hf_frac : float
        High-frequency cutoff fraction for highfreq_removed_energy.
    data_range : float or None
        Passed to SSIM. If None, inferred from truth when truth is given.
    extra : dict or None
        Optional metadata to include in the returned dictionary.

    Returns
    -------
    results : dict
        Dictionary of metrics and optional metadata.
    """
    image = np.asarray(image, dtype=float)
    results = {}

    if truth is not None:
        truth = np.asarray(truth, dtype=float)

        results["truth_rel_l2_err"] = rel_l2_err(image, truth)
        results["truth_ssim"] = ssim(image, truth, data_range=data_range)
        results["truth_gw_ssim"] = gradient_weighted_ssim(
            image, truth, data_range=data_range
            )
        
        if dx is not None and dy is not None:
            results["truth_gradient_error"] = gradient_error(image, truth, dx=dx, dy=dy)
        else:
            results["truth_gradient_error"] = np.nan

    if reference is not None:
        reference = np.asarray(reference, dtype=float)

        # Plain relative difference to reference
        results["ref_rel_l2_err"] = rel_l2_err(image, reference)

        # Energy removed relative to reference
        # removed_energy(x, y) interprets x as original and y as processed
        Erem, Erel = removed_energy(reference, image)
        results["ref_removed_energy_rel"] = Erel

        if dx is not None and dy is not None:
            Ehf, Ehf_rel_total, Ehf_rel_hf = highfreq_removed_energy(
                reference, image, dx=dx, dy=dy, frac=hf_frac
            )
            results["ref_hf_removed_energy_rel_hf"] = Ehf_rel_hf
        else:
            results["ref_hf_removed_energy_rel_hf"] = np.nan

    if extra is not None:
        results.update(extra)

    return results


def build_metrics_table(
    cases: dict,
    truth: np.ndarray | None = None,
    dx: float | None = None,
    dy: float | None = None,
    hf_frac: float = 0.6,
    data_range=None,
):
    """
    Build a metrics table for multiple named cases.

    Parameters
    ----------
    cases : dict
        Dictionary of cases. Each case should look like

        cases = {
            "FBP-ramp": {
                "image": ...,
            },
            "FBP-cosine": {
                "image": ...,
                "reference": ...,
            },
            "SIAC(FBP-ramp)": {
                "image": ...,
                "reference": ...,
                "extra": {"method": "SIAC", "filter": "ramp"}
            },
            ...
        }

        Optional per-case keys:
        - "truth": overrides global truth
        - "reference": optional reference image
        - "extra": dict of metadata columns to include

    truth : ndarray or None
        Global ground truth used unless overridden per case.
    dx, dy : float or None
        Grid spacings.
    hf_frac : float
        High-frequency cutoff fraction.
    data_range : float or None
        Passed to SSIM.

    Returns
    -------
    df : pandas.DataFrame
        Table with one row per case.
    """
    rows = []

    for name, case in cases.items():
        if "image" not in case:
            raise ValueError(f"Case '{name}' is missing required key 'image'.")

        case_image = case["image"]
        case_truth = case.get("truth", truth)
        case_reference = case.get("reference", None)
        case_extra = dict(case.get("extra", {}))
        case_extra["name"] = name

        row = eval_metrics(
            image=case_image,
            truth=case_truth,
            reference=case_reference,
            dx=dx,
            dy=dy,
            hf_frac=hf_frac,
            data_range=data_range,
            extra=case_extra,
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    # Put name first if present
    cols = list(df.columns)
    if "name" in cols:
        cols = ["name"] + [c for c in cols if c != "name"]
        df = df[cols]

    return df