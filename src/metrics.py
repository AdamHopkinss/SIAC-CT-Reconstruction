import numpy as np
import pandas as pd

from skimage.metrics import structural_similarity as ssim_metric
from scipy.ndimage import (
    binary_fill_holes,
    binary_dilation,
    label,
    generate_binary_structure,
)



def rel_l2_err(x: np.ndarray, xtrue: np.ndarray):
    """
    Relative L2 error with respect to ground truth.

    ||x - xtrue||_2 / ||xtrue||_2
    """
    return np.linalg.norm(x - xtrue) / np.linalg.norm(xtrue)


def phantom_support_mask(truth, tau=1e-8, pad_pixels=2, keep_largest=True):
    """
    Build a binary mask for the compact support of the phantom, including
    interior zero-valued regions, and optionally dilate it by `pad_pixels`.

    Parameters
    ----------
    truth : np.ndarray
        Ground-truth phantom image.
    tau : float
        Threshold for deciding initial nonzero support.
    pad_pixels : int
        Number of pixel dilations to apply to expand the support.
    keep_largest : bool
        If True, keep only the largest connected component before filling/dilation.

    Returns
    -------
    mask : np.ndarray of bool
        Support mask including holes/interior and padding.
    """
    truth = np.asarray(truth, dtype=float)

    # Initial support from nonzero values
    mask0 = np.abs(truth) > tau

    # Optionally keep only the largest connected component
    if keep_largest:
        structure = generate_binary_structure(mask0.ndim, 1)
        lbl, ncomp = label(mask0, structure=structure)
        if ncomp > 0:
            sizes = np.bincount(lbl.ravel())
            sizes[0] = 0  # background
            largest = sizes.argmax()
            mask0 = (lbl == largest)

    # Fill interior holes so zero-valued regions inside the phantom are included
    mask = binary_fill_holes(mask0)

    # Dilate outward by a few pixels
    if pad_pixels > 0:
        structure = generate_binary_structure(mask.ndim, 1)
        mask = binary_dilation(mask, structure=structure, iterations=pad_pixels)

    return mask

def masked_rel_l2_err(
    x: np.ndarray,
    xtrue: np.ndarray,
    mask: np.ndarray | None = None,
    eps: float = 1e-14,
):
    """
    Masked relative L2 error with respect to the ground truth.

    Computes

        sqrt( sum(M_ij (x_ij - xtrue_ij)^2) / sum(M_ij (xtrue_ij)^2) )

    where M is a binary mask selecting the phantom support.
    """
    image = np.asarray(x, dtype=float)
    truth = np.asarray(xtrue, dtype=float)

    if image.shape != truth.shape:
        raise ValueError("x and xtrue must have the same shape")

    if mask is None:
        mask = phantom_support_mask(truth, tau=1e-8, pad_pixels=2)
    else:
        mask = np.asarray(mask, dtype=bool)

    if mask.shape != truth.shape:
        raise ValueError("mask must have the same shape as x and xtrue")

    num = np.sum(((image - truth) ** 2)[mask])
    den = np.sum((truth ** 2)[mask])

    return np.sqrt(num / (den + eps))
    

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


def eval_metrics(
    image: np.ndarray,
    truth: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    dx: float | None = None,
    dy: float | None = None,
    hf_frac: float = 0.6,
    data_range=None,
    clip_range=None,
    extra: dict | None = None,
    compute_reference_metrics: bool = True,
    compute_masked_rel_l2: bool = True,
    mask: np.ndarray | None = None,
    mask_pad_pixels: int = 2,
    mask_tau: float = 1e-8,
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
        Optional baseline/reference image. If given and
        compute_reference_metrics=True, compute reference-based metrics.
    dx, dy : float or None
        Grid spacings. Needed for gradient_error and highfreq_removed_energy.
    hf_frac : float
        High-frequency cutoff fraction for highfreq_removed_energy.
    data_range : float or None
        Passed to SSIM. If None, inferred from truth when truth is given.
    clip_range : tuple or None
        Optionally clips the image to e.g. (0, 1).
    extra : dict or None
        Optional metadata to include in the returned dictionary.
    compute_reference_metrics : bool
        If True, compute metrics against `reference` when reference is provided.
    compute_masked_rel_l2 : bool
        If True, compute masked relative L2 error against truth.
    mask : ndarray or None
        Optional precomputed phantom support mask. If None and
        compute_masked_rel_l2=True, it is built from truth.
    mask_pad_pixels : int
        Padding used when building the phantom support mask.
    mask_tau : float
        Threshold used when building the phantom support mask.

    Returns
    -------
    results : dict
        Dictionary of metrics and optional metadata.
    """
    image = np.asarray(image, dtype=float)

    if clip_range is not None:
        image = np.clip(image, clip_range[0], clip_range[1])

    results = {}

    # -------------------------
    # Truth-based metrics
    # -------------------------
    if truth is not None:
        truth = np.asarray(truth, dtype=float)

        results["rel_l2_err"] = rel_l2_err(image, truth)
        results["ssim"] = ssim(image, truth, data_range=data_range)
        results["gw_ssim"] = gradient_weighted_ssim(
            image, truth, data_range=data_range
        )

        if compute_masked_rel_l2:
            if mask is None:
                mask_use = phantom_support_mask(
                    truth, tau=mask_tau, pad_pixels=mask_pad_pixels
                )
            else:
                mask_use = np.asarray(mask, dtype=bool)

            results["masked_rel_l2_err"] = masked_rel_l2_err(
                image, truth, mask=mask_use
            )
        else:
            results["masked_rel_l2_err"] = np.nan

        if dx is not None and dy is not None:
            results["gradient_error"] = gradient_error(
                image, truth, dx=dx, dy=dy
            )
        else:
            results["gradient_error"] = np.nan

    # -------------------------
    # Reference-based metrics
    # -------------------------
    if compute_reference_metrics and (reference is not None):
        reference = np.asarray(reference, dtype=float)

        results["ref_rel_l2_err"] = rel_l2_err(image, reference)

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
    compute_reference_metrics: bool = True,
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
    compute_reference_metrics : bool
        If True, compute reference-based metrics when reference is available.

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
            compute_reference_metrics=compute_reference_metrics,
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    # Put name first if present
    cols = list(df.columns)
    if "name" in cols:
        cols = ["name"] + [c for c in cols if c != "name"]
        df = df[cols]

    return df


### Metric functions for the Monte Carlo simulation ###

def compute_metrics(recon, reference):
    
    rel_l2 = rel_l2_err(x=recon, xtrue=reference)
    masked_rel_l2 = masked_rel_l2_err(x=recon, xtrue=reference)
    gw_ssim_val = gradient_weighted_ssim(image=recon, truth=reference)
    ssim_val = ssim(x=recon, xtrue=reference)
    
    metric_dict = {
        "rel_l2_err": rel_l2,
        "masked_rel_l2_err": masked_rel_l2, 
        "gw_ssim": gw_ssim_val, 
        "ssim": ssim_val 
    }
    return metric_dict
    

def summarize_mc_results(
    results_df,
    group_cols=None,
    metric_cols=None,
):
    """
    Summarize Monte Carlo results by computing mean/std for each metric.
    """
    if group_cols is None:
        candidate_group_cols = ["family", "method", "noise_level", "p", "moments", "BSorder"]
        group_cols = [c for c in candidate_group_cols if c in results_df.columns]

    if metric_cols is None:
        exclude = set(group_cols + ["rep", "seed"])
        metric_cols = [
            c for c in results_df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(results_df[c])
        ]

    agg_dict = {}
    for col in metric_cols:
        agg_dict[col] = ["mean", "std", "min", "max"]

    summary = results_df.groupby(group_cols, dropna=False).agg(agg_dict)

    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns.to_flat_index()
    ]
    summary = summary.reset_index()

    return summary

def select_best_by_noise(
    summary_df,
    metric,
    method_col="method",
    noise_col="noise_level",
    minimize=True,
    exclude_families=None,
):
    """
    Select the best parameter combination for each method and noise level.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Summarized MC dataframe.
    metric : str
        Base metric name, e.g. "rel_l2_err" or "ssim".
    method_col : str
        Method column name.
    noise_col : str
        Noise-level column name.
    minimize : bool
        If True, selects the smallest mean value.
        If False, selects the largest mean value.
    exclude_families : sequence[str] or None
        Optional list/tuple of families to exclude before selection.

    Returns
    -------
    best_df : pandas.DataFrame
        One row per method/noise level with the best parameter combination.
    """
    mean_col = f"{metric}_mean"
    if mean_col not in summary_df.columns:
        raise ValueError(f"Column '{mean_col}' not found in summary_df.")

    df = summary_df.copy()

    if exclude_families is not None and "family" in df.columns:
        df = df[~df["family"].isin(exclude_families)]

    grouped = df.groupby([method_col, noise_col], dropna=False)

    idx = grouped[mean_col].idxmin() if minimize else grouped[mean_col].idxmax()
    best_df = df.loc[idx].sort_values([method_col, noise_col]).reset_index(drop=True)

    return best_df

def select_fixed_params_from_reference_noise(
    summary_df,
    metric,
    reference_noise=0.10,
    method_col="method",
    noise_col="noise_level",
    minimize=True,
    exclude_families=None,
):
    """
    Select one fixed parameter setting per method by taking the best
    configuration at a chosen reference noise level.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Summarized MC dataframe.
    metric : str
        Base metric name.
    reference_noise : float
        Noise level at which to choose the fixed parameter set.
    method_col : str
        Method column name.
    noise_col : str
        Noise-level column name.
    minimize : bool
        If True, choose the smallest mean value; otherwise choose the largest.
    exclude_families : sequence[str] or None
        Optional families to exclude before selecting.

    Returns
    -------
    fixed_params_df : pandas.DataFrame
        One row per method with the chosen parameter values.
    """
    mean_col = f"{metric}_mean"
    if mean_col not in summary_df.columns:
        raise ValueError(f"Column '{mean_col}' not found in summary_df.")

    ref_df = summary_df[summary_df[noise_col] == reference_noise].copy()
    if ref_df.empty:
        raise ValueError(f"No rows found for reference noise level {reference_noise}.")

    if exclude_families is not None and "family" in ref_df.columns:
        ref_df = ref_df[~ref_df["family"].isin(exclude_families)]

    grouped = ref_df.groupby(method_col, dropna=False)
    idx = grouped[mean_col].idxmin() if minimize else grouped[mean_col].idxmax()

    fixed_params_df = ref_df.loc[idx].copy()
    fixed_params_df = fixed_params_df.sort_values(method_col).reset_index(drop=True)

    return fixed_params_df

def filter_summary_by_fixed_params(
    summary_df,
    fixed_params_df,
    method_col="method",
    noise_col="noise_level",
):
    """
    Filter summary_df so that, for each method, only the rows matching the
    chosen fixed parameters remain across all noise levels.
    """
    param_cols = [c for c in ["p", "moments", "BSorder"] if c in summary_df.columns]

    filtered_parts = []

    for _, row in fixed_params_df.iterrows():
        method = row[method_col]
        subdf = summary_df[summary_df[method_col] == method].copy()

        for col in param_cols:
            val = row[col]

            if pd.isna(val):
                subdf = subdf[subdf[col].isna()]
            else:
                subdf = subdf[subdf[col] == val]

        filtered_parts.append(subdf)

    if not filtered_parts:
        raise ValueError("No rows matched the fixed parameter selections.")

    filtered_df = pd.concat(filtered_parts, ignore_index=True)
    filtered_df = filtered_df.sort_values([method_col, noise_col]).reset_index(drop=True)

    return filtered_df

from IPython.display import display

def display_fixed_params(
    fixed_params_df,
    metric,
    exclude_families=("FBP",),
):
    """
    Display the fixed parameter selections chosen at the reference noise level.

    Parameters
    ----------
    fixed_params_df : pandas.DataFrame
        Output from select_fixed_params_from_reference_noise(...).
    metric : str
        Base metric name.
    exclude_families : sequence[str] or None
        Families to exclude from the display table.
    """
    df = fixed_params_df.copy()

    if "family" in df.columns and exclude_families is not None:
        df = df[~df["family"].isin(exclude_families)]

    cols = []
    for c in ["family", "method", "p", "moments", "BSorder"]:
        if c in df.columns:
            cols.append(c)

    cols += [f"{metric}_mean"]
    if f"{metric}_std" in df.columns:
        cols.append(f"{metric}_std")

    display(
        df[cols]
        .sort_values("method")
        .reset_index(drop=True)
    )