from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import _to_numpy


def _get_zoom_box(arr, zoom_frac=(0.30, 0.70, 0.30, 0.70)):
    """
    Compute integer zoom box from fractional coordinates.

    Parameters
    ----------
    arr : np.ndarray
        2D image array.
    zoom_frac : tuple
        (x0f, x1f, y0f, y1f) as fractions of image size.

    Returns
    -------
    tuple
        (x0, x1, y0, y1)
    """
    ny, nx = arr.shape[:2]
    x0f, x1f, y0f, y1f = zoom_frac
    x0 = int(x0f * nx)
    x1 = int(x1f * nx)
    y0 = int(y0f * ny)
    y1 = int(y1f * ny)
    return x0, x1, y0, y1


def plot_img(img, title=None, figsize=(5, 5), vmin=0.0, vmax=1.0):
    """
    Plot a full image.

    Parameters
    ----------
    img : array-like or ODL element
        Image to display.
    title : str, optional
        Figure title.
    figsize : tuple, optional
        Matplotlib figure size.
    vmin, vmax : float, optional
        Intensity range for display.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """
    arr = _to_numpy(img)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)

    plt.show()
    return fig, ax


def plot_img_zoom(img, title=None, figsize=(5, 5), vmin=0.0, vmax=1.0,
                  zoom_frac=(0.30, 0.70, 0.30, 0.70)):
    """
    Plot a zoomed crop of an image.

    Parameters
    ----------
    img : array-like or ODL element
        Image to display.
    title : str, optional
        Figure title.
    figsize : tuple, optional
        Matplotlib figure size.
    vmin, vmax : float, optional
        Intensity range for display.
    zoom_frac : tuple, optional
        Fractional crop box (x0f, x1f, y0f, y1f).

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """
    arr = _to_numpy(img)
    x0, x1, y0, y1 = _get_zoom_box(arr, zoom_frac)
    arr_zoom = arr[y0:y1, x0:x1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(arr_zoom, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)

    plt.show()
    return fig, ax


def save_image_w_zoom(img, save_path, figsize=(5, 5), dpi=300,
                      vmin=0.0, vmax=1.0,
                      zoom_frac=(0.30, 0.70, 0.30, 0.70)):
    """
    Save a full image and a zoomed version.

    The zoomed version gets '_zoom' appended before the file extension.

    Parameters
    ----------
    img : array-like or ODL element
        Image to save.
    save_path : str or Path
        Output path, e.g. 'results/recon.png'.
    figsize : tuple, optional
        Figure size used for both saved figures.
    dpi : int, optional
        Save resolution.
    vmin, vmax : float, optional
        Intensity range for display.
    zoom_frac : tuple, optional
        Fractional crop box (x0f, x1f, y0f, y1f).

    Returns
    -------
    tuple
        (full_path, zoom_path)
    """
    arr = _to_numpy(img)

    save_path = Path(save_path)
    zoom_path = save_path.with_name(save_path.stem + "_zoom" + save_path.suffix)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    x0, x1, y0, y1 = _get_zoom_box(arr, zoom_frac)
    arr_zoom = arr[y0:y1, x0:x1]

    # Full image
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Zoom image
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(arr_zoom, cmap="gray", vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.savefig(zoom_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return str(save_path), str(zoom_path)


def plot_mc_metric(
    summary_df,
    metric,
    method_col="method",
    noise_col="noise_level",
    style_cols=None,
    fixed_filters=None,
    title=None,
    ylabel=None,
    show_std=True,
):
    """
    Plot a summarized MC metric versus noise level.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Output from summarize_mc_results(...).
    metric : str
        Base metric name, e.g. "rel_l2_err", "ssim", "gw_ssim".
    method_col : str
        Column containing method names.
    noise_col : str
        Column containing noise levels.
    style_cols : list[str] or None
        Extra parameter columns used to distinguish curves, e.g.
        ["moments", "BSorder"] or ["p", "moments", "BSorder"].
    fixed_filters : dict or None
        Optional filters applied before plotting, e.g.
        {"p": 1, "moments": 4}.
    title : str or None
        Plot title.
    ylabel : str or None
        Y-axis label. Defaults to metric name.
    show_std : bool
        Whether to show mean ± std as a shaded region.

    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if mean_col not in summary_df.columns:
        raise ValueError(f"Column '{mean_col}' not found in summary_df.")

    df = summary_df.copy()

    if fixed_filters is not None:
        for col, val in fixed_filters.items():
            df = df[df[col] == val]

    if style_cols is None:
        style_cols = []

    fig, ax = plt.subplots(figsize=(7, 4.5))

    group_cols = [method_col] + style_cols
    grouped = df.groupby(group_cols, dropna=False)

    for key, subdf in grouped:
        subdf = subdf.sort_values(noise_col)

        if not isinstance(key, tuple):
            key = (key,)

        label_parts = []
        for col, val in zip(group_cols, key):
            if pd.isna(val):
                continue
            label_parts.append(f"{col}={val}")
        label = ", ".join(label_parts)

        x = subdf[noise_col].to_numpy()
        y = subdf[mean_col].to_numpy()

        ax.plot(x, y, marker="o", label=label)

        if show_std and std_col in subdf.columns:
            ystd = subdf[std_col].fillna(0.0).to_numpy()
            ax.fill_between(x, y - ystd, y + ystd, alpha=0.2)

    ax.set_xlabel("Noise level")
    ax.set_ylabel(ylabel if ylabel is not None else metric)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    return fig, ax

from IPython.display import display
def display_best_params(best_df, metric):
    cols = ["method", "noise_level"]

    for c in ["p", "moments", "BSorder"]:
        if c in best_df.columns:
            cols.append(c)

    cols += [f"{metric}_mean"]
    if f"{metric}_std" in best_df.columns:
        cols.append(f"{metric}_std")

    display(best_df[cols].sort_values(["method", "noise_level"]).reset_index(drop=True))
    
def plot_selected_param_vs_noise(
    best_df,
    param,
    method_col="method",
    noise_col="noise_level",
    title=None,
    ylabel=None,
):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for method, subdf in best_df.groupby(method_col, dropna=False):
        subdf = subdf.sort_values(noise_col)

        x = subdf[noise_col].to_numpy()
        y = subdf[param].to_numpy()

        mask = ~pd.isna(y)
        ax.plot(x[mask], y[mask], marker="o", label=method)

    ax.set_xlabel("Noise level")
    ax.set_ylabel(ylabel if ylabel is not None else param)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    return fig, ax


def plot_param_heatmap(
    summary_df,
    method,
    metric,
    noise_level,
    x_col="BSorder",
    y_col="moments",
    fixed_filters=None,
    title=None,
    cmap="viridis",
    mark_best=True,
):
    val_col = f"{metric}_mean"
    if val_col not in summary_df.columns:
        raise ValueError(f"Column '{val_col}' not found in summary_df.")

    df = summary_df.copy()
    df = df[(df["method"] == method) & (df["noise_level"] == noise_level)]

    if fixed_filters is not None:
        for col, val in fixed_filters.items():
            df = df[df[col] == val]

    if df.empty:
        raise ValueError("No rows left after filtering. Check method/noise_level/fixed_filters.")

    pivot = df.pivot(index=y_col, columns=x_col, values=val_col)
    pivot = pivot.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if title is not None:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{metric} mean")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    if mark_best:
        vals = pivot.values.astype(float)
        idx = np.nanargmin(vals)
        i_best, j_best = np.unravel_index(idx, vals.shape)
        rect = plt.Rectangle((j_best - 0.5, i_best - 0.5), 1, 1,
                             fill=False, edgecolor="red", linewidth=2)
        ax.add_patch(rect)

    plt.tight_layout()
    return fig, ax

def compare_fixed_vs_retuned(
    retuned_df,
    fixed_df,
    metric,
    method_col="method",
    noise_col="noise_level",
    title=None,
    ylabel=None,
):
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for method in sorted(retuned_df[method_col].dropna().unique()):
        sub_retuned = retuned_df[retuned_df[method_col] == method].sort_values(noise_col)
        sub_fixed = fixed_df[fixed_df[method_col] == method].sort_values(noise_col)

        x_r = sub_retuned[noise_col].to_numpy()
        y_r = sub_retuned[mean_col].to_numpy()

        x_f = sub_fixed[noise_col].to_numpy()
        y_f = sub_fixed[mean_col].to_numpy()

        ax.plot(x_r, y_r, marker="o", label=f"{method} (retuned)")
        ax.plot(x_f, y_f, marker="s", linestyle="--", label=f"{method} (fixed @ 0.10)")

        if std_col in sub_retuned.columns:
            ystd_r = sub_retuned[std_col].fillna(0.0).to_numpy()
            ax.fill_between(x_r, y_r - ystd_r, y_r + ystd_r, alpha=0.15)

        if std_col in sub_fixed.columns:
            ystd_f = sub_fixed[std_col].fillna(0.0).to_numpy()
            ax.fill_between(x_f, y_f - ystd_f, y_f + ystd_f, alpha=0.10)

    ax.set_xlabel("Noise level")
    ax.set_ylabel(ylabel if ylabel is not None else metric)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    return fig, ax