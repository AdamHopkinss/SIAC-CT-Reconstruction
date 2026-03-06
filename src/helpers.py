import numpy as np
import matplotlib.pyplot as plt

def _to_numpy(img):
    """Accept ODL element or numpy array."""
    if hasattr(img, "asarray"):
        return img.asarray()
    return np.asarray(img)

def _crop(img, x0, x1, y0, y1):
    return img[y0:y1, x0:x1]

def plot_recon_grid_3x2_with_zoom(
    reconstructions,      # list of 6 images
    titles,               # list of 6 strings
    suptitle_full="Reconstructions",
    suptitle_zoom="Reconstructions (zoom)",
    cmap="gray",
    common_scale=True,
    zoom_frac=(0.30, 0.70, 0.30, 0.70),
    zoom_box=None,
    figsize=(10, 12),
):
    """
    Plot a 3x2 grid of reconstructions and a zoomed 3x2 grid.

    zoom_frac: fractions of width/height.
    zoom_box: explicit pixel box overrides zoom_frac, format (x0,x1,y0,y1).
    """

    imgs = [_to_numpy(im) for im in reconstructions]
    assert len(imgs) == 6 and len(titles) == 6, "Need exactly 6 images and 6 titles."

    # ----- FULL SCALE -----
    if common_scale:
        vmin = 0 #min(im.min() for im in imgs)
        vmax = 1 #max(im.max() for im in imgs)
    else:
        vmin = vmax = None

    fig_full, axes = plt.subplots(3, 2, figsize=figsize, constrained_layout=True)
    axes = axes.ravel()

    ims = []
    for ax, im, title in zip(axes, imgs, titles):
        imh = ax.imshow(im, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ims.append(imh)

    cbar = fig_full.colorbar(ims[0], ax=axes, shrink=0.9, pad=0.02)
    fig_full.suptitle(suptitle_full, fontsize=14)

    plt.show()
    plt.close(fig_full)

    # ----- ZOOM BOX -----
    Ny, Nx = imgs[0].shape[:2]

    if zoom_box is None:
        x0f, x1f, y0f, y1f = zoom_frac
        x0 = int(x0f * Nx)
        x1 = int(x1f * Nx)
        y0 = int(y0f * Ny)
        y1 = int(y1f * Ny)
    else:
        x0, x1, y0, y1 = zoom_box

    imgs_zoom = [_crop(im, x0, x1, y0, y1) for im in imgs]

    if common_scale:
        vmin_z = 0 #min(im.min() for im in imgs_zoom)
        vmax_z = 1 #max(im.max() for im in imgs_zoom)
    else:
        vmin_z = vmax_z = None

    fig_zoom, axes = plt.subplots(3, 2, figsize=figsize, constrained_layout=True)
    axes = axes.ravel()

    ims = []
    for ax, im, title in zip(axes, imgs_zoom, titles):
        imh = ax.imshow(im, vmin=vmin_z, vmax=vmax_z, cmap=cmap)
        ax.set_title(title + f"\n(zoom x[{x0}:{x1}] y[{y0}:{y1}])")
        ax.set_xticks([])
        ax.set_yticks([])
        ims.append(imh)

    cbar = fig_zoom.colorbar(ims[0], ax=axes, shrink=0.9, pad=0.02)
    fig_zoom.suptitle(suptitle_zoom, fontsize=14)

    plt.show()
    plt.close(fig_zoom)

    return (fig_full, fig_zoom)


import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _to_numpy(img):
    """Accept ODL element or numpy array."""
    if hasattr(img, "asarray"):
        return img.asarray()
    return np.asarray(img)


def _crop(img, x0, x1, y0, y1):
    return img[y0:y1, x0:x1]


def save_and_plot_image_with_zoom(
    img,
    save_path,
    zoom_save_path=None,
    title=None,
    zoom_title=None,
    cmap="gray",
    vmin=0.0,
    vmax=1.0,
    zoom_frac=(0.30, 0.70, 0.30, 0.70),
    zoom_box=None,
    show=True,
    save=True,
    raw=True,
    add_colorbar=False,
    figsize=(5, 5),
    zoom_figsize=(5, 5),
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.0,
):
    """
    Plot and save a single reconstruction image and a zoomed version.

    Parameters
    ----------
    img : ODL element or np.ndarray
        Image to plot/save.
    save_path : str or Path
        Output path for the full image.
    zoom_save_path : str or Path, optional
        Output path for the zoomed image. If None, derived automatically
        from save_path by appending '_zoom' before the extension.
    title : str, optional
        Title for the full image. Ignored if raw=True.
    zoom_title : str, optional
        Title for the zoomed image. Ignored if raw=True.
    cmap : str
        Colormap.
    vmin, vmax : float or None
        Display range for imshow. If None, matplotlib auto-scales.
    zoom_frac : tuple
        Fractional crop box (x0f, x1f, y0f, y1f) used if zoom_box is None.
    zoom_box : tuple, optional
        Explicit crop box (x0, x1, y0, y1).
    show : bool
        If True, show the figures.
    save : bool
        If True, save the figures.
    raw : bool
        If True, save report-ready images with no title, axes, ticks, or colorbar.
    add_colorbar : bool
        If True, add a colorbar. Mainly useful when raw=False.
    figsize : tuple
        Figure size for the full image.
    zoom_figsize : tuple
        Figure size for the zoomed image.
    dpi : int
        Save resolution.
    bbox_inches : str
        Passed to savefig.
    pad_inches : float
        Passed to savefig.

    Returns
    -------
    info : dict
        Dictionary with figure handles, zoom box, and save paths.
    """
    arr = _to_numpy(img)

    save_path = Path(save_path)
    if zoom_save_path is None:
        zoom_save_path = save_path.with_name(save_path.stem + "_zoom" + save_path.suffix)
    else:
        zoom_save_path = Path(zoom_save_path)

    # Make parent folders if needed
    if save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        zoom_save_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine zoom box
    Ny, Nx = arr.shape[:2]
    if zoom_box is None:
        x0f, x1f, y0f, y1f = zoom_frac
        x0 = int(x0f * Nx)
        x1 = int(x1f * Nx)
        y0 = int(y0f * Ny)
        y1 = int(y1f * Ny)
    else:
        x0, x1, y0, y1 = zoom_box

    arr_zoom = _crop(arr, x0, x1, y0, y1)

    # -------- Full image --------
    fig_full, ax_full = plt.subplots(figsize=figsize)
    im_full = ax_full.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)

    if raw:
        ax_full.axis("off")
    else:
        if title is not None:
            ax_full.set_title(title)
        ax_full.set_xticks([])
        ax_full.set_yticks([])
        if add_colorbar:
            fig_full.colorbar(im_full, ax=ax_full, shrink=0.9, pad=0.02)

    if save:
        fig_full.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)

    if show:
        plt.show()
    else:
        plt.close(fig_full)

    # -------- Zoomed image --------
    fig_zoom, ax_zoom = plt.subplots(figsize=zoom_figsize)
    im_zoom = ax_zoom.imshow(arr_zoom, cmap=cmap, vmin=vmin, vmax=vmax)

    if raw:
        ax_zoom.axis("off")
    else:
        if zoom_title is not None:
            ax_zoom.set_title(zoom_title)
        elif title is not None:
            ax_zoom.set_title(f"{title} (zoom)")
        ax_zoom.set_xticks([])
        ax_zoom.set_yticks([])
        if add_colorbar:
            fig_zoom.colorbar(im_zoom, ax=ax_zoom, shrink=0.9, pad=0.02)

    if save:
        fig_zoom.savefig(zoom_save_path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)

    if show:
        plt.show()
    else:
        plt.close(fig_zoom)

    return {
        "fig_full": fig_full,
        "fig_zoom": fig_zoom,
        "save_path": str(save_path),
        "zoom_save_path": str(zoom_save_path),
        "zoom_box": (x0, x1, y0, y1),
    }