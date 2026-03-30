import numpy as np

from src.dg_utils import eval_orthonormal_legendre_1d
from src.dg_nodal_transform import reference_nodes_cell_centers_1d
from src.siac_modal import (
    local_pixel_center_nodes,
    siac_cgam,
    centered_cardinal_bspline,
    grab_integrals,
    pad_modal_coeffs_2d,
)


def local_cell_center_nodes_1d(nloc):
    """
    Return nloc equispaced cell-center nodes on [-1,1]:
        r_j = -1 + (2j+1)/nloc,  j=0,...,nloc-1
    """
    if not isinstance(nloc, int):
        raise TypeError("nloc must be an integer.")
    if nloc <= 0:
        raise ValueError("nloc must be positive.")

    j = np.arange(nloc, dtype=float)
    return -1.0 + (2.0 * j + 1.0) / nloc


def build_grid_from_local_nodes_2d(mesh, eval_nodes):
    """
    Build the global physical grid corresponding to local reference nodes
    repeated identically in each element.

    Parameters
    ----------
    mesh : dict
        DG mesh dictionary.
    eval_nodes : array_like
        Local reference nodes in [-1,1].

    Returns
    -------
    X, Y : ndarray
        Global physical grid of shape (Ky*n_eval, Kx*n_eval).
    """
    eval_nodes = np.asarray(eval_nodes, dtype=float)
    n_eval = len(eval_nodes)

    Kx = mesh["Kx"]
    Ky = mesh["Ky"]
    x_edges = mesh["x_edges"]
    y_edges = mesh["y_edges"]

    x_blocks = np.zeros((Kx, n_eval), dtype=float)
    y_blocks = np.zeros((Ky, n_eval), dtype=float)

    for ex in range(Kx):
        x0, x1 = x_edges[ex], x_edges[ex + 1]
        x_blocks[ex, :] = 0.5 * (x0 + x1) + 0.5 * (x1 - x0) * eval_nodes

    for ey in range(Ky):
        y0, y1 = y_edges[ey], y_edges[ey + 1]
        y_blocks[ey, :] = 0.5 * (y0 + y1) + 0.5 * (y1 - y0) * eval_nodes

    xgrid = x_blocks.reshape(Kx * n_eval)
    ygrid = y_blocks.reshape(Ky * n_eval)

    X, Y = np.meshgrid(xgrid, ygrid, indexing="xy")
    return X, Y


def eval_dg_modal_local_nodes(dg, eval_nodes=None):
    """
    Evaluate a modal tensor-product DG field at local reference nodes
    in every element.

    Parameters
    ----------
    dg : dict
        DG dictionary with coeffs[ey, ex, my, mx].
    eval_nodes : array_like or None
        Local reference nodes in [-1,1]. If None, use the original
        pixel-center nodes of length p+1.

    Returns
    -------
    img_dg : ndarray
        DG field on the global grid induced by the local nodes.
        Shape = (Ky*n_eval, Kx*n_eval).
    """
    mesh = dg["mesh"]
    coeffs = dg["coeffs"]

    p = mesh["p"]
    order = mesh["order"]
    Kx = mesh["Kx"]
    Ky = mesh["Ky"]

    if eval_nodes is None:
        nodes = local_pixel_center_nodes(order)
    else:
        nodes = np.asarray(eval_nodes, dtype=float)

    n_eval = len(nodes)
    phi = eval_orthonormal_legendre_1d(nodes, p)   # (order, n_eval)

    ueval = np.zeros((Ky, Kx, n_eval, n_eval), dtype=float)

    for ey in range(Ky):
        for ex in range(Kx):
            A_block_yx = coeffs[ey, ex]            # (my, mx)
            U_block_yx = phi.T @ A_block_yx @ phi  # (n_eval, n_eval)
            ueval[ey, ex] = U_block_yx

    img_dg = ueval.transpose(0, 2, 1, 3).reshape(Ky * n_eval, Kx * n_eval)
    return img_dg


def apply_siac_modal_dg_local_nodes(dg, moments=None, BSorder=None, eval_nodes=None):
    """
    Apply the SIAC filter to a modal DG solution, evaluated at arbitrary
    symmetric local reference nodes in each element.

    Parameters
    ----------
    dg : dict
        DG representation with coeffs[ey, ex, my, mx].
    moments : int or None
        Number of reproduced moments. Defaults to 2*p.
    BSorder : int or None
        B-spline order. Defaults to p+1.
    eval_nodes : array_like or None
        Local reference evaluation nodes in [-1,1]. If None, use the original
        pixel-center nodes of length p+1.

    Returns
    -------
    img_siac : ndarray
        SIAC field on the global grid induced by the local nodes.
        Shape = (Ky*n_eval, Kx*n_eval).
    """
    mesh = dg["mesh"]
    coeffs = dg["coeffs"]

    p = mesh["p"]
    order = mesh["order"]
    Kx = mesh["Kx"]
    Ky = mesh["Ky"]

    if moments is None:
        moments = 2 * p
    if BSorder is None:
        BSorder = p + 1

    if eval_nodes is None:
        nodes = local_pixel_center_nodes(order)
    else:
        nodes = np.asarray(eval_nodes, dtype=float)

    n_eval = len(nodes)

    BSknots = np.linspace(-BSorder / 2, BSorder / 2, BSorder + 1)
    BSsupport = np.array(
        [np.floor(BSknots[0]), np.ceil(BSknots[-1])],
        dtype=int
    )
    BSlen = int(BSsupport[1] - BSsupport[0] + 1)

    cgam = siac_cgam(moments, BSorder)

    BSInt = grab_integrals(
        eval_nodes=nodes,
        p=p,
        BSorder=BSorder,
        BSsupport=BSsupport
    )   # (order, BSlen, n_eval)

    kernellength = int(2 * np.ceil((moments + BSorder) / 2) + 1)
    halfker = int(np.ceil((moments + BSorder) / 2))

    SIACmatrix = np.zeros((order, kernellength, n_eval), dtype=float)

    for k in range(n_eval):
        for igam in range(moments + 1):
            SIACmatrix[:, igam:igam + BSlen, k] += cgam[igam] * BSInt[:, :, k]

    pad = halfker + 1
    coeffs_pad = pad_modal_coeffs_2d(coeffs, pad_x=pad, pad_y=pad)

    ustar = np.zeros((Ky, Kx, n_eval, n_eval), dtype=float)

    for ey in range(Ky):
        for ex in range(Kx):
            cy = ey + pad
            cx = ex + pad

            block = coeffs_pad[
                cy - halfker : cy + halfker + 1,
                cx - halfker : cx + halfker + 1,
                :,
                :
            ]   # (kernellength, kernellength, order, order)

            for ky in range(n_eval):
                Sy = SIACmatrix[:, :, ky]
                for kx in range(n_eval):
                    Sx = SIACmatrix[:, :, kx]
                    ustar[ey, ex, ky, kx] = np.einsum("mr,ns,rsmn->", Sy, Sx, block)

    img_siac = ustar.transpose(0, 2, 1, 3).reshape(Ky * n_eval, Kx * n_eval)
    return img_siac


def trim_valid_siac_region(arr, n_eval, moments, BSorder):
    """
    Trim away the boundary region affected by SIAC zero-padding.

    Parameters
    ----------
    arr : ndarray
        Global array of shape (Ky*n_eval, Kx*n_eval) or similar.
    n_eval : int
        Number of evaluation points per element in each direction.
    moments : int
        Number of reproduced moments.
    BSorder : int
        B-spline order.
    safety_pad : bool
        If True, trim using pad = halfker + 1. Otherwise trim using halfker.

    Returns
    -------
    arr_trim : ndarray
        Interior region.
    trim : int
        Number of grid points removed from each side.
    """
    halfker = int(np.ceil((moments + BSorder) / 2))
    pad = halfker
    trim = pad * n_eval

    sl_y = slice(trim, -trim)
    sl_x = slice(trim, -trim)
    return arr[sl_y, sl_x], trim