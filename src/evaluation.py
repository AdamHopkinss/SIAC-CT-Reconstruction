import numpy as np
from src.basis import eval_orthonormal_legendre_1d
from src.grid import local_cell_center_nodes_1d


def eval_dg_on_local_nodes_1d(dg, eval_nodes=None, return_blocks=False):
    """
    Evaluate the DG modal polynomial in each element at given local reference nodes.

    Parameters
    ----------
    dg : dict
    eval_nodes : array_like or None
        Local reference nodes in [-1,1].
        If None, use original DG nodes.
    return_blocks : bool
        If True, also return array of shape (K, n_eval).

    Returns
    -------
    img_dg : ndarray, shape (K * n_eval,)
    blocks : ndarray, optional, shape (K, n_eval)
    """
    coeffs = dg["coeffs"]
    p = dg["p"]
    mesh = dg["mesh"]
    K = mesh["K"]

    if eval_nodes is None:
        eval_nodes = local_cell_center_nodes_1d(p + 1)
    else:
        eval_nodes = np.asarray(eval_nodes, dtype=float)

    phi = eval_orthonormal_legendre_1d(eval_nodes, p)  # shape (p+1m n_eval)
    n_eval = len(eval_nodes)

    ueval = np.zeros((K, n_eval), dtype=float)
    for e in range(K):
        ueval[e, :] = coeffs[e, :] @ phi

    img_dg = ueval.reshape(K * n_eval)

    if return_blocks:
        return img_dg, ueval
    return img_dg

def eval_dg_on_local_nodes_2d(dg, eval_nodes=None):
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
        nodes = local_cell_center_nodes_1d(order)
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

def eval_dg_on_img_grid(dg):
    """
    Wrapper to evaluate on original image grid (no eval_nodes argument)
    Mainly to make the pipeline clearer.
    """
    img_dg = eval_dg_on_local_nodes_2d(dg=dg, eval_nodes=None)
    return img_dg
        