import numpy as np

from src.dg_utils import eval_orthonormal_legendre_1d
from src.dg_nodal_transform import vandermonde_1d
from src.research_utils_2d import local_cell_center_nodes_1d
from src.siac_modal import(
    siac_cgam, 
    grab_integrals
)

def build_mesh_1d(K, domain=(-1, 1)):
    xmin, xmax = domain
    h = (xmax - xmin) / K
    edges = np.linspace(xmin, xmax, K + 1)
    return {
        "domain": (xmin, xmax),
        "K": K,
        "h": h,
        "edges": edges,
    }


def build_grid_from_local_nodes_1d(mesh, eval_nodes, return_blocks=False):
    """
    Map local reference nodes in [-1,1] to each physical element.
    
    Parameters
    ----------
    mesh : dict
    eval_nodes : array_like, shape (n_eval,)
        Local reference nodes in [-1,1].
    
    Returns
    -------
    grid : ndarray, shape (K * n_eval,)
        Flattened physical grid.
    blocks : ndarray, shape (K, n_eval)
        Physical grid grouped per element.
    """
    eval_nodes = np.asarray(eval_nodes, dtype=float)
    K = mesh["K"]
    edges = mesh["edges"]
    n_eval = len(eval_nodes)

    blocks = np.zeros((K, n_eval), dtype=float)

    for e in range(K):
        x0, x1 = edges[e], edges[e + 1]
        xc = 0.5 * (x0 + x1)
        hx = x1 - x0
        blocks[e, :] = xc + 0.5 * hx * eval_nodes

    grid = blocks.reshape(K * n_eval)
    if return_blocks:
        return grid, blocks
    return grid


def nodal_to_modal_1d(Unode, mesh, p):
    """
    Convert 1D elementwise nodal data to modal coefficients.

    Parameters
    ----------
    Unode : ndarray, shape (K, p+1)
        Nodal values per element.
    mesh : dict
    p : int

    Returns
    -------
    dg : dict
    """
    K = mesh["K"]
    order = p + 1
    nodes = local_cell_center_nodes_1d(order)
    V, Vinv = vandermonde_1d(nodes=nodes, p=p)

    Unode = np.asarray(Unode, dtype=float)
    if Unode.shape != (K, order):
        raise ValueError(f"Expected Unode.shape = {(K, order)}, got {Unode.shape}")

    coeffs = np.zeros((K, order), dtype=float)
    for e in range(K):
        coeffs[e, :] = Vinv @ Unode[e, :]

    dg = {
        "coeffs": coeffs,
        "nodes": nodes,
        "V": V,
        "Vinv": Vinv,
        "p": p,
        "mesh": mesh,
        "construction": "nodal_transform_1d",
    }
    return dg


def modal_to_nodal_1d(dg, return_blocks=False):
    """
    Recover nodal values from modal coefficients at the original DG nodes.

    Returns
    -------
    Urec : ndarray, shape (K, p+1)
    """
    coeffs = dg["coeffs"]
    V = dg["V"]
    K, order = coeffs.shape

    Urec = np.zeros((K, order), dtype=float)
    for e in range(K):
        Urec[e, :] = V @ coeffs[e, :]
    img_dg = Urec.reshape(K * order)
    if return_blocks:
        return img_dg, Urec
    return Urec


def eval_dg_modal_local_nodes_1d(dg, eval_nodes=None, return_blocks=False):
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

    phi = eval_orthonormal_legendre_1d(eval_nodes, p)  # shape (p+1, n_eval)
    n_eval = len(eval_nodes)

    ueval = np.zeros((K, n_eval), dtype=float)
    for e in range(K):
        ueval[e, :] = coeffs[e, :] @ phi

    img_dg = ueval.reshape(K * n_eval)

    if return_blocks:
        return img_dg, ueval
    return img_dg


def pad_modal_coeffs_1d(coeffs, pad):
    """
    Zero-pad modal DG coefficients in element space
    
    Assumes coeffs has shape (K, order)
    """
    K, order = coeffs.shape
    
    out = np.zeros(
        (K + 2*pad, order), dtype=coeffs.dtype
        )
    out[pad:pad + K, :] = coeffs
    return out


def apply_siac_modal_dg_local_nodes_1d(dg, moments=None, BSorder=None, eval_nodes=None, quad_order=None, return_blocks=False):
    """
    Apply the SIAC filter to a modal DG solution, evaluated at arbitrary
    symmetric local reference nodes in each element.

    Parameters
    ----------
    dg : dict
        DG representation with coeffs[e, mode].
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
        Shape = (K*n_eval).
    """  
    mesh = dg["mesh"]
    coeffs = dg["coeffs"]
    
    p = dg["p"]
    order = p + 1
    K = mesh["K"]
    
    if moments is None:
        moments = 2 * p
    if BSorder is None:
        BSorder = p + 1
    
    if eval_nodes is None:
        nodes = local_cell_center_nodes_1d(order)
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
        BSsupport=BSsupport, 
        quad_order=quad_order
    )   # (order, BSlen, n_eval)
    
    kernellength = int(2 * np.ceil((moments + BSorder) / 2) + 1)
    halfker = int(np.ceil((moments + BSorder) / 2))

    SIACmatrix = np.zeros((order, kernellength, n_eval), dtype=float)
    
    for k in range(n_eval):
        for igam in range(moments + 1):
            SIACmatrix[:, igam:igam + BSlen, k] += cgam[igam] * BSInt[:, :, k]

    pad = halfker + 1
    coeffs_pad = pad_modal_coeffs_1d(coeffs, pad)
    
    ustar = np.zeros((K, n_eval), dtype=float)
    
    for e in range(K):
        c = e + pad
        
        block = coeffs_pad[c - halfker:c + halfker + 1, :]
        
        for k in range(n_eval):
            S = SIACmatrix[:, :, k]
            # block: (kernellength, order), S: (order, kernellength)
            ustar[e, k] = np.einsum("mr,rm->", S, block)
            # ustar[e, k] = np.sum(S * block.T)

    img_siac = ustar.reshape(K * n_eval)
    if return_blocks:
        return img_siac, ustar
    return img_siac

def trim_valid_siac_region_1d(arr, n_eval, moments, BSorder, safety_pad=False, return_trim=False):
    """
    Trim away the boundary region affected by SIAC zero-padding.
    """
    halfker = int(np.ceil((moments + BSorder) / 2))
    pad = halfker + 1 if safety_pad else halfker
    trim = pad * n_eval

    sl = slice(trim, -trim)
    if return_trim:
        return arr[sl], trim
    return arr[sl]