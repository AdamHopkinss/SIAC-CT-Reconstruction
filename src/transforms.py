import numpy as np

from src.basis import eval_orthonormal_legendre_1d
from src.grid import local_cell_center_nodes_1d
from src.utils import _to_numpy

def vandermonde_1d(nodes, p):
    """
    Build the 1D Vandermonde matrix for the orthonormal Legendre basis.

    V[m, i] = phi_m(nodes[i])

    Returns
    -------
    V : ndarray, shape (p+1, p+1)
    Vinv : ndarray, shape (p+1, p+1)
    """
    V = eval_orthonormal_legendre_1d(nodes, p)
    Vinv = np.linalg.inv(V)
    return V, Vinv


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
    coeffs = Unode @ Vinv

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
    Urec = coeffs @ V
    
    img_dg = Urec.reshape(K * order)
    if return_blocks:
        return img_dg, Urec
    return Urec

#### 2D ####

def nodal_to_modal_2d(Unode, mesh, p):
    """
    Convert 2D elementwise nodal data to modal coefficients.

    Parameters
    ----------
    Unode : ndarray, shape (Ky, Kx, p+1, p+1)
        Nodal values per element.
    mesh : dict
    p : int

    Returns
    -------
    dg : dict
    """
    Kx, Ky = mesh["Kx"], mesh["Ky"]
    order = p + 1
    
    nodes = local_cell_center_nodes_1d(order)
    V, Vinv = vandermonde_1d(nodes, p)
    
    Unode = np.asarray(Unode, dtype=float)
    if Unode.shape != (Ky, Kx, order, order):
        raise ValueError(f"Expected Unode.shape = {(Ky, Kx, order, order)}, got {Unode.shape}")
    
    # coeffs = np.zeros((Ky, Kx, order, order), dtype=float)
    # for ey in range(Ky):
    #     for ex in range(Kx):
    #         U_block_yx = Unode[ey, ex, :, :]                # shape (ly, lx)
    #         A_block_yx = Vinv @ U_block_yx @ Vinv.T
    #         coeffs[ey, ex, :, :] = A_block_yx               # shape (my, mx)
    
    # fast with matmul
    coeffs = np.matmul(np.matmul(Vinv.T, Unode), Vinv)
    
    dg = {
        "mesh": mesh,
        "coeffs": coeffs,
        "basis": "orthonormal_legendre_tensor",
        "construction": "nodal_transform",
        "nodes_1d": nodes, 
        "V": V, 
        "Vinv": Vinv
    }
    
    return dg

def modal_to_nodal_2d(dg, return_blocks=False):
    """
    Recover nodal values from modal coefficients at the original DG nodes.

    Returns
    -------
    Urec : ndarray, shape (Ky, Kx, p+1, p+1)
    """
    coeffs = dg["coeffs"]
    V = dg["V"]
    Ky, Kx, order, _ = coeffs.shape
    
    # Urec = np.zeros((Ky, Kx, order, order), dtype=float)
    # for ey in range(Ky):
    #     for ex in range(Kx):
    #         A_block_yx = coeffs[ey, ex, :, :]         # shape (my, mx)
    #         U_block_yx = V @ A_block_yx @ V.T
    #         Urec[ey, ex, :, :] = U_block_yx           # shape(ly, lx)
    
    # fast with matmul
    Urec = np.matmul(np.matmul(V.T, coeffs), V)
    
    img_dg = Urec.transpose(0, 2, 1, 3).reshape(Ky * order, Kx * order)
    if return_blocks:
        return img_dg, Urec
    return img_dg

# -----------------------------------------------------------------------------
# nodal_image_to_dg
#
# Converts point-sampled (image / finite-difference) data into a tensor-product
# DG modal representation. The data are interpreted elementwise as nodal values
# on a DG grid, and local Vandermonde transforms are applied to recover modal
# coefficients in each element.
#
# This implementation is inspired by the SIAC MATLAB test codes by
# Jennifer K. Ryan:
# https://github.com/jennkryan/SIACMagicTestCodes/
# (e.g. SIACmatrix_driver.m), but is reimplemented here for:
#   - 2D tensor-product DG representations
#   - image/tomographic data
#   - Python-based workflows
# -----------------------------------------------------------------------------
from src.grid import build_image_grid
from src.mesh import build_img_dg_mesh

def nodal_image_to_dg(recon, xlim=(-1, 1), ylim=(-1, 1), p=3, verbose=True):
    """
    Interpret image samples as elementwise nodal values on a tensor-product DG grid
    and convert them to modal DG coefficients.

    Storage convention
    ------------------
    coeffs[ey, ex, my, mx]
        ey, ex : element indices in y and x
        my, mx : modal indices in y and x

    Parameters
    ----------
    recon : array_like
        Image/sample array of shape (DOF_y, DOF_x).
    xlim, ylim : tuple[float, float]
        Physical domain limits.
    deg : int
        Requested minimum polynomial degree.

    Returns
    -------
    dg : dict
        DG representation with mesh, coefficients, and grid metadata.
    """
    
    arr = _to_numpy(recon)
    
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array/image for recon.")
    
    DOF_y, DOF_x = arr.shape    # numpy convention in contrast to ODL with (x, y)
    
    xgrid, ygrid, dx, dy = build_image_grid(DOF_x, DOF_y, xlim=xlim, ylim=ylim)
    mesh = build_img_dg_mesh(DOF_x, DOF_y, xlim=xlim, ylim=ylim, deg=p, verbose=False)

    # Unpack if p changed in resolve degree under mesh creation
    p = mesh["p"]
    Kx = mesh["Kx"]
    Ky = mesh["Ky"]
    order = mesh["order"]
    
    blocks = arr.reshape(Ky, order, Kx, order).transpose(0, 2, 1, 3)
    
    dg_temp = nodal_to_modal_2d(Unode=blocks, mesh=mesh, p=p)
    
    coeffs = dg_temp["coeffs"]
    nodes = dg_temp["nodes_1d"]
    V = dg_temp["V"]
    
    dg = {
        "mesh": mesh,
        "coeffs": dg_temp["coeffs"],
        "basis": "orthonormal_legendre_tensor",
        "construction": "nodal_transform",
        "nodes_1d": dg_temp["nodes_1d"],
        "vandermonde_1d": dg_temp["V"],
        "V": dg_temp["V"],
        "Vinv": dg_temp["Vinv"],
        "xgrid": xgrid,
        "ygrid": ygrid,
        "dx": dx,
        "dy": dy,
        "p": p
    }
    return dg