# Converts image / FD point-sample data into a tensor-product DG modal representation.
# The data are interpreted elementwise as nodal values on a DG grid, and local
# Vandermonde transforms are used to obtain modal coefficients.
#
# Adapted from ideas in Jennifer Ryan's SIAC MATLAB codes
# https://github.com/jennkryan/SIACMagicTestCodes/
# (SIACmatrix_driver.m), but modified here for 2D tomographic image data.


import numpy as np
from src.dg_utils import _to_numpy, build_dg_mesh, eval_orthonormal_legendre_1d, build_image_grid

def reference_nodes_cell_centers_1d(p):
    """
    Return the p+1 equispaced cell-center nodes on [-1,1].

    These are the midpoints of the p+1 equal subintervals of [-1,1]:
        r_j = -1 + (2j+1)/(p+1),   j=0,...,p
    """
    if not isinstance(p, int):
        raise TypeError("p must be an integer.")
    if p < 0:
        raise ValueError("p must be nonnegative.")

    j = np.arange(p + 1, dtype=float)
    return -1.0 + (2.0 * j + 1.0) / (p + 1.0)
    
def vandermonde_1d(nodes, p):
    """
    Build the Vandermonde matrix V for the orthonormal Legendre basis
    evaluated at the given nodes.

    V[i, m] = phi_m(nodes[i])

    Returns
    -------
    V : ndarray, shape (p+1, p+1)
    """
    L = eval_orthonormal_legendre_1d(nodes, p)  # (p+1, p+1)
    V = L.T
    Vinv = np.linalg.inv(V)
    return V, Vinv
    
def nodal_image_to_dg(recon, xlim=(-1,1), ylim=(-1,1), deg=3):
    """
    Interpret image samples as local nodal DG values and convert them
    elementwise to modal DG coefficients.

    Parameters
    ----------
    recon : array-like or ODL element
        Input image data of shape (DOF_y, DOF_x).
    xlim, ylim : tuple[float, float]
        Physical domain limits.
    deg : int
        Requested minimum polynomial degree.

    Returns
    -------
    dg : dict
        DG representation with modal coefficients and mesh metadata.
    """
    
    arr = _to_numpy(recon)
    
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array/image for recon.")

    # NumPy shape convention: (DOF_y, DOF_x)
    DOF_y, DOF_x = arr.shape
    
    xgrid, ygrid, dx, dy = build_image_grid(DOF_x, DOF_y, xlim=xlim, ylim=ylim)
    mesh = build_dg_mesh(DOF_x, DOF_y, xlim=xlim, ylim=ylim, deg=deg)
    
    p = mesh["p"]
    Kx = mesh["Kx"]
    Ky = mesh["Ky"]
    order = mesh["order"]
    
    # Reference nodes and Vandermonde
    nodes = reference_nodes_cell_centers_1d(p)
    V, Vinv = vandermonde_1d(nodes, p)
    
    coeffs = np.zeros((Ky, Kx, order, order))
    
    blocks = arr.reshape(Ky, order, Kx, order).transpose(0, 2, 1, 3) # -> shape: (Ky, Kx, order, order)
    for ey in range(Ky):
        # yslice = slice(ey * order, (ey + 1) * order)
        for ex in range(Kx):
            # xslice = slice(ex * order, (ex + 1) * order)
            # # Local nodal block for one element
            # U_block = arr[yslice, xslice]
            
            # Local nodal block for one element
            U_block_yx = blocks[ey, ex, :, :]   # (local_y, local_x)
            
            # Convert nodal -> modal
            A_block = Vinv @ U_block_yx @ Vinv.T
            
            # Assemble
            coeffs[ey, ex, :, :] = A_block
            
    dg = {
        "mesh": mesh, 
        "coeffs": coeffs,
        "basis": "orthonormal_legendre_tensor",
        "construction": "nodal_transform",
        "nodes_1d": nodes,
        "vandermonde_1d": V,

    }
    
    # Optional metadata
    dg["xgrid"] = xgrid
    dg["ygrid"] = ygrid
    dg["dx"] = dx
    dg["dy"] = dy

    return dg
    
def eval_dg_modal_on_img_grid_nodal(dg):
    mesh = dg["mesh"]
    coeffs = dg["coeffs"]

    p = mesh["p"]
    Kx = mesh["Kx"]
    Ky = mesh["Ky"]
    order = mesh["order"]

    nodes = dg.get("nodes_1d", reference_nodes_cell_centers_1d(p))
    V, _ = vandermonde_1d(nodes, p)

    blocks = np.zeros((Ky, Kx, order, order), dtype=float)

    for ey in range(Ky):
        for ex in range(Kx):
            A_block = coeffs[ey, ex, :, :]
            U_block_xy = V @ A_block @ V.T
            blocks[ey, ex, :, :] = U_block_xy # back to (local_y, local_x)

    arr = blocks.transpose(0, 2, 1, 3).reshape(Ky * order, Kx * order)
    return arr