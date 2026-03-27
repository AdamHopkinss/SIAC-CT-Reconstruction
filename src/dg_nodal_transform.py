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
    Return the p+1 equispaced cell-center nodes on [-1,1]:

        r_j = -1 + (2j+1)/(p+1),  j=0,...,p
    """
    if not isinstance(p, int):
        raise TypeError("p must be an integer.")
    if p < 0:
        raise ValueError("p must be nonnegative.")

    j = np.arange(p + 1, dtype=float)
    return -1.0 + (2.0 * j + 1.0) / (p + 1.0)


def vandermonde_1d(nodes, p):
    """
    Build the 1D Vandermonde matrix for the orthonormal Legendre basis.

    V[i, m] = phi_m(nodes[i])

    Returns
    -------
    V : ndarray, shape (p+1, p+1)
    Vinv : ndarray, shape (p+1, p+1)
    """
    V = eval_orthonormal_legendre_1d(nodes, p).T
    Vinv = np.linalg.inv(V)
    return V, Vinv


def nodal_image_to_dg(recon, xlim=(-1, 1), ylim=(-1, 1), deg=3):
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

    DOF_y, DOF_x = arr.shape

    xgrid, ygrid, dx, dy = build_image_grid(DOF_x, DOF_y, xlim=xlim, ylim=ylim)
    mesh = build_dg_mesh(DOF_x, DOF_y, xlim=xlim, ylim=ylim, deg=deg)

    p = mesh["p"]
    Kx = mesh["Kx"]
    Ky = mesh["Ky"]
    order = mesh["order"]

    nodes = reference_nodes_cell_centers_1d(p)
    V, Vinv = vandermonde_1d(nodes, p)

    # Reshape global image into local element blocks:
    # arr[iy, ix] -> blocks[ey, ex, ly, lx]
    blocks = arr.reshape(Ky, order, Kx, order).transpose(0, 2, 1, 3)

    coeffs = np.zeros((Ky, Kx, order, order), dtype=float)

    for ey in range(Ky):
        for ex in range(Kx):
            U_block_yx = blocks[ey, ex]          # shape (ly, lx)
            A_block_yx = Vinv @ U_block_yx @ Vinv.T
            coeffs[ey, ex] = A_block_yx          # shape (my, mx)

    dg = {
        "mesh": mesh,
        "coeffs": coeffs,
        "basis": "orthonormal_legendre_tensor",
        "construction": "nodal_transform",
        "nodes_1d": nodes,
        "vandermonde_1d": V,
        "xgrid": xgrid,
        "ygrid": ygrid,
        "dx": dx,
        "dy": dy,
    }
    return dg

def dg_modal_to_nodal_image(dg):
    """
    Reconstruct the original nodal image values from the modal DG coefficients
    using the same local Vandermonde transform.

    This should roundtrip to machine precision for dg objects created by
    nodal_image_to_dg.

    Returns
    -------
    arr : ndarray, shape (DOF_y, DOF_x)
    """
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
            A_block_yx = coeffs[ey, ex]      # (my, mx)
            U_block_yx = V @ A_block_yx @ V.T
            blocks[ey, ex] = U_block_yx      # (ly, lx)

    arr = blocks.transpose(0, 2, 1, 3).reshape(Ky * order, Kx * order)
    return arr
