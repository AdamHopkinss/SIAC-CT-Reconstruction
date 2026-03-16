# Takes image data interpreted as point samples at pixel centers.
# Builds a bilinear interpolant of the sampled field and computes
# the elementwise L2 projection of that interpolant onto a DG(Q^p) space.

# The image data are interpreted as point samples at pixel centers of an 
# underlying field. A bilinear interpolant is constructed on the 
# pixel-center grid, and values outside this sampled support are taken 
# to be zero. This is consistent with the assumption that the reconstructed 
# object is compactly supported in the computational domain.

import numpy as np

from numpy.polynomial.legendre import leggauss
from scipy.interpolate import RegularGridInterpolator

from src.dg_utils import (_to_numpy, 
                          build_dg_mesh, 
                          eval_orthonormal_legendre_1d, 
                          build_image_grid,
                          eval_dg_modal
                          )


# Helper function: Create a function f(x,y) from the pixel grid, defined by the bilinear interpolator (callable)
def make_bilinear_function(arr, xgrid, ygrid, fill_value=0.0):
    """
    Returns a callable f(x,y) performing bilinear interpolation
    of img on the tensor grid (xgrid,ygrid).
    """ 
    interp = RegularGridInterpolator(
        (ygrid, xgrid), arr, method="linear",
        bounds_error=False, fill_value=fill_value
    )
    
    def f(x,y):
        x = np.asarray(x)
        y = np.asarray(y)
        pts = np.column_stack([y.ravel(), x.ravel()])
        vals = interp(pts)
        shape = np.broadcast(x, y).shape
        return vals.reshape(shape)

    return f  


def l2_project_pixels_to_dg(arr, xgrid, ygrid, mesh, quad_order=None):
    Kx, Ky, p = mesh["Kx"], mesh["Ky"], mesh["p"]
    x_edges, y_edges = mesh["x_edges"], mesh["y_edges"]
    
    if quad_order is None:
        quad_order = max(p + 2, 6)
    # Create a function f(x,y) from the pixel grid, defined by the bilinear interpolator (callable)
    f = make_bilinear_function(arr, xgrid=xgrid, ygrid=ygrid)
    
    rq, wq = leggauss(quad_order)
    sq, vq = leggauss(quad_order)
    
    Lr = eval_orthonormal_legendre_1d(rq, p)  # (p+1, nq)
    Ls = eval_orthonormal_legendre_1d(sq, p)  # (p+1, nq)
    
    coeffs = np.zeros((Ky, Kx, p + 1, p + 1), dtype=float)
    
    R, S = np.meshgrid(rq, sq, indexing="xy")   # (nq,nq)
    W = np.outer(vq, wq)                        # (nq,nq)
    
    for ey in range(Ky):
        y0, y1 = y_edges[ey], y_edges[ey + 1]
        Y = y0 + 0.5 * (y1 - y0) * (S + 1.0)
        
        for ex in range(Kx):
            x0, x1 = x_edges[ex], x_edges[ex + 1]
            X = x0 + 0.5 * (x1 - x0) * (R + 1.0)
            
            F = f(X, Y)     # (nq, nq)
            G = W * F       # weights included
            
            # Integrate in r:  A[b,i] = sum_a G[b,a] * Lr[i,a]
            # (nq,nq) @ (nq,p+1) -> (nq,p+1)
            A = G @ Lr.T
            
            # Integrate in s:  c[j,i] = sum_b Ls[j,b] * A[b,i]
            # (p+1,nq) @ (nq,p+1) -> (p+1,p+1)
            c_ji = Ls @ A
            
            coeffs[ey, ex, :, :] = c_ji.T  # store as (i,j)
            
    return {"mesh": mesh, 
            "coeffs": coeffs, 
            "basis": "orthonormal_legendre_tensor", 
            "construction": "l2_projection",
            "quad_order": quad_order
            }


def l2_project_image_to_dg(recon, xlim=(-1, 1), ylim=(-1, 1), deg=3):
    """
    Project reconstruction data onto a DG(Q^p) space on a uniform mesh.
    """
    arr = _to_numpy(recon)

    if arr.ndim != 2:
        raise ValueError("Expected a 2D array/image for recon.")

    # NumPy shape convention: (DOF_y, DOF_x)
    DOF_y, DOF_x = arr.shape

    xgrid, ygrid, dx, dy = build_image_grid(DOF_x, DOF_y, xlim=xlim, ylim=ylim)
    mesh = build_dg_mesh(DOF_x, DOF_y, xlim=xlim, ylim=ylim, deg=deg)

    dg = l2_project_pixels_to_dg(arr, xgrid, ygrid, mesh)

    # Optional metadata
    dg["xgrid"] = xgrid
    dg["ygrid"] = ygrid
    dg["dx"] = dx
    dg["dy"] = dy

    return dg
