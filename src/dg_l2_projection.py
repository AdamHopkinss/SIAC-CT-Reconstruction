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
                          build_image_grid
                          )


def build_dg_mesh_l2(Kx, Ky, xlim=(-1, 1), ylim=(-1, 1), deg=3, DOF_x=None, DOF_y=None):
    """
    Build a uniform tensor-product DG mesh for L2 projection.

    Unlike the nodal-to-modal construction, this mesh is chosen
    independently of the image resolution.
    """
    xmin, xmax = xlim
    ymin, ymax = ylim

    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Require xlim[1] > xlim[0] and ylim[1] > ylim[0].")

    if Kx <= 0 or Ky <= 0:
        raise ValueError("Require Kx > 0 and Ky > 0.")

    p = int(deg)
    if p < 0:
        raise ValueError("Require deg >= 0.")

    hx = (xmax - xmin) / Kx
    hy = (ymax - ymin) / Ky

    x_edges = np.linspace(xmin, xmax, Kx + 1)
    y_edges = np.linspace(ymin, ymax, Ky + 1)

    return {
        "domain": [xmin, xmax, ymin, ymax],
        "Kx": Kx,
        "Ky": Ky,
        "p": p,
        "order": p + 1,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "hx": hx,
        "hy": hy,
        "DOF_x": DOF_x,
        "DOF_y": DOF_y,
    }


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

import numpy as np


def make_piecewise_constant_function(arr, xgrid, ygrid, fill_value=0.0):
    """
    Returns a callable f(x, y) that interprets arr as piecewise constant
    over pixel cells centered at (xgrid, ygrid).

    Outside the pixel-cell domain, returns fill_value.

    Parameters
    ----------
    arr : ndarray, shape (len(ygrid), len(xgrid))
        Pixel values.
    xgrid, ygrid : ndarray
        Pixel-center coordinates, assumed uniform.
    fill_value : float
        Value returned outside the covered pixel support.
    """
    arr = np.asarray(arr)
    xgrid = np.asarray(xgrid)
    ygrid = np.asarray(ygrid)

    if arr.shape != (len(ygrid), len(xgrid)):
        raise ValueError("arr shape must be (len(ygrid), len(xgrid))")

    if len(xgrid) < 2 or len(ygrid) < 2:
        raise ValueError("Need at least 2 grid points in each direction.")

    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]

    # pixel-cell boundaries
    xmin = xgrid[0] - 0.5 * dx
    xmax = xgrid[-1] + 0.5 * dx
    ymin = ygrid[0] - 0.5 * dy
    ymax = ygrid[-1] + 0.5 * dy

    def f(x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        shape = np.broadcast(x, y).shape
        xb = np.broadcast_to(x, shape)
        yb = np.broadcast_to(y, shape)

        out = np.full(shape, fill_value, dtype=float)

        mask = (xb >= xmin) & (xb < xmax) & (yb >= ymin) & (yb < ymax)

        if np.any(mask):
            ix = np.floor((xb[mask] - xmin) / dx).astype(int)
            iy = np.floor((yb[mask] - ymin) / dy).astype(int)

            # safeguard against roundoff at the upper boundary
            ix = np.clip(ix, 0, len(xgrid) - 1)
            iy = np.clip(iy, 0, len(ygrid) - 1)

            out[mask] = arr[iy, ix]

        return out

    return f

def make_sampled_function(arr, xgrid, ygrid, mode="bilinear", fill_value=0.0):
    if mode == "bilinear":
        return make_bilinear_function(arr, xgrid, ygrid, fill_value=fill_value)
    elif mode == "piecewise_constant":
        return make_piecewise_constant_function(arr, xgrid, ygrid, fill_value=fill_value)
    else:
        raise ValueError(f'mode must be either "bilinear" or "piecewise_constant", mode given: {mode}')
    
def l2_project_pixels_to_dg(arr, xgrid, ygrid, mesh, mode="bilinear", quad_order=None):
    Kx, Ky, p = mesh["Kx"], mesh["Ky"], mesh["p"]
    x_edges, y_edges = mesh["x_edges"], mesh["y_edges"]
    
    if quad_order is None:
        quad_order = max(2*p + 4, 8)

    f = make_sampled_function(arr, xgrid, ygrid, mode, fill_value=0.0)

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
            c_ij = Ls @ A
            
            coeffs[ey, ex, :, :] = c_ij  # store as (i,j)
            
    return {"mesh": mesh, 
            "coeffs": coeffs, 
            "basis": "orthonormal_legendre_tensor", 
            "construction": "l2_projection",
            "quad_order": quad_order
            }


def recommend_l2_mesh(DOF_x, DOF_y, deg, same_in_both=True):
    """
    As a practical heuristic, the DG mesh may be chosen so that each element
    spans at least p + 1 image samples in each coordinate direction, in order to
    avoid using a DG space that is too fine relative to the sampled data
    """
    
    p = int(deg)
    if same_in_both:
        K = min(DOF_x // (p + 1), DOF_y // (p + 1))
        K = max(K, 1)
        return K, K
    else:
        Kx = max(DOF_x // (p + 1), 1)
        Ky = max(DOF_y // (p + 1), 1)
        return Kx, Ky
    
    
# NOTE:
# The L2 projection does not require the nodal/modal compatibility
# DOF_x = Kx*(p+1), DOF_y = Ky*(p+1).
# However, the current SIAC postprocessing routine evaluates the filtered
# DG field at (p+1) points per element in each direction, so its output
# is naturally defined on a grid of shape (Ky*(p+1), Kx*(p+1)).
# Therefore, direct comparison of SIAC output with the original image grid
# only works when DOF_x = Kx*(p+1) and DOF_y = Ky*(p+1).

def l2_project_image_to_dg(
    recon,
    xlim=(-1, 1),
    ylim=(-1, 1),
    deg=3,
    mode="bilinear",
    Kx=None,
    Ky=None,
):
    """
    Project reconstruction data onto a DG(Q^p) space on a uniform mesh.
    """
    arr = _to_numpy(recon)

    if arr.ndim != 2:
        raise ValueError("Expected a 2D array/image for recon.")

    DOF_y, DOF_x = arr.shape

    xgrid, ygrid, dx, dy = build_image_grid(DOF_x, DOF_y, xlim=xlim, ylim=ylim)

    if Kx is None:
        Kx, Ky = recommend_l2_mesh(DOF_x=DOF_x, DOF_y=DOF_y, deg=deg, same_in_both=True)

    mesh = build_dg_mesh_l2(
        Kx=Kx, Ky=Ky, xlim=xlim, ylim=ylim, deg=deg,
        DOF_x=DOF_x, DOF_y=DOF_y
    )

    dg = l2_project_pixels_to_dg(arr, xgrid, ygrid, mesh, mode)

    dg["xgrid"] = xgrid
    dg["ygrid"] = ygrid
    dg["dx"] = dx
    dg["dy"] = dy

    return dg
