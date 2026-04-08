import numpy as np
from src.basis import eval_orthonormal_legendre_1d
from numpy.polynomial.legendre import leggauss


def l2_project_exact_func_to_dg_1d(func, mesh, poly_max_deg=None, quad_order=None, add_noise=False):
    """
    Project an exact callable function func(x, y) onto the modal DG space.

    Parameters
    ----------
    func : callable
        Exact function, vectorized in x and y.
    mesh : dict
        DG mesh dictionary.
    quad_order : int or None
        Tensor-product Gauss-Legendre quadrature order in each direction.

    Returns
    -------
    dg : dict
        Modal DG representation.
    """
    K = mesh["K"]
    edges = mesh["edges"]
    p = mesh["p"]
    order = p + 1

    # For 1D Gauss-Legendre, order n is exact up to degree 2n-1.
    # If func is polynomial with directional degrees k, and basis degree p,
    # then choose quad_order so that:
    #   quad_order >= (k + p + 1) / 2
    if quad_order is None:
        if poly_max_deg is None:
            quad_order = max(2 * p + 4, 8)
        else:
            quad_order = int(np.ceil( (poly_max_deg + p + 1) / 2 ))

    xi, wi = leggauss(quad_order)

    L = eval_orthonormal_legendre_1d(xi, p)   # shape (order, nq)

    coeffs = np.zeros((K, order), dtype=float)
    
    if add_noise:
        rng = np.random.default_rng(seed=62)
        h = mesh["h"]               # characteristic mesh length 
        pw = p + 1
        sigma = h ** pw

    for e in range(K):
        x0, x1 = edges[e], edges[e + 1]
        X = x0 + 0.5 * (x1 - x0) * (xi + 1.0)

        F = np.asarray(func(X), dtype=float)     # shape (nq,)
        if F.shape != X.shape:
            raise ValueError("func(X) must return an array of shape matching X.")
            
        if add_noise:
            noise = rng.standard_normal(size=F.shape) * sigma
            F += noise 
            
        G = wi * F
        c = L @ G       # change here?
        coeffs[e, :] = c

    dg = {
        "mesh": mesh,
        "coeffs": coeffs,
        "basis": "orthonormal_legendre_tensor",
        "construction": "l2_projection_exact_function",
        "quad_order": quad_order,
        "p": p,
    }
    return dg

def l2_project_exact_func_to_dg_2d(func, mesh, poly_max_deg=None, quad_order=None, add_noise=False):
    """
    Project an exact callable function func(x, y) onto the modal DG space.

    Parameters
    ----------
    func : callable
        Exact function, vectorized in x and y.
    mesh : dict
        DG mesh dictionary.
    quad_order : int or None
        Tensor-product Gauss-Legendre quadrature order in each direction.

    Returns
    -------
    dg : dict
        Modal DG representation.
    """
    Kx, Ky = mesh["Kx"], mesh["Ky"]
    x_edges, y_edges = mesh["x_edges"], mesh["y_edges"]
    p = mesh["p"]
    order = p + 1

    # For 1D Gauss-Legendre, order n is exact up to degree 2n-1.
    # In tensor-product 2D, exactness is determined separately in x and y.
    # If func is polynomial with directional degrees kx, ky, and basis degree p,
    # then choose quad_order so that:
    #   quad_order >= (kx + p + 1) / 2
    #   quad_order >= (ky + p + 1) / 2
    if quad_order is None:
        if poly_max_deg is None:
            quad_order = max(2 * p + 4, 8)
        else:
            quad_order = int(np.ceil( (poly_max_deg + p + 1) / 2 ))

    xi, wi = leggauss(quad_order)

    L = eval_orthonormal_legendre_1d(xi, p)   # shape (order, nq)

    coeffs = np.zeros((Ky, Kx, order, order), dtype=float)

    R, S = np.meshgrid(xi, xi, indexing="xy")   # shape (nq, nq)
    W = np.outer(wi, wi)                        # shape (nq, nq)
    
    if add_noise:
        rng = np.random.default_rng(seed=62)
        hx, hy = mesh["hx"], mesh["hy"]
        h = np.sqrt(hx**2 + hy**2)  #characteristic mesh length
        pw = p + 1
        sigma = h ** pw

    for ey in range(Ky):
        y0, y1 = y_edges[ey], y_edges[ey + 1]
        Y = y0 + 0.5 * (y1 - y0) * (S + 1.0)

        for ex in range(Kx):
            x0, x1 = x_edges[ex], x_edges[ex + 1]
            X = x0 + 0.5 * (x1 - x0) * (R + 1.0)
            
            F = np.asarray(func(X, Y), dtype=float)     # shape (nq, nq)
            if F.shape != X.shape:
                raise ValueError("func(X, Y) must return an array of shape matching X and Y.")
            
            if add_noise:
                noise = rng.standard_normal(size=F.shape) * sigma
                F += noise 
            
            G = W * F
            c_ij = L @ G @ L.T      # change here?
            coeffs[ey, ex, :, :] = c_ij

    dg = {
        "mesh": mesh,
        "coeffs": coeffs,
        "basis": "orthonormal_legendre_tensor",
        "construction": "l2_projection_exact_function",
        "quad_order": quad_order,
        "p": p,
    }
    return dg