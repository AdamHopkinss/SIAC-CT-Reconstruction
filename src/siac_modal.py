import numpy as np
import math

from scipy.special import binom
import scipy.linalg   # SciPy Linear Algebra Library                                                                                                                
from scipy.linalg import lu
from scipy.linalg import lu_factor
from scipy.linalg import lu_solve

from scipy.interpolate import BSpline
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre


##_______________________Helper functions_____________________________________##

def siac_cgam(moments: int, BSorder: int):
    """
    Compute the SIAC cosine-series coefficients c_gamma by enforcing
    polynomial reproduction (moment conditions).

    moments : even integer r (number of enforced moments)
    BSorder : B-spline order n (controls smoothness / dissipation)

    Returns
    -------
    cgam : array of length RS+1 with symmetric coefficients used in the cosine sum
    """
    assert moments % 2 == 0, "moments should be even!"
    RS = int(np.ceil(moments / 2))
    numspline = moments + 1
    # Define matrix to determine kernel coefficients
    # Linear system A c = b encodes the moment conditions
    A=np.zeros((numspline, numspline), dtype=float)
    for m in np.arange(numspline):
        for gam in np.arange(numspline):
            component = 0.
            for n in np.arange(m+1):
                jsum = 0.
                jsum = sum((-1)**(j + BSorder-1) * binom(BSorder-1,j) * ((j - 0.5*(BSorder-2))**(BSorder+n) - (j - 0.5*BSorder)**(BSorder+n)) for j in np.arange(BSorder))
                
                component += binom(m,n)*(gam-RS)**(m-n) * math.factorial(n)/math.factorial(n+BSorder)*jsum

            A[m, gam] = component
                
    b = np.zeros((numspline))
    b[0] = 1    # consistency (zeroth moment): integral of kernel = 1
    
    #call the lu_factor function LU = linalg.lu_factor(A)
    Piv = scipy.linalg.lu_factor(A)
    #P, L, U = scipy.linalg.lu(A)
    #solve given LU and B
    cgam = scipy.linalg.lu_solve(Piv, b)

    return cgam

def local_pixel_center_nodes(order):
    """
    Original pixel centers mapped to [-1,1].
    """
    k = np.arange(order)    # array([0,1,...,order-1])
    return -1.0 + (2.0 * k + 1.0) / order 

def centered_cardinal_bspline(BSorder):
    """
    Centered cardinal B-spline of given order.
    Support: [-order/2, order/2]
    Integral = 1
    """
    degree = BSorder - 1
    knots = np.arange(BSorder + 1, dtype=float)

    spline = BSpline.basis_element(knots, extrapolate=False)
    support = (-BSorder / 2, BSorder / 2)

    def B(x):
        x = np.asarray(x)
        y = spline(x + BSorder/2)
        y = np.asarray(y, dtype=float)

        mask = (x < support[0]) | (support[1] < x)
        if y.ndim == 0:
            return 0.0 if mask else float(y)
        y[mask] = 0.0
        return y

    return B

# Helper function: orthonormal Legendre basis values on [-1,1]
# l_n(r) = sqrt((2n + 1) / 2) * P_n(r)
def eval_orthonormal_legendre_1d(x, p):
    """
    Evaluate orthonormal Legendre basis l_0,...,l_p at points x.
    
    Returns array of shape (p+1, len(x)).
    """
    x = np.asarray(x)
    out = np.zeros((p + 1, x.size))

    for n in range(p + 1):
        out[n, :] = np.sqrt((2*n + 1)/2.0) * eval_legendre(n, x)

    return out


def centered_cardinal_bspline(BSorder):
    """
    Centered cardinal B-spline of given order.

    Support: [-BSorder/2, BSorder/2]
    Integral = 1
    """
    degree = BSorder - 1
    knots = np.arange(BSorder + 1, dtype=float)
    spline = BSpline.basis_element(knots, extrapolate=False)
    support = (-BSorder / 2, BSorder / 2)

    def B(x):
        x = np.asarray(x)
        y = spline(x + BSorder / 2)
        y = np.asarray(y, dtype=float)

        mask = (x < support[0]) | (x > support[1])
        if y.ndim == 0:
            return 0.0 if bool(mask) else float(y)
        y[mask] = 0.0
        return y

    return B


def eval_orthonormal_legendre_1d(x, p):
    """
    Evaluate orthonormal Legendre basis l_0,...,l_p at points x.

    Returns array of shape (p+1, len(x)).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.zeros((p + 1, x.size))

    for n in range(p + 1):
        out[n, :] = np.sqrt((2 * n + 1) / 2.0) * eval_legendre(n, x)

    return out


def grab_integrals(eval_nodes, p, BSorder, BSsupport, quad_order=None):
    """
    Compute SIAC spline-basis integrals BSInt(mode, cell, node)
    using orthonormal Legendre basis on [-1,1].

    Parameters
    ----------
    eval_nodes : array_like, shape (p+1,)
        Reference evaluation nodes zeta_k in [-1,1].
    p : int
        DG polynomial degree.
    BSorder : int
        B-spline order.
    BSsupport : array_like of length 2
        Integer stencil bounds [min_shift, max_shift].
    quad_order : int or None
        Quadrature order for Gauss-Legendre integration.

    Returns
    -------
    BSInt : ndarray, shape (p+1, BSlen, p+1)
        BSInt[m, j, k] = integral block for mode m,
        support-index j, evaluation-node k.
    """
    eval_nodes = np.asarray(eval_nodes, dtype=float)
    order = p + 1

    BSmin, BSmax = int(BSsupport[0]), int(BSsupport[1])
    BSlen = BSmax - BSmin + 1

    B = centered_cardinal_bspline(BSorder)

    if quad_order is None:
        quad_order = max(2 * p + 4, 12)

    q_ref, w_ref = leggauss(quad_order)

    BIntL = np.zeros((order, BSlen, order))
    BIntR = np.zeros((order, BSlen, order))

    for k in range(order):
        zeta = eval_nodes[k]

        # Keep your current split for now.
        # MIGHT WANT TO LOOK HERE FOR ERRORS
        xicell = zeta - np.sign(zeta) * np.mod(BSorder, 2)

        # Left interval [-1, xicell]
        qL = 0.5 * ((xicell + 1.0) * q_ref + (xicell - 1.0))
        wL = 0.5 * (xicell + 1.0) * w_ref

        # Right interval [xicell, 1]
        qR = 0.5 * ((1.0 - xicell) * q_ref + (1.0 + xicell))
        wR = 0.5 * (1.0 - xicell) * w_ref

        phiL = eval_orthonormal_legendre_1d(qL, p)   # shape (order, nq)
        phiR = eval_orthonormal_legendre_1d(qR, p)

        for i in range(BSmin, BSmax + 1):
            j = i - BSmin

            bsL = B(0.5 * (zeta - qL) - i)   # shape (nq,)
            bsR = B(0.5 * (zeta - qR) - i)   # shape (nq,)

            for m in range(order):
                BIntL[m, j, k] = 0.5 * np.sum(wL * bsL * phiL[m, :])
                BIntR[m, j, k] = 0.5 * np.sum(wR * bsR * phiR[m, :])

    BSInt = BIntL + BIntR
    return BSInt
                
                
               
def pad_modal_coeffs_2d(coeffs, pad_x, pad_y):
    """
    Zero-pad modal DG coefficients in element space
    
    Assumes coeffs has shape (Ky, Kx, order, order)
    """
    Ky, Kx, order_y, order_x = coeffs.shape
    
    out = np.zeros(
        (Ky + 2*pad_y, Kx + 2*pad_x, order_y, order_x),
        dtype=coeffs.dtype
    )
    out[pad_y:pad_y + Ky, pad_x:pad_x + Kx, :, :] = coeffs
    return out


def apply_siac_modal_dg(dg, moments=None, BSorder=None):
    """
    Apply the SIAC filter to a modal DG solution.

    Parameters
    ----------
    dg : dict
        DG representation with keys "mesh" and "coeffs".
    moments : int or None
        Number of reproduced moments. Defaults to 2*p.
    BSorder : int or None
        B-spline order. Defaults to p+1.
    """
    mesh = dg["mesh"]
    coeffs = dg["coeffs"]
    
    p = mesh["p"]
    order = mesh["order"]
    Kx = mesh["Kx"]
    Ky =  mesh["Ky"]
    hx = mesh["hx"]         # element spacing
    hy = mesh["hy"]
    
    if moments is None:
        moments = 2 * p     # Standard choices
    if BSorder is None:
        BSorder = p + 1
    
    # evaluation grid
    xgrid = dg["xgrid"]
    ygrid = dg["ygrid"]
    dx = dg["dx"]           # physical grid spacings
    dy = dg["dy"]
    
    
    # reference element
    nodes = local_pixel_center_nodes(order)
    
    #### Postprocessor ####
    BSknots = np.linspace(-BSorder/2, BSorder/2, BSorder+1)
    BSsupport = [np.floor(BSknots[0]), np.ceil(BSknots[-1])]
    BSsupport = np.array(BSsupport, dtype=int)
    BSlen = int((BSsupport[1] - BSsupport[0]) + 1)
    
    # Grab the symmetric SIAC weights
    cgam = siac_cgam(moments, BSorder)
    
    # Calculate integrals of B-Splines
    

    
    
    # Kernel half-width measured in element units
    halfwidth =  (moments + BSorder) / 2
    
    # The original phantom has compact support, so we pad the mesh 
    # with ghost elements where all coefficients are zero
    pad = int(np.max(np.abs(88888)))       # number of ghost elements
    
    # Keep originals for clipping back later
    orig_Kx, orig_Ky = Kx, Ky
    orig_coeffs = coeffs
    coeffs = pad_modal_coeffs_2d(coeffs, pad_x=pad, pad_y=pad)
        
    mesh_work = mesh.copy()
    
    mesh_work["Kx"] = Kx + 2*pad
    mesh_work["Ky"] = Ky + 2*pad
    Kx = mesh_work["Kx"]
    Ky = mesh_work["Ky"]
    
    