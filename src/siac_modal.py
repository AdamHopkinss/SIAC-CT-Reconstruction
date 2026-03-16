import numpy as np
import math

from scipy.special import binom
import scipy.linalg   # SciPy Linear Algebra Library                                                                                                                
from scipy.linalg import lu
from scipy.linalg import lu_factor
from scipy.linalg import lu_solve


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
    pad = int(np.max(np.abs(shifts)))       # number of ghost elements
    
    # Keep originals for clipping back later
    orig_Kx, orig_Ky = Kx, Ky
    orig_coeffs = coeffs
    coeffs = pad_modal_coeffs_2d(coeffs, pad_x=pad, pad_y=pad)
        
    mesh_work = mesh.copy()
    
    mesh_work["Kx"] = Kx + 2*pad
    mesh_work["Ky"] = Ky + 2*pad
    Kx = mesh_work["Kx"]
    Ky = mesh_work["Ky"]
    
    