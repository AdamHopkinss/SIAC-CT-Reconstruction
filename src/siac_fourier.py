# This file contains functions to apply the SIAC filter axis-wise

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
    cgamtemp = scipy.linalg.lu_solve(Piv, b)
    cgam = np.zeros((RS + 1))
    for igam in np.arange(RS+1):
        cgam[igam] = cgamtemp[RS-igam]
    
    # Sanity check: coefficients should sum to ~1 (can be outcommented)
    # sumcoeff = sum(cgamtemp[n] for n in np.arange(numspline))
    # print('Sum of coefficients',sumcoeff) 
    return cgam
    

def siac_hat_1d(omega: np.ndarray, cgam: np.ndarray, BSorder: int, h: float):
    """
    omega: radian frequencies (should be same shape as FFT freq grid)
    h: grid spacing in the corresponding direction (dx or dy)
    """
    RS = len(cgam) - 1
    
    # dimensionless freq variable
    w = h * omega
    #w = omega
    
    # cosine sum
    cgamterm = cgam[0] * np.ones_like(w, dtype=float)
    for igam in range(1, RS + 1):
        cgamterm +=  2.0 * (cgam[igam] * np.cos(igam * w))
    
    # numpy sinc(x) = sin(pi x)/(pi x)
    # sin(omega/2)/(omega/2) = sinc(omega / (2*pi))
    
    sinc_factor = np.sinc(w / (2.0 * np.pi)) ** BSorder
    
    return sinc_factor * cgamterm


def _siac_support_pad(moments: int, BSorder: int) -> int:
    # mirror your existing heuristic
    R = int(np.ceil((moments + BSorder + 1) / 2))
    return R + 2


def _siac_freq_response_1d(N: int, d: float, moments: int, BSorder: int, cgam: np.ndarray):
    omega = 2.0 * np.pi * np.fft.fftfreq(N, d=d)      # radian freq
    S = siac_hat_1d(omega, cgam, BSorder, h=d)        # shape (N,)
    return S
##____________________________________________________________________________##

##_______________________Main function________________________________________##

def apply_siac_fft_nd(arr: np.ndarray,
                      h_per_axis,
                      moments: int = 2,
                      BSorder: int = 2,
                      axes=(0, 1),
                      pad_mode: str = "reflect"):
    """
    Apply SIAC via 1D FFT along specified axis/axes of an N-D array.

    Parameters
    ----------
    arr : ndarray
        Input array (image, sinogram, volume, etc.)
    h_per_axis : float or sequence
        Grid spacing per axis. If scalar, uses same spacing for all axes.
        If sequence, must have length of how many axes exists and spacing is taken as h_per_axis[axis].
    axes : int or iterable of int
        Which axes to filter along (e.g. (0,1) for 2D image; (0) for first axis only etc.).
    """
    x = np.asarray(arr, dtype=float)

    if np.isscalar(h_per_axis):
        h_per_axis = [float(h_per_axis)] * x.ndim
    else:
        h_per_axis = list(h_per_axis)
        if len(h_per_axis) != x.ndim:
            raise ValueError("h_per_axis must be scalar or length arr.ndim")

    # normalize axes
    if isinstance(axes, (int, np.integer)):
        axes = [int(axes)]
    else:
        axes = list(axes)

    axes = [ax if ax >= 0 else ax + x.ndim for ax in axes]

    # coefficients once
    cgam = siac_cgam(moments, BSorder)
    pad = _siac_support_pad(moments, BSorder)

    # Padding is applied ONCE in all dimensions.
    # If padding in the axes loop, then the second padding can be affected by the first SIAC result (not relevant if SIAC applied to one axis only)
    pad_width = [(pad, pad)] * x.ndim
    xpad = np.pad(x, pad_width, mode=pad_mode)

    # apply along each requested axis
    for ax in axes:
        h = h_per_axis[ax]
        Np = xpad.shape[ax]

        omega = 2.0 * np.pi * np.fft.fftfreq(Np, d=h)
        S = siac_hat_1d(omega, cgam, BSorder, h=h)  # (Np,)

        F = np.fft.fft(xpad, axis=ax)

        shape = [1] * xpad.ndim
        shape[ax] = Np
        F *= S.reshape(shape)

        xpad = np.real(np.fft.ifft(F, axis=ax))

    # crop once
    crop_slices = []
    for ax in range(x.ndim):
        start = pad
        stop  = pad + x.shape[ax]
        crop_slices.append(slice(start, stop))

    return xpad[tuple(crop_slices)]

##____________________________________________________________________________##

##___________________________SIAC_on_Modal_DG_________________________________##

from scipy.interpolate import BSpline
from numpy.polynomial.legendre import leggauss

from scipy.special import eval_legendre
from src.dg_utils import eval_dg_modal_on_img_grid

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

def centered_cardinal_bspline(order):
    """
    Centered cardinal B-spline of given order.
    Support: [-order/2, order/2]
    Integral = 1
    """
    degree = order - 1
    knots = np.arange(order + 1, dtype=float)

    spline = BSpline.basis_element(knots, extrapolate=False)
    support = (-order / 2, order / 2)

    def B(x):
        x = np.asarray(x)
        y = spline(x + order/2)
        y = np.asarray(y, dtype=float)

        mask = (x < support[0]) | (support[1] < x)
        if y.ndim == 0:
            return 0.0 if mask else float(y)
        y[mask] = 0.0
        return y

    return B

def build_siac_kernel_1d(cgam, BSorder, gamma_vals):
    """
    Returns K(t) = sum_g c_g B(t - gamma_g)
    in cell units.
    """
    B = centered_cardinal_bspline(BSorder)

    def K(t):
        t = np.asarray(t)
        out = np.zeros_like(t, dtype=float)
        for cg, gam in zip(cgam, gamma_vals):
            out += cg * B(t - gam)
        return out

    return K

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
    

def siac_kernel_support(BSorder, gamma_vals):
    """
    Support of K(t) = sum_g c_g B(t-g), where B has support [-BSorder/2, BSorder/2].
    """
    a = np.min(gamma_vals) - BSorder / 2.0
    b = np.max(gamma_vals) + BSorder / 2.0
    return a, b

def compute_siac_shifts_1d(BSorder, gamma_vals):
    """
    Integer element shifts s that can contribute, based on
    K((xi-eta)/2 - s), with (xi-eta)/2 in [-1,1].
    """
    a, b = siac_kernel_support(BSorder, gamma_vals)
    smin = int(np.ceil(-1.0 - b))
    smax = int(np.floor(1.0 - a))
    return np.arange(smin, smax + 1)

def local_pixel_center_nodes(order):
    """
    Original pixel centers mapped to [-1,1].
    """
    k = np.arange(order)    # array([0,1,...,order-1])
    return -1.0 + (2.0 * k + 1.0) / order 

def precompute_siac_matrix_1d_modal(kernel, p, eval_nodes, shifts, quad_order=None):
    """
    Precompute SIAC convolution weights for a 1D modal DG basis.

    A[i, q, m] approximates
        1/2 \int_{-1}^1 K((xi_i - eta)/2 - shift_q) phi_m(eta) d eta

    where:
      - xi_i     : evaluation node on the reference element
      - shift_q  : element offset in the SIAC stencil
      - phi_m    : orthonormal Legendre basis function of degree m
      - K        : full SIAC kernel

    Later, these weights are applied to DG modal coefficients from neighboring
    elements to evaluate the SIAC-filtered solution.
    """
    n_modes = p + 1

    if quad_order is None:
        quad_order = max(2 * p + 4, 12)

    eta_q, w_q = leggauss(quad_order)
    phi_at_q = eval_orthonormal_legendre_1d(eta_q, p)   # shape (n_modes, quad_order)

    A = np.zeros((len(eval_nodes), len(shifts), n_modes), dtype=float)

    for i_eval, xi in enumerate(eval_nodes):
        for i_shift, shift in enumerate(shifts):
            kernel_arg = 0.5 * (xi - eta_q) - shift
            K_q = kernel(kernel_arg)

            # Quadrature approximation of:
            # 1/2 eta K((xi-eta)/2 - shift) phi_m(eta) d eta
            A[i_eval, i_shift, :] = phi_at_q @ (0.5 * w_q * K_q)

    return A

def apply_x_pass(coeffs, A, shifts, padded=False):
    """
    Apply the SIAC operator in the x-direction.

    Parameters
    ----------
    coeffs : ndarray, shape (Ky, Kx, order, order)
        Modal DG coefficients:
            coeffs[ey, ex, my, mx]
    A : ndarray, shape (order, nshift, order)
        1D SIAC modal matrix:
            A[ix, q, mx]
    shifts : ndarray, shape (nshift,)
        Integer x-element offsets
    fill_boundary : bool
        If True, copy coeffs into untouched boundary region.
        If False, leave boundary values as zero.

    Returns
    -------
    W : ndarray, shape (Ky, Kx, order, order)
        Intermediate field:
            W[ey, ex, my, ix]
        modal in y, evaluated in x
    x_valid : tuple
        Valid x-element range (x0, x1), meaning ex in range(x0, x1)

    coeffs shape:   (Ky, Kx, order, order)
    A shape:        (order, len(shifts), order)
    
    fix row ey, fix y-mode my, evaluate at target x-node ix
    W[ey,ex,my,ix] = sum_{q,mx} A[ix,q,mx] coeffs[ey, ex + sq, my, mx]
    for shifts s
    """
    Ky, Kx, order, _ = coeffs.shape
    
    W = np.zeros((Ky, Kx, order, order), dtype=float)
    
    smin = shifts.min()
    smax = shifts.max()
    
    x0 = -smin
    x1 = Kx - smax      # ensure kernel stays internal
    
    for ey in range(Ky):            # allow all rows
        for ex in range(x0, x1):    # only x interior
            # extract neighbours
            neigh = coeffs[ey, ex + shifts, :, :]
            
            # W_block[my, ix] = sum_{q,mx} neigh[q,my,mx] * A[ix,q,mx]
            W[ey, ex, :, :] = np.einsum("qym,iqm->yi", neigh, A)

            # for my in range(order):
            #     for ix in range(order):
            #         acc = 0.0
            #         for q, s in enumerate(shifts):
            #             for mx in range(order):
            #                 acc += A[ix, q, mx] * coeffs[ey, ex + s, my, mx]
            #         W[ey, ex, my, ix] = acc
    
    # Fill in the untouched values
    # if fill_boundary:
    #     # copy unfiltered values on left/right boundary strips
    #     W[:, :x0, :, :] = coeffs[:, :x0, :, :]
    #     W[:, x1:, :, :] = coeffs[:, x1:, :, :]

    return W, (x0, x1)
            
    
def apply_y_pass(W, A, shifts):
    """
    Apply the SIAC operator in the y-direction.

    Parameters
    ----------
    W : ndarray, shape (Ky, Kx, order, order)
        Intermediate field after x-pass:
            W[ey, ex, my, ix]
        modal in y, evaluated in x
    A : ndarray, shape (order, nshift, order)
        1D SIAC modal matrix:
            A[iy, q, my]
    shifts : ndarray, shape (nshift,)
        Integer y-element offsets
    x_valid : tuple or None
        Valid x-element range (x0, x1) returned from the x-pass.
        If provided, only this x-interior is processed.
        If None, all x-columns are processed.
    fill_boundary : bool
        If True, copy W into untouched boundary region.
        If False, leave boundary values as zero.

    Returns
    -------
    Ustar : ndarray, shape (Ky, Kx, order, order)
        Filtered field:
            Ustar[ey, ex, iy, ix]
        evaluated in both y and x
    y_valid : tuple
        Valid y-element range (y0, y1), meaning ey in range(y0, y1)

    W shape:        (Ky, Kx, order, order)
    A shape:        (order, len(shifts), order)

    fix column ex, fix x-node ix, evaluate at target y-node iy
    Ustar[ey,ex,iy,ix] = sum_{q,my} A[iy,q,my] * W[ey + sq, ex, my, ix]
    for shifts s
    """
    Ky, Kx, order, _ = W.shape

    Ustar = np.zeros((Ky, Kx, order, order), dtype=float)

    smin = shifts.min()
    smax = shifts.max()

    y0 = -smin
    y1 = Ky - smax      # ensure kernel stays internal

    for ex in range(Kx):            # allow all columns
        for ey in range(y0, y1):    # only y interior
            # extract neighbours
            neigh = W[ey + shifts, ex, :, :]

            # U_block[iy, ix] = sum_{q,my} neigh[q,my,ix] * A[iy,q,my]
            Ustar[ey, ex, :, :] = np.einsum("qmi,jqm->ji", neigh, A)

            # for iy in range(order):
            #     for ix in range(order):
            #         acc = 0.0
            #         for q, s in enumerate(shifts):
            #             for my in range(order):
            #                 acc += A[iy, q, my] * W[ey + s, ex, my, ix]
            #         Ustar[ey, ex, iy, ix] = acc

    # # Fill in the untouched values
    # if fill_boundary:
    #     # copy x-boundary strips from W
    #     Ustar[:, :x0, :, :] = W[:, :x0, :, :]
    #     Ustar[:, x1:, :, :] = W[:, x1:, :, :]

    #     # copy y-boundary strips from W
    #     Ustar[:y0, :, :, :] = W[:y0, :, :, :]
    #     Ustar[y1:, :, :, :] = W[y1:, :, :, :]

    return Ustar, (y0, y1)


def scatter_local_blocks_to_image(Ustar):
    """
    Convert local element blocks to a global image.

    Parameters
    ----------
    Ustar : ndarray, shape (Ky, Kx, order, order)
        Local filtered values:
            Ustar[ey, ex, iy, ix]

    Returns
    -------
    img : ndarray, shape (Ky*order, Kx*order)
        Global image on the original pixel-center grid
    """
    Ky, Kx, order_y, order_x = Ustar.shape
    assert order_y == order_x
    order = order_y

    img = Ustar.transpose(0, 2, 1, 3).reshape(Ky * order, Kx * order)
    return img

def apply_siac_modal_dg(dg, moments=None, BSorder=None, fill_boundary=False, pad_boundary=True):
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
    fill_boundary : bool
        If True, fill any region not covered by SIAC with pointwise DG values.
    pad_boundary : bool
        If True, extend the modal element grid with zero-valued ghost elements
        so that the SIAC kernel can cover the whole original domain, then clip back.
    """
    mesh = dg["mesh"]
    coeffs = dg["coeffs"]
    
    p = mesh["p"]
    order = mesh["order"]
    Kx = mesh["Kx"]
    Ky =  mesh["Ky"]
    
    if moments is None:
        moments = 2 * p     # Standard choices
    if BSorder is None:
        BSorder = p + 1
    
    # Local evaluation nodes on reference element [-1,1]
    nodes = local_pixel_center_nodes(order)
    
    # Build SIAC kernel 
    cgam_temp = siac_cgam(moments, BSorder)  # returns only one half
    RS = int(np.ceil(moments / 2))
    gamma_vals = np.arange(-RS, RS + 1)
    cgam = np.array([cgam_temp[abs(g)] for g in gamma_vals], dtype=float)
    
    shifts = compute_siac_shifts_1d(BSorder=BSorder, gamma_vals=gamma_vals)
    kernel = build_siac_kernel_1d(cgam=cgam, BSorder=BSorder, gamma_vals=gamma_vals)
    
    # Kernel half-width measured in element units
    halfwidth =  (moments + BSorder) / 2
    pad = int(np.max(np.abs(shifts)))       # number of ghost elements
    
    # print(f"halfwidth = {halfwidth}")
    # print("pad =", pad)
    # print("shifts =", shifts)
    # print("required pad =", max(-shifts.min(), shifts.max()))
    # print("\nrequired: pad >= max(|shifts|)")
    
    # Keep originals for clipping back later
    orig_Kx, orig_Ky = Kx, Ky
    orig_coeffs = coeffs
    
    # Optional zero-padding in element space
    if pad_boundary:
        coeffs = pad_modal_coeffs_2d(coeffs, pad_x=pad, pad_y=pad)
        
        mesh_work = mesh.copy()
        
        mesh_work["Kx"] = Kx + 2*pad
        mesh_work["Ky"] = Ky + 2*pad
        Kx = mesh_work["Kx"]
        Ky = mesh_work["Ky"]
    else:
        mesh_work = mesh
        # admissibility check
        if Kx <= 2 * halfwidth or Ky <= 2 * halfwidth:
            raise ValueError("SIAC kernel support too wide for DG mesh")
        
    # print("coeffs shape:", coeffs.shape)
    # print("orig_Kx, orig_Ky:", orig_Kx, orig_Ky)
    # print("pad:", pad)
    # print("expected: coeffs.shape == (orig_Ky + 2*pad, orig_Kx + 2*pad, order, order)")
    
    # Build SIAC 1D modal matrix (normalized legendre basis)
    A = precompute_siac_matrix_1d_modal(kernel=kernel, p=p, eval_nodes=nodes, shifts=shifts)
    # A.shape = (order, len(shifts), order)
    
    # # Apply SIAC passes
    W, x_bounds = apply_x_pass(coeffs=coeffs, A=A, shifts=shifts)
    
    # print("x_bounds:", x_bounds)
    # print("original x-range:", (pad, pad + orig_Kx))
    # print("required: pad >= x_bounds[0], pad + orig_Kx <= x_bounds[1]")
    
    Ustar, y_bounds = apply_y_pass(W=W, A=A, shifts=shifts)
    
    # print("y_bounds:", y_bounds)
    # print("original y-range:", (pad, pad + orig_Ky))
    # print("required: pad >= y_bounds[0], pad + orig_Ky <= y_bounds[1]")
    # if padded, clip back to original element mesh
    
    # print("Ustar shape before clipping:", Ustar.shape)
    if pad_boundary and pad > 0:
        Ustar = Ustar[
            pad:-pad,   # y-elements
            pad:-pad,   # x-elements
            :, :
        ]
    # print("Ustar shape after clipping:", Ustar.shape)
    # print(f"expected: ({orig_Ky}, {orig_Kx}, {order}, {order})")
    
    # Scatter result onto the grid (Ky*(p+1), Kx*(p+1))
    img_siac = scatter_local_blocks_to_image(Ustar)
    
    # No fallback requested
    if not fill_boundary:
        return img_siac

    # In padded mode, SIAC should already cover everything
    if pad_boundary:
        return img_siac
    
    # Non-padded -> fallback-to-DG mode
    img_dg = eval_dg_modal_on_img_grid(dg)

    # valid SIAC range in element indices
    y0, y1 = y_bounds
    x0, x1 = x_bounds

    # convert to pixel/image indices
    px0 = x0 * order
    px1 = x1 * order
    py0 = y0 * order
    py1 = y1 * order

    img_out = img_dg.copy()
    img_out[py0:py1, px0:px1] = img_siac[py0:py1, px0:px1]
    return img_out