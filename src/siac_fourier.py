# This file contains functions to apply the SIAC filter axis-wise

import numpy as np
import math

from scipy.special import binom
import scipy.linalg   # SciPy Linear Algebra Library                                                                                                                
from scipy.linalg import lu
from scipy.linalg import lu_factor
from scipy.linalg import lu_solve


##_______________________Helper functions_____________________________________##

def siac_cgam_fourier(moments: int, BSorder: int):
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
    cgam = siac_cgam_fourier(moments, BSorder)
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