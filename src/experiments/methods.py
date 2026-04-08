
from src.transforms import nodal_image_to_dg
from src.siac_modal import apply_siac_modal_dg_2d
from src.siac_fourier import apply_siac_fft_nd
from src.tomo import reconstruct_fbp

def run_post_recon_dg_siac(recon, xlim, ylim, p, moments, BSorder):
    """
    Apply SIAC postprocessing in a modal DG representation of a reconstructed image.

    The input image is interpreted as nodal data on a uniform grid (with pixel
    centers corresponding to nodal points), transformed into a modal DG
    representation, and then postprocessed using a SIAC kernel.

    Parameters
    ----------
    recon : numpy.ndarray or odl.element
        Reconstructed image (e.g. FBP output), assumed to be defined on a
        uniform Cartesian grid. Axis 0 corresponds to y, axis 1 to x (when converted to numpy array).
    xlim : tuple of float
        Domain limits in the x-direction, (xmin, xmax).
    ylim : tuple of float
        Domain limits in the y-direction, (ymin, ymax).
    p : int
        Polynomial degree of the DG basis (p >= 0).
    moments : int
        Number of SIAC moments to enforce. Must be even. The kernel reproduces
        polynomials up to degree moments + 1.
    BSorder : int
        Order of the B-spline used in the SIAC kernel.

    Returns
    -------
    Ustar : numpy.ndarray
        SIAC-postprocessed reconstruction, defined on the same grid as `recon`.

    Notes
    -----
    - Assumes a uniform grid.
    - Assumes pixel values correspond to cell-center nodal values.
    - Internally performs a nodal-to-modal DG transformation followed by SIAC convolution.
    """
    # nodal interpretation: nodal --> modal transform
    dg = nodal_image_to_dg(recon=recon, xlim=xlim, ylim=ylim, p=p, verbose=False) 
    
    # Postprocessor
    Ustar = apply_siac_modal_dg_2d(dg=dg, moments=moments, BSorder=BSorder)
    
    return Ustar

def run_post_recon_fourier_siac(recon, dy, dx, moments, BSorder, pad_mode="reflect"):
    """
    Apply SIAC postprocessing via Fourier-domain filtering to a reconstructed image.

    The SIAC kernel is applied as a separable Fourier multiplier along each spatial
    axis, using FFT-based convolution.

    Parameters
    ----------
    recon : numpy.ndarray or odl.element
        Reconstructed image (e.g. FBP output), assumed to be defined on a
        uniform Cartesian grid. Axis 0 corresponds to y, axis 1 to x (when converted to numpy array).
    dy : float
        Grid spacing in the y-direction.
    dx : float
        Grid spacing in the x-direction.
    moments : int
        Number of SIAC moments to enforce. Must be even. The kernel reproduces
        polynomials up to degree moments + 1.
    BSorder : int
        Order of the B-spline used in the SIAC kernel.
    pad_mode : str, optional
        Padding strategy used before FFT-based convolution. Default is "reflect".

    Returns
    -------
    Ustar : numpy.ndarray
        SIAC-postprocessed reconstruction, defined on the same grid as `recon`.

    Notes
    -----
    - Assumes a uniform grid.
    - Applies a separable 1D SIAC filter along each axis in Fourier space.
    - Boundary handling is controlled via `pad_mode`.
    """
    Ustar = apply_siac_fft_nd(
        arr=recon, 
        h_per_axis=(dy, dx), 
        axes=(0, 1), 
        moments=moments, 
        BSorder=BSorder, 
        pad_mode=pad_mode
    )
    
    return Ustar

def run_pre_recon_siac_detector(sinogram, A, d_det, moments, BSorder, pad_mode="reflect"):
    """
    Apply SIAC filtering in the detector variable prior to filtered backprojection.

    The input sinogram is filtered along the detector axis using a Fourier-domain
    SIAC kernel, and the result is reconstructed using standard FBP with a Ram-Lak filter.

    Parameters
    ----------
    sinogram : numpy.ndarray or odl.element
        Input sinogram (possibly noisy). Axis 0 corresponds to projection angles,
        axis 1 to detector coordinates (when converted to numpy array). 
    A : odl.Operator
        Forward operator (2D ray transform) defining the acquisition geometry.
    d_det : float
        Detector spacing.
    moments : int
        Number of SIAC moments to enforce. Must be even. The kernel reproduces
        polynomials up to degree moments + 1.
    BSorder : int
        Order of the B-spline used in the SIAC kernel.
    pad_mode : str, optional
        Padding strategy used before FFT-based convolution. Default is "reflect".

    Returns
    -------
    Ustar : numpy.ndarray
        Reconstructed image obtained from SIAC-filtered sinogram via FBP.

    Notes
    -----
    - SIAC filtering is applied only along the detector axis.
    - Reconstruction is performed using standard FBP with a Ram-Lak filter.
    - Assumes uniform detector spacing.
    """
    sino_np_siac = apply_siac_fft_nd(arr=sinogram, 
                                    h_per_axis = d_det, 
                                    axes=(1,),
                                    moments=moments, 
                                    BSorder=BSorder, 
                                    pad_mode=pad_mode
                                    )
    
    data_space = A.range
    ### Backproject the filtered sinogram ###
    sino_odl_siac = data_space.element(sino_np_siac)    # odl dataspace element
    fbp_siac_odl = reconstruct_fbp(sino_odl_siac, A, filter_name="Ram-Lak")
    Ustar = fbp_siac_odl.asarray()
    
    return Ustar