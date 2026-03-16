import odl
import numpy as np


def solve_tikhonov(A, data, alpha, L=None, niter=50, x0=None, callback=None):
    r"""
    Solve the Tikhonov-regularized least squares problem

        min_x  0.5 ||A x - data||_2^2
            + 0.5 alpha ||L x||_2^2.

    Parameters
    ----------
    A : odl.Operator
        Forward operator.
    data : element of A.range
        Measured data.
    alpha : float
        Regularization parameter.
    L : odl.Operator, optional
        Linear regularization operator. If None, the identity is used.
    niter : int
        Number of CG iterations.
    x0 : element of A.domain, optional
        Initial guess.
    callback : callable, optional
        ODL callback.

    Returns
    -------
    x : element of A.domain
        Reconstructed solution.
    """
    X = A.domain
    if L is None:
        L = odl.IdentityOperator(X)
    H = A.adjoint*A + alpha*(L.adjoint*L)
    b = A.adjoint(data)
    x = X.zero() if x0 is None else x0.copy()
    odl.solvers.conjugate_gradient(H, x, b, niter=niter, callback=callback)
    return x

from src.siac_fourier import siac_cgam, siac_hat_1d

def make_siac_operator_odl(space, moments=2, BSorder=2):
    r"""
    Construct the SIAC operator K: space -> space as an ODL Fourier-multiplier
    Kx = IFFT( Khat * FFT(x) )
    
    Notes: 
    This correpsonds to periodic convolution on the discrete grid.
    The problem is to mimic the reflect-padding behavior, then we would have to
    pad/crop outside the operator, or use some different construction (but then teh adjoint is non-trivial).
    """
    fft = odl.trafos.FourierTransform(space)
    ifft = fft.inverse
    
    dx, dy = map(float, space.cell_sides)   # grid spacing
    
    # SIAC cosine coefficients
    cgam = siac_cgam(moments=moments, BSorder=BSorder)
    
    # Build Khat on the Fourier grid
    pts = fft.range.grid.points()
    wx = pts[:, 0].reshape(fft.range.shape) # radian frequencies in x
    wy = pts[:, 1].reshape(fft.range.shape) # radian frequencies in y
    
    Khatx = siac_hat_1d(omega=wx,cgam=cgam, BSorder=BSorder, h=dx)
    Khaty = siac_hat_1d(omega=wy,cgam=cgam, BSorder=BSorder, h=dy)
    
    # Seperable 2D response
    Khat = (Khatx * Khaty).astype(np.complex64)
    
    assert Khat.shape == fft.range.shape
    
    # ODL operators
    mult = odl.MultiplyOperator(fft.range.element(Khat))
    K = odl.RealPart(space) * ifft * mult * fft
    return K


def solve_siac(A, data, alpha, moments=2, BSorder=2, niter=50, x0=None, callback=None):
    r"""
    Solve the SIAC-regularized least squares problem

        min_x  0.5 ||A x - data||_2^2
            + 0.5 alpha ||(I - K) x||_2^2,

    where K is a SIAC convolution operator enforcing polynomial reproduction.

    Parameters
    ----------
    A : odl.Operator
        Forward operator.
    data : element of A.range
        Measured data.
    alpha : float
        Regularization parameter.
    K : odl.Operator
        SIAC operator.
    niter : int
        Number of CG iterations.
    x0 : element of A.domain, optional
        Initial guess.
    callback : callable, optional
        ODL callback.

    Returns
    -------
    x : element of A.domain
        Reconstructed solution.
    """
    X = A.domain
    # Create the operator K
    K = make_siac_operator_odl(space=X, moments=moments, BSorder=BSorder)
    
    I = odl.IdentityOperator(X)
    B = I - K
    H = A.adjoint*A + alpha*(B.adjoint*B)
    b = A.adjoint(data)
    
    x = X.zero() if x0 is None else x0.copy()
    odl.solvers.conjugate_gradient(H, x, b, niter=niter, callback=callback)
    return x


"""
TV and TGV solvers adapted from ODL example implementations.

Original source:
https://github.com/odlgroup/odl/tree/master/examples/solvers
BSD-3-Clause License.

Modified and refactored for MSc Thesis (Adam Hopkins, 2026).
"""


import odl
from odl.solvers.functional.functional import Functional


class _ShiftedL2NormSquaredConj(Functional):
    r"""F*(p) = 0.5||p||^2 + <p, data>.

    Prox:
        prox_{sigma F*}(p) = (p - sigma*data) / (1 + sigma)

    IMPORTANT:
    ODL's PDHG expects proximal factories to return an odl.Operator (with
    .domain/.range), not a plain Python callable.
    """

    def __init__(self, space, data):
        super().__init__(space, linear=False)
        self.data = data

    def __call__(self, p):
        return 0.5 * p.inner(p) + p.inner(self.data)

    @property
    def proximal(self):
        data = self.data
        space = self.domain

        def prox_factory(sigma):
            # prox(p) = (p - sigma*data)/(1+sigma) = a*p + c
            a = 1.0 / (1.0 + sigma)
            c = (-sigma * a) * data

            # Return an ODL operator (affine map = scaling + constant shift)
            return odl.ScalingOperator(space, a) + odl.ConstantOperator(c, domain=space)

        return prox_factory


class ShiftedL2NormSquared(Functional):
    r"""F(y) = 0.5 * ||y - data||^2.

    Why:
    ODL 0.8.3 can fail for `0.5*L2NormSquared(space).translated(data)` in PDHG
    (translation-related prox path). We keep the same fidelity but implement
    the conjugate prox in closed form.
    """

    def __init__(self, space, data):
        super().__init__(space, linear=False)
        self.data = data
        self._conj = _ShiftedL2NormSquaredConj(space, data)

    def __call__(self, y):
        d = y - self.data
        return 0.5 * d.inner(d)

    @property
    def convex_conj(self):
        # PDHG calls g.convex_conj.proximal, so we must return a Functional here.
        return self._conj


def solve_tv(A, data, alpha=1e-2, niter=200, x0=None, isotropic=True, 
             grad_method="forward", pad_mode="symmetric",
             tau=None, sigma=None, op_norm=None, callback=None):
    r"""
    Solve the TV-regularized least squares problem using PDHG:

        min_x  0.5 ||A x - data||_2^2
            + 0.5 alpha TV(x),

    where TV(x) = ||∇x||_1.

    Two variants are supported:

    • isotropic TV:
        TV(x) = sum_i sqrt( (D_x x_i)^2 + (D_y x_i)^2 )

    • anisotropic TV:
        TV(x) = sum_i ( |D_x x_i| + |D_y x_i| )

    Parameters
    ----------
    A : odl.Operator
        Forward operator.
    data : element of A.range
        Measured data.
    alpha : float
        Regularization parameter.
    niter : int
        Number of PDHG iterations.
    x0 : element of A.domain, optional
        Initial guess.
    isotropic : bool
        If True, use isotropic TV (default).
        If False, use anisotropic TV.
    grad_method : str
        Discretization method for the gradient.
    pad_mode : str
        Boundary condition for derivatives.
    tau, sigma : float, optional
        PDHG step sizes. If None, estimated automatically.
    op_norm : float, optional
        Precomputed operator norm.
    callback : callable, optional
        ODL callback.

    Returns
    -------
    x : element of A.domain
        Reconstructed solution.
    """
    X = A.domain
    G = odl.Gradient(X, method=grad_method, pad_mode=pad_mode)
    V = G.range
    
    # We form K(x) = [A x, G x]
    op = odl.BroadcastOperator(A, G)
    
    # l2 = 0.5 * odl.solvers.L2NormSquared(A.range).translated(data)
    
    # NOTE: Avoid ODL's `.translated(data)` quadratic prox (can type-error in 0.8.3).
    # This is the same data fidelity: 0.5*||Ax - data||^2, but with a safe closed-form prox of the conjugate.
    l2 = ShiftedL2NormSquared(A.range, data)
    
    f = odl.solvers.ZeroFunctional(X)

    # TV functional (Isotropic or Anisotropic)
    if isotropic:
        tv = 0.5 * alpha * odl.solvers.GroupL1Norm(V)
    else:
        tv = 0.5 * alpha * odl.solvers.L1Norm(V)
    g = odl.solvers.SeparableSum(l2, tv)
    
    # Step sizes
    if op_norm is None:
        # Estimated operator norm, add 10 percent 
        # to ensure ||K||_2^2 * sigma * tau < 1
        op_norm = op_norm = 1.1 * odl.power_method_opnorm(op)
    if tau is None:
        tau = 1.0 / op_norm
    if sigma is None:
        sigma = 1.0 / op_norm
    
    x = op.domain.zero() if x0 is None else x0.copy()
    
    odl.solvers.pdhg(x, f, g, op, niter=niter, tau=tau, sigma=sigma, callback=callback)
    
    return x


def solve_tgv2(A, data, alpha=4e-1, beta=1.0, niter=200, x0=None, y0=None,
               grad_method="forward", pad_mode="symmetric", tau=None, sigma=None, 
               op_norm=None, callback=None,):
    r"""
    Solve the second-order TGV-regularized least squares problem using PDHG:

        min_x  0.5 ||A x - data||_2^2
            + 0.5 alpha TGV^2_beta(x),

    where the second-order Total Generalized Variation is defined by

        TGV^2_beta(x) = min_y  ||∇x - y||_1 + beta ||E y||_1,

    and E denotes the symmetrized gradient operator.

    The problem is solved in saddle form over (x, y):

        min_{x,y}  0.5 ||A x - data||_2^2
                + 0.5 alpha ||∇x - y||_1
                + 0.5 alpha beta ||E y||_1.

    Parameters
    ----------
    A : odl.Operator
        Forward operator.
    data : element of A.range
        Measured data.
    alpha : float
        First-order TGV weight.
    beta : float
        Second-order balancing parameter.
    niter : int
        Number of PDHG iterations.
    x0 : element of A.domain, optional
        Initial guess for x.
    y0 : element of Gradient(A.domain).range, optional
        Initial guess for auxiliary variable y.
    grad_method : str
        Discretization method for the gradient.
    pad_mode : str
        Boundary condition for derivatives.
    tau, sigma : float, optional
        PDHG step sizes.
    op_norm : float, optional
        Precomputed operator norm.
    callback : callable, optional
        ODL callback.

    Returns
    -------
    x : element of A.domain
        Reconstructed image.
    y : element of Gradient(A.domain).range
        Auxiliary vector field.
    """
    U = A.domain

    # Gradient for Gx
    G = odl.Gradient(U, method=grad_method, pad_mode=pad_mode)
    V = G.range  # vector-field space

    # Backward partial derivatives for symmetric gradient
    Dx = odl.PartialDerivative(U, 0, method="backward", pad_mode=pad_mode)
    Dy = odl.PartialDerivative(U, 1, method="backward", pad_mode=pad_mode)

    # Symmetrized gradient E acting on vector fields y=(y1,y2).
    # Workaround used in the ODL example: duplicate the shear component
    # to emulate weighting (since weighted product space wasn't supported there).
    E = odl.operator.ProductSpaceOperator(
        [[Dx, 0],
         [0, Dy],
         [0.5 * Dy, 0.5 * Dx],
         [0.5 * Dy, 0.5 * Dx]]
    )
    W = E.range

    # Joint variable z = (x, y) in U x V
    domain = odl.ProductSpace(U, V)

    # Operator K(z) = (A x, G x - y, E y)
    op = odl.BroadcastOperator(
        A * odl.ComponentProjection(domain, 0),
        odl.ReductionOperator(G, odl.ScalingOperator(V, -1)),
        E * odl.ComponentProjection(domain, 1)
    )
    
    # l2 = 0.5 * odl.solvers.L2NormSquared(A.range).translated(data)
    # NOTE: Avoid ODL's `.translated(data)` quadratic prox (can type-error in 0.8.3).
    # Same fidelity term 0.5*||Ax - data||^2, but with robust prox_{σ F*}(p) = (p - σ data)/(1+σ).
    l2 = ShiftedL2NormSquared(A.range, data)
    
    f = odl.solvers.ZeroFunctional(domain)

    # Use isotropic L1 (ODL's L1Norm on product spaces is "global L1, local L2"
    # as described in your docstring).
    l1_1 = 0.5 * alpha * odl.solvers.L1Norm(V)      # ||Gx - y||_1
    l1_2 = 0.5 * alpha * beta * odl.solvers.L1Norm(W)  # ||E y||_1

    g = odl.solvers.SeparableSum(l2, l1_1, l1_2)

    # Step sizes
    if op_norm is None:
        # Estimated operator norm, add 10 percent 
        # to ensure ||K||_2^2 * sigma * tau < 1
        op_norm = 1.1 * odl.power_method_opnorm(op)
    if tau is None:
        tau = 1.0 / op_norm
    if sigma is None:
        sigma = 1.0 / op_norm

    z = op.domain.zero()
    if x0 is not None:
        z[0] = x0
    if y0 is not None:
        z[1] = y0

    odl.solvers.pdhg(z, f, g, op, niter=niter, tau=tau, sigma=sigma, callback=callback)
    return z[0], z[1]
        
    
# Make the SIAC penalty in L1 for testing difference
def solve_siac_L1(A, data, alpha, moments=2, BSorder=2, niter=300,
                  x0=None, callback=None, tau=None, sigma=None, op_norm=None):
    X = A.domain

    # SIAC operator K and residual operator B = I - K
    K = make_siac_operator_odl(space=X, moments=moments, BSorder=BSorder)
    B = odl.IdentityOperator(X) - K

    # Fidelity on A x in A.range (no translated quadratic issues)
    l2 = ShiftedL2NormSquared(A.range, data)

    # L1 penalty on Bx, i.e. g2(z2) = alpha * ||z2||_1
    l1 = 0.5 * alpha * odl.solvers.L1Norm(X)

    # g(z1, z2) = l2(z1) + l1(z2), with z1 = A x, z2 = B x
    g = odl.solvers.SeparableSum(l2, l1)
    op = odl.BroadcastOperator(A, B)

    # no extra primal term
    f = odl.solvers.ZeroFunctional(X)

    # Step sizes
    if op_norm is None:
        op_norm = 1.1 * odl.power_method_opnorm(op)
    if tau is None:
        tau = 1.0 / op_norm
    if sigma is None:
        sigma = 1.0 / op_norm

    x = X.zero() if x0 is None else x0.copy()

    odl.solvers.pdhg(x, f, g, op, niter=niter, tau=tau, sigma=sigma, callback=callback)
    return x

def solve_siac_grad(A, data, alpha=1e-2, moments=2, BSorder=2, niter=200,
                  x0=None, isotropic=True, grad_method="forward",
                  pad_mode="symmetric", tau=None, sigma=None,
                  op_norm=None, callback=None):
    r"""
    Solve a SIAC-filtered TV-type least-squares problem using PDHG:

        min_x  0.5 ||A x - data||_2^2
            + 0.5 alpha ||(I - K) ∇x||_1,

    where K is a SIAC smoothing operator applied componentwise to the
    gradient field ∇x.

    Thus, instead of penalizing the gradient directly as in standard TV,
    this model penalizes the non-SIAC-reproduced part of the gradient.
    In other words, oscillatory / non-polynomial components of the gradient
    are penalized, while gradient fields locally reproduced by the SIAC
    kernel are less penalized.

    Two variants are supported:

    • isotropic SIAC-TV:
        sum_i sqrt( ((I-K)D_x x)_i^2 + ((I-K)D_y x)_i^2 )

    • anisotropic SIAC-TV:
        sum_i ( |((I-K)D_x x)_i| + |((I-K)D_y x)_i| )

    Parameters
    ----------
    A : odl.Operator
        Forward operator.
    data : element of A.range
        Measured data.
    alpha : float
        Regularization parameter.
    moments : int
        Number of enforced SIAC moments.
    BSorder : int
        B-spline order used in the SIAC kernel.
    niter : int
        Number of PDHG iterations.
    x0 : element of A.domain, optional
        Initial guess.
    isotropic : bool
        If True, use isotropic SIAC-TV (default).
        If False, use anisotropic SIAC-TV.
    grad_method : str
        Discretization method for the gradient.
    pad_mode : str
        Boundary condition for derivatives.
    tau, sigma : float, optional
        PDHG step sizes. If None, estimated automatically.
    op_norm : float, optional
        Precomputed operator norm.
    callback : callable, optional
        ODL callback.

    Returns
    -------
    x : element of A.domain
        Reconstructed solution.
    """
    X = A.domain
    G = odl.Gradient(X, method=grad_method, pad_mode=pad_mode)
    V = G.range

    # Build scalar SIAC operator on the image space, then apply it
    # componentwise to the gradient field.
    K_scalar = make_siac_operator_odl(X, moments=moments, BSorder=BSorder)
    K_vec = odl.ProductSpaceOperator([[K_scalar, 0],
                                      [0, K_scalar]])
    B = 0.5*odl.IdentityOperator(V) - K_vec

    # We form Op(x) = [A x, (I - K) ∇x]
    op = odl.BroadcastOperator(A, B * G)

    # l2 = 0.5 * odl.solvers.L2NormSquared(A.range).translated(data)

    # NOTE: Avoid ODL's `.translated(data)` quadratic prox (can type-error in 0.8.3).
    # This is the same data fidelity: 0.5*||Ax - data||^2, but with a safe
    # closed-form prox of the conjugate.
    l2 = ShiftedL2NormSquared(A.range, data)

    f = odl.solvers.ZeroFunctional(X)

    # SIAC-filtered TV functional (isotropic or anisotropic)
    if isotropic:
        tv = 0.5 * alpha * odl.solvers.GroupL1Norm(V)
    else:
        tv = 0.5 * alpha * odl.solvers.L1Norm(V)
    g = odl.solvers.SeparableSum(l2, tv)

    # Step sizes
    if op_norm is None:
        # Estimated operator norm, add 10 percent
        # to ensure ||Op||_2^2 * sigma * tau < 1
        op_norm = 1.1 * odl.power_method_opnorm(op)
    if tau is None:
        tau = 1.0 / op_norm
    if sigma is None:
        sigma = 1.0 / op_norm

    x = op.domain.zero() if x0 is None else x0.copy()

    odl.solvers.pdhg(x, f, g, op, niter=niter,
                     tau=tau, sigma=sigma, callback=callback)

    return x
    
# We build a function searching for good alpha values using
# the Morozovs discrepency principle since the data is synthetic
# and the noise level is consequently known
    
def choose_alpha_morozov(reconstruct, A, data, delta, 
                         alpha_lo = 1e-8, alpha_hi = 1e2, 
                         max_expand = 30, max_iter = 40, 
                         rtol = 1e-2, atol = 0.0,
                         logspace = True, 
                         return_full = False):
    
    if delta <= 0:
        raise ValueError("delta must be positive")
    
    # Helpers 
    def get_x(alpha):   # for functions returning two arguments (e.g., TGV, assume first is reconstruction)
        out = reconstruct(alpha)
        return out[0] if isinstance(out, (tuple, list)) else out
    
    def residual(alpha):
        x = get_x(alpha)
        r = A(x) - data
        # ODL elements have .norm(); numpy arrays use np.linalg.norm
        res = float(r.norm() if hasattr(r, "norm") else np.linalg.norm(r))
        return x, res
    
    x_lo, r_lo = residual(alpha_lo)
    x_hi, r_hi = residual(alpha_hi)
    hist = [(alpha_lo, r_lo), (alpha_hi, r_hi)]
    
    # shift the residual landscape s.t. r_lo <= delta <= r_hi
    steps = 0
    while not (r_lo <= delta <= r_hi) and steps < max_expand:
        steps += 1
        if r_lo > delta:
            # reset ceiling
            alpha_hi, x_hi, r_hi = alpha_lo, x_lo, r_lo
            # lower current lowest alpha by 1 magnitude (for log-space)
            alpha_lo /= 10
            # compute new low
            x_lo, r_lo = residual(alpha_lo)
            hist.append((alpha_lo, r_lo))
        else:   # r_hi < delta
            # reset floor
            alpha_lo, x_lo, r_lo = alpha_hi, x_hi, r_hi
            # increase current highest alpha by 1 magnitude (for log-space)
            alpha_hi *= 10
            # compute new high
            x_hi, r_hi = residual(alpha_hi)
            hist.append((alpha_hi, r_hi))
        
    if not (r_lo <= delta <= r_hi):
        raise RuntimeError(
            f"Could not find $\delta$={delta:.6g}"
            f"Got r_lo={r_lo:.6g} at $\alpha$={alpha_lo:.3g}, r_hi={r_hi:.6g} at $\alpha$={alpha_hi:.3g}."
        )
    
    # Bisect to find optimal
    alpha_star, x_star, r_star = None, None, None
    
    for _ in range(max_iter):
        # in log-space: (log(a)+log(b))/2 = log(sqrt(ab))
        alpha_mid = float(np.sqrt(alpha_lo * alpha_hi)) if logspace else 0.5 * (alpha_hi + alpha_lo)
        x_mid, r_mid = residual(alpha_mid)
        hist.append((alpha_mid, r_mid))
        
        alpha_star, x_star, r_star = alpha_mid, x_mid, r_mid
        
        if abs(r_mid - delta) <= atol + rtol * delta:
            break
        
        if r_mid < delta:
            alpha_lo, x_lo, r_lo = alpha_mid, x_mid, r_mid
        else:
            alpha_hi, x_hi, r_hi = alpha_mid, x_mid, r_mid
        
    if return_full:
        info = {
            "history": hist,
            "delta": float(delta), 
            "alpha_star": float(alpha_star), 
            "res_star": float(r_star)
        }
        return alpha_star, x_star, info
    
    return alpha_star, x_star
        