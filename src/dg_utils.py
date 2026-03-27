import numpy as np
import warnings
from scipy.special import eval_legendre


def _to_numpy(recon):
    """Accept ODL element or numpy array."""
    if hasattr(recon, "asarray"):
        return recon.asarray()
    return np.asarray(recon)


def resolve_degree(DOF_x: int, DOF_y: int, deg: int) -> tuple[int, int, int]:
    """
    Resolve an admissible polynomial degree p and element counts Kx, Ky such that

        DOF_x = (p + 1) * Kx
        DOF_y = (p + 1) * Ky
    """
    if not isinstance(DOF_x, int) or not isinstance(DOF_y, int) or not isinstance(deg, int):
        raise TypeError("DOF_x, DOF_y, and deg must all be integers.")
    if DOF_x <= 0 or DOF_y <= 0:
        raise ValueError("DOF_x and DOF_y must be positive.")
    if deg < 0:
        raise ValueError("deg must be nonnegative.")

    requested_deg = deg
    max_p = min(DOF_x, DOF_y) - 1

    for p in range(deg, max_p + 1):
        order = p + 1
        if DOF_x % order == 0 and DOF_y % order == 0:
            Kx = DOF_x // order
            Ky = DOF_y // order

            if p != requested_deg:
                warnings.warn(
                    f"Requested degree deg={requested_deg} is not admissible for "
                    f"(DOF_x, DOF_y)=({DOF_x}, {DOF_y}). Using p={p} instead, "
                    f"giving (Kx, Ky)=({Kx}, {Ky}).",
                    stacklevel=2
                )
            return p, Kx, Ky

    raise ValueError(
        f"No admissible polynomial degree p >= {deg} found such that "
        f"DOF_x = (p+1)Kx and DOF_y = (p+1)Ky for "
        f"(DOF_x, DOF_y)=({DOF_x}, {DOF_y})."
    )


def build_image_grid(DOF_x, DOF_y, xlim=(-1, 1), ylim=(-1, 1)):
    """
    Build pixel-center coordinates for an image of shape (DOF_y, DOF_x).

    Returns
    -------
    xgrid : ndarray, shape (DOF_x,)
        Pixel-center x-coordinates.
    ygrid : ndarray, shape (DOF_y,)
        Pixel-center y-coordinates.
    dx, dy : float
        Pixel spacings.
    """
    xmin, xmax = xlim
    ymin, ymax = ylim

    dx = (xmax - xmin) / DOF_x
    dy = (ymax - ymin) / DOF_y

    xgrid = xmin + (np.arange(DOF_x) + 0.5) * dx
    ygrid = ymin + (np.arange(DOF_y) + 0.5) * dy

    return xgrid, ygrid, dx, dy


def build_dg_mesh(DOF_x, DOF_y, xlim=(-1, 1), ylim=(-1, 1), deg=3):
    """
    Build a uniform tensor-product DG mesh compatible with the image resolution.
    """
    xmin, xmax = xlim
    ymin, ymax = ylim

    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Require xlim[1] > xlim[0] and ylim[1] > ylim[0].")

    p, Kx, Ky = resolve_degree(DOF_x=DOF_x, DOF_y=DOF_y, deg=deg)

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


def eval_orthonormal_legendre_1d(x, p):
    """
    Evaluate orthonormal Legendre basis functions l_0,...,l_p at points x.

    l_n(x) = sqrt((2n+1)/2) P_n(x)

    Parameters
    ----------
    x : array_like
        Evaluation points in [-1,1].
    p : int
        Polynomial degree.

    Returns
    -------
    out : ndarray, shape (p+1, len(x))
        out[m, i] = l_m(x_i)
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros((p + 1, x.size), dtype=float)

    for n in range(p + 1):
        out[n, :] = np.sqrt((2*n + 1) / 2.0) * eval_legendre(n, x)

    return out

def eval_dg_modal(dg, X, Y, fill_value=0.0):
    """
    Evaluate a modal tensor-product DG field at arbitrary physical points (X,Y).

    Storage convention
    ------------------
    coeffs[ey, ex, my, mx]

    Expansion on one element
    ------------------------
    u(r,s) = sum_{my=0}^p sum_{mx=0}^p
             coeffs[ey,ex,my,mx] * phi_my(s) * phi_mx(r)

    where
        r = reference x-coordinate in [-1,1]
        s = reference y-coordinate in [-1,1]

    Parameters
    ----------
    dg : dict
        DG dictionary returned by nodal_image_to_dg or equivalent.
    X, Y : array_like
        Physical evaluation coordinates. They may be scalars, vectors, or
        broadcastable arrays.
    fill_value : float
        Value used outside the domain.

    Returns
    -------
    U : ndarray
        Evaluated field with broadcasted shape of X and Y.
    """
    mesh = dg["mesh"]
    coeffs = dg["coeffs"]

    xmin, xmax, ymin, ymax = mesh["domain"]
    Kx, Ky, p = mesh["Kx"], mesh["Ky"], mesh["p"]
    x_edges, y_edges = mesh["x_edges"], mesh["y_edges"]

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Xb, Yb = np.broadcast_arrays(X, Y)

    Xf = Xb.ravel()
    Yf = Yb.ravel()
    U = np.full(Xf.shape, fill_value, dtype=float)

    for k in range(Xf.size):
        x = Xf[k]
        y = Yf[k]

        if (x < xmin) or (x > xmax) or (y < ymin) or (y > ymax):
            continue

        ex = np.searchsorted(x_edges, x, side="right") - 1
        ey = np.searchsorted(y_edges, y, side="right") - 1
        ex = int(np.clip(ex, 0, Kx - 1))
        ey = int(np.clip(ey, 0, Ky - 1))

        x0, x1 = x_edges[ex], x_edges[ex + 1]
        y0, y1 = y_edges[ey], y_edges[ey + 1]

        r = 2.0 * (x - x0) / (x1 - x0) - 1.0
        s = 2.0 * (y - y0) / (y1 - y0) - 1.0

        r = np.clip(r, -1.0, 1.0)
        s = np.clip(s, -1.0, 1.0)

        phi_x = eval_orthonormal_legendre_1d(np.array([r]), p)[:, 0]   # (mx,)
        phi_y = eval_orthonormal_legendre_1d(np.array([s]), p)[:, 0]   # (my,)

        A_block_yx = coeffs[ey, ex]    # shape (my, mx)

        # Sum_{my,mx} A[my,mx] phi_y[my] phi_x[mx]
        U[k] = np.einsum("ab,a,b->", A_block_yx, phi_y, phi_x)

    return U.reshape(Xb.shape)

def eval_dg_modal_on_img_grid(dg, fill_value=0.0):
    """
    Evaluate the DG modal field on the original image pixel-center grid
    stored in the dg dictionary.

    Returns
    -------
    U : ndarray, shape (DOF_y, DOF_x)
        DG field evaluated at the original image grid.
    """
    xgrid = dg["xgrid"]
    ygrid = dg["ygrid"]

    X, Y = np.meshgrid(xgrid, ygrid, indexing="xy")
    return eval_dg_modal(dg, X, Y, fill_value=fill_value)

