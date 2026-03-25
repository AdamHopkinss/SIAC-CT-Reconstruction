
import numpy as np
import warnings
from numpy.polynomial.legendre import leggauss
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
    Build a uniform DG mesh compatible with the image resolution.

    Parameters
    ----------
    DOF_x, DOF_y : int
        Number of image samples in x and y.
    xlim, ylim : tuple[float, float]
        Physical domain bounds.
    deg : int
        Requested minimum polynomial degree.

    Returns
    -------
    mesh : dict
        DG mesh metadata.
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

    mesh = {
        "domain": [xmin, xmax, ymin, ymax],
        "Kx": Kx,              # number of DG elements in x
        "Ky": Ky,              # number of DG elements in y
        "p": p,                # polynomial degree
        "order": p + 1,        # local nodal count per direction
        "x_edges": x_edges,
        "y_edges": y_edges,
        "hx": hx,              # element width in x
        "hy": hy,              # element width in y
        "DOF_x": DOF_x,        # image resolution in x
        "DOF_y": DOF_y,        # image resolution in y
    }
    return mesh

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


def eval_dg_modal(dg, X, Y, fill_value=0.0):
    """
    Evaluate a DG modal field at arbitrary points (X,Y).
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

        # Outside domain
        if (x < xmin) or (x > xmax) or (y < ymin) or (y > ymax):
            continue

        # Find element
        ex = np.searchsorted(x_edges, x, side="right") - 1
        ey = np.searchsorted(y_edges, y, side="right") - 1
        ex = int(np.clip(ex, 0, Kx - 1))
        ey = int(np.clip(ey, 0, Ky - 1))

        # Element bounds
        x0, x1 = x_edges[ex], x_edges[ex + 1]
        y0, y1 = y_edges[ey], y_edges[ey + 1]

        # Map to reference square [-1,1]^2
        r = 2.0 * (x - x0) / (x1 - x0) - 1.0
        s = 2.0 * (y - y0) / (y1 - y0) - 1.0
        r = np.clip(r, -1.0, 1.0)
        s = np.clip(s, -1.0, 1.0)

        # Basis values
        lr = eval_orthonormal_legendre_1d(np.array([r]), p)[:, 0]
        ls = eval_orthonormal_legendre_1d(np.array([s]), p)[:, 0]

        # Evaluate modal expansion
        Ce = coeffs[ey, ex]   # shape (p+1, p+1)
        # Ce[my, mx], ls[my], lr[mx]
        U[k] = np.einsum("ij,i,j->", Ce, ls, lr)

    return U.reshape(Xb.shape)

def eval_dg_modal_on_img_grid(dg, fill_value=0.0):
    """
    Evaluate DG field on the original image grid stored in dg.
    """
    X, Y = np.meshgrid(dg["xgrid"], dg["ygrid"], indexing="xy")
    return eval_dg_modal(dg, X, Y, fill_value)