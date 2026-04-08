import numpy as np

def build_uniform_mesh_1d(K, p, domain=(-1, 1)):
    xmin, xmax = domain
    h = (xmax - xmin) / K
    edges = np.linspace(xmin, xmax, K + 1)
    return {
        "domain": (xmin, xmax),
        "K": K,
        "h": h,
        "edges": edges,
        "p": p
    }
    
def build_uniform_mesh_2d(Kx, Ky, p, xlim=(-1, 1), ylim=(-1, 1)):
    """
    Build a uniform tensor-product DG mesh for exact-function L2 projection.
    """
    xmin, xmax = xlim
    ymin, ymax = ylim

    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Require xlim[1] > xlim[0] and ylim[1] > ylim[0].")

    if Kx <= 0 or Ky <= 0:
        raise ValueError("Require Kx > 0 and Ky > 0.")

    hx = (xmax - xmin) / Kx
    hy = (ymax - ymin) / Ky

    x_edges = np.linspace(xmin, xmax, Kx + 1)
    y_edges = np.linspace(ymin, ymax, Ky + 1)

    return {
        "domain": [xmin, xmax, ymin, ymax],
        "Kx": Kx,
        "Ky": Ky,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "hx": hx,
        "hy": hy,
        "p": p,
        "order": p + 1,
    }
    
    
from src.utils import resolve_degree
def build_img_dg_mesh(DOF_x, DOF_y, xlim=(-1, 1), ylim=(-1, 1), deg=3, verbose=True):
    """
    Build a uniform tensor-product DG mesh compatible with the image resolution.
    """
    xmin, xmax = xlim
    ymin, ymax = ylim

    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Require xlim[1] > xlim[0] and ylim[1] > ylim[0].")

    p, Ky, Kx = resolve_degree(DOF_x=DOF_x, DOF_y=DOF_y, deg=deg)

    if verbose:
        print(f"degree used: p = {p}")
    
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