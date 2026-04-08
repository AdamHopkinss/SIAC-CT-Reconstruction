import numpy as np

def local_cell_center_nodes_1d(nloc):
    """
    Return nloc equispaced cell-center nodes on [-1,1]:
        r_j = -1 + (2j+1)/nloc,  j=0,...,nloc-1
    """
    if not isinstance(nloc, int):
        raise TypeError("nloc must be an integer.")
    if nloc <= 0:
        raise ValueError("nloc must be positive.")

    j = np.arange(nloc, dtype=float)
    return -1.0 + (2.0 * j + 1.0) / nloc

def build_grid_from_local_nodes_1d(mesh, eval_nodes, return_blocks=False):
    """
    Map local reference nodes in [-1,1] to each physical element.
    
    Parameters
    ----------
    mesh : dict
    eval_nodes : array_like, shape (n_eval,)
        Local reference nodes in [-1,1].
    
    Returns
    -------
    grid : ndarray, shape (K * n_eval,)
        Flattened physical grid.
    blocks : ndarray, shape (K, n_eval)
        Physical grid grouped per element.
    """
    eval_nodes = np.asarray(eval_nodes, dtype=float)
    K = mesh["K"]
    edges = mesh["edges"]
    n_eval = len(eval_nodes)

    blocks = np.zeros((K, n_eval), dtype=float)

    for e in range(K):
        x0, x1 = edges[e], edges[e + 1]
        xc = 0.5 * (x0 + x1)
        hx = x1 - x0
        blocks[e, :] = xc + 0.5 * hx * eval_nodes

    grid = blocks.reshape(K * n_eval)
    if return_blocks:
        return grid, blocks
    return grid

def build_grid_from_local_nodes_2d(mesh, eval_nodes):
    """
    Build the global physical grid corresponding to local reference nodes
    repeated identically in each element.

    Parameters
    ----------
    mesh : dict
        DG mesh dictionary.
    eval_nodes : array_like
        Local reference nodes in [-1,1].

    Returns
    -------
    X, Y : ndarray
        Global physical grid of shape (Ky*n_eval, Kx*n_eval).
    """
    eval_nodes = np.asarray(eval_nodes, dtype=float)
    n_eval = len(eval_nodes)

    Kx = mesh["Kx"]
    Ky = mesh["Ky"]
    x_edges = mesh["x_edges"]
    y_edges = mesh["y_edges"]

    x_blocks = np.zeros((Kx, n_eval), dtype=float)
    y_blocks = np.zeros((Ky, n_eval), dtype=float)

    for ex in range(Kx):
        x0, x1 = x_edges[ex], x_edges[ex + 1]
        x_blocks[ex, :] = 0.5 * (x0 + x1) + 0.5 * (x1 - x0) * eval_nodes

    for ey in range(Ky):
        y0, y1 = y_edges[ey], y_edges[ey + 1]
        y_blocks[ey, :] = 0.5 * (y0 + y1) + 0.5 * (y1 - y0) * eval_nodes

    xgrid = x_blocks.reshape(Kx * n_eval)
    ygrid = y_blocks.reshape(Ky * n_eval)

    X, Y = np.meshgrid(xgrid, ygrid, indexing="xy")
    return X, Y


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