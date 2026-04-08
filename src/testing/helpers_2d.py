import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from src.mesh import build_uniform_mesh_2d
from src.grid import local_cell_center_nodes_1d, build_grid_from_local_nodes_2d
from src.siac_modal import apply_siac_modal_dg_2d, trim_valid_siac_region_2d
from src.projection_approach import l2_project_exact_func_to_dg_2d
from src.evaluation import eval_dg_on_local_nodes_2d
from src.transforms import nodal_to_modal_2d

def random_modal_coeffs_2d(Kx, Ky, p, seed=0, scale=1.0):
    """
    Random modal coefficients for (Kx, Ky) elements in 2D.
    Shape: (Kx, Ky, p+1, p+1)
    """
    rng = np.random.default_rng(seed)
    return scale * rng.standard_normal((Kx, Ky,p + 1,p + 1))


def run_l2_vs_nodal_modal_experiment_2d(
    exact_func,
    *,
    Kx,
    Ky,
    xlim=(-1.0, 1.0),
    ylim=(-1.0, 1.0),
    p=None,
    poly_deg=None,
    moments=None,
    BSorder=None,
    n_eval=None,
    n_eval_fine=None,
    quad_order=None,
    add_noise=False,
):
    """
    Compare two DG construction pipelines in 2D:
        exact L2 projection -> DG -> SIAC
        nodal sampling -> modal DG -> SIAC

    Parameters
    ----------
    exact_func : callable
        Exact function exact_func(x, y).
    Kx, Ky : int
        Number of elements in x and y.
    xlim, ylim : tuple
        Domain limits.
    p : int or None
        DG polynomial degree. If None, set to ceil((poly_deg - 1)/2).
    poly_deg : int or None
        Polynomial degree of the exact test polynomial (if applicable)
    moments : int or None
        SIAC moments. If None, set to 2*p.
    BSorder : int or None
        SIAC B-spline order. If None, set to p+1.
    n_eval : int or None
        Local nodes per element for nodal->modal construction.
        If None, set to p+1.
    n_eval_fine : int or None
        Fine local nodes per element for evaluation.
        If None, set to 10*(p+1).
    quad_order : int or None
        Quadrature order passed to exact L2 projection.
        If None, set to ceil((poly_deg + p + 1)/2).
    add_noise : bool
        Passed into L2 projection helper.

    Returns
    -------
    results : dict
        Dictionary containing mesh, fields, trimmed fields, errors, and summaries.
    """
    if p is None:
        p = int(np.ceil((poly_deg - 1) / 2))

    order = p + 1

    if moments is None:
        moments = 2 * p
    if BSorder is None:
        BSorder = p + 1
    if n_eval is None:
        n_eval = order
    if n_eval_fine is None:
        n_eval_fine = 10 * order
    if quad_order is None:
        if poly_deg is None:
            quad_order = max(2 * p + 4, 8)
        else:
            quad_order = int(np.ceil( (poly_deg + p + 1) / 2 ))

    # --------------------------------------------------
    # 1. Build mesh
    # --------------------------------------------------
    mesh = build_uniform_mesh_2d(Kx=Kx, Ky=Ky, p=p, xlim=xlim, ylim=ylim)

    # --------------------------------------------------
    # 2. Exact L2 projection onto modal DG
    # --------------------------------------------------
    dg_l2 = l2_project_exact_func_to_dg_2d(
        func=exact_func,
        mesh=mesh,
        poly_max_deg=poly_deg,
        quad_order=quad_order,
        add_noise=add_noise,
    )

    # --------------------------------------------------
    # 3. Nodal -> modal DG construction
    # --------------------------------------------------
    nodes = local_cell_center_nodes_1d(n_eval)
    X_nodes, Y_nodes = build_grid_from_local_nodes_2d(mesh, nodes)

    U_nodes = exact_func(X_nodes, Y_nodes)
    Unode = U_nodes.reshape(Ky, order, Kx, order).transpose(0, 2, 1, 3)

    dg_nm = nodal_to_modal_2d(Unode, mesh, p=p)

    # --------------------------------------------------
    # 4. Fine-grid evaluation
    # --------------------------------------------------
    nodes_fine = local_cell_center_nodes_1d(n_eval_fine)
    X_fine, Y_fine = build_grid_from_local_nodes_2d(mesh, nodes_fine)

    U_exact = exact_func(X_fine, Y_fine)

    U_dg_l2 = eval_dg_on_local_nodes_2d(dg_l2, eval_nodes=nodes_fine)
    U_dg_nm = eval_dg_on_local_nodes_2d(dg_nm, eval_nodes=nodes_fine)

    U_siac_l2 = apply_siac_modal_dg_2d(
        dg_l2, moments=moments, BSorder=BSorder, eval_nodes=nodes_fine
    )
    U_siac_nm = apply_siac_modal_dg_2d(
        dg_nm, moments=moments, BSorder=BSorder, eval_nodes=nodes_fine
    )

    # --------------------------------------------------
    # 5. Trim valid SIAC interior region
    # --------------------------------------------------
    exact_trim, trim = trim_valid_siac_region_2d(
        U_exact,
        n_eval=n_eval_fine,
        moments=moments,
        BSorder=BSorder,
        return_trim=True,
    )

    dg_l2_trim = trim_valid_siac_region_2d(
        U_dg_l2, n_eval=n_eval_fine, moments=moments, BSorder=BSorder
    )
    siac_l2_trim = trim_valid_siac_region_2d(
        U_siac_l2, n_eval=n_eval_fine, moments=moments, BSorder=BSorder
    )

    dg_nm_trim = trim_valid_siac_region_2d(
        U_dg_nm, n_eval=n_eval_fine, moments=moments, BSorder=BSorder
    )
    siac_nm_trim = trim_valid_siac_region_2d(
        U_siac_nm, n_eval=n_eval_fine, moments=moments, BSorder=BSorder
    )

    # --------------------------------------------------
    # 6. Errors on trimmed region
    # --------------------------------------------------
    err_dg_l2 = dg_l2_trim - exact_trim
    err_siac_l2 = siac_l2_trim - exact_trim

    err_dg_nm = dg_nm_trim - exact_trim
    err_siac_nm = siac_nm_trim - exact_trim

    exact_norm = np.linalg.norm(exact_trim)

    error_summary = {
        "dg_l2_max": np.max(np.abs(err_dg_l2)),
        "dg_l2_rel_l2": np.linalg.norm(err_dg_l2) / exact_norm,
        "siac_l2_max": np.max(np.abs(err_siac_l2)),
        "siac_l2_rel_l2": np.linalg.norm(err_siac_l2) / exact_norm,
        "dg_nm_max": np.max(np.abs(err_dg_nm)),
        "dg_nm_rel_l2": np.linalg.norm(err_dg_nm) / exact_norm,
        "siac_nm_max": np.max(np.abs(err_siac_nm)),
        "siac_nm_rel_l2": np.linalg.norm(err_siac_nm) / exact_norm,
    }

    return {
        "params": {
            "poly_deg": poly_deg,
            "p": p,
            "order": order,
            "Kx": Kx,
            "Ky": Ky,
            "moments": moments,
            "BSorder": BSorder,
            "xlim": xlim,
            "ylim": ylim,
            "n_eval": n_eval,
            "n_eval_fine": n_eval_fine,
            "quad_order": quad_order,
            "trim": trim,
        },
        "mesh": mesh,
        "nodes": nodes,
        "nodes_fine": nodes_fine,
        "grids": {
            "X_nodes": X_nodes,
            "Y_nodes": Y_nodes,
            "X_fine": X_fine,
            "Y_fine": Y_fine,
        },
        "dg": {
            "l2": dg_l2,
            "nm": dg_nm,
        },
        "fields": {
            "exact": U_exact,
            "dg_l2": U_dg_l2,
            "dg_nm": U_dg_nm,
            "siac_l2": U_siac_l2,
            "siac_nm": U_siac_nm,
        },
        "trimmed_fields": {
            "exact": exact_trim,
            "dg_l2": dg_l2_trim,
            "dg_nm": dg_nm_trim,
            "siac_l2": siac_l2_trim,
            "siac_nm": siac_nm_trim,
        },
        "errors": {
            "dg_l2": err_dg_l2,
            "siac_l2": err_siac_l2,
            "dg_nm": err_dg_nm,
            "siac_nm": err_siac_nm,
        },
        "error_summary": error_summary,
    }


def _positive_finite_values(arr):
    arr = np.asarray(arr)
    mask = np.isfinite(arr) & (arr > 0)
    return arr[mask]


def _plot_max(arr):
    vals = _positive_finite_values(arr)
    if vals.size == 0:
        return 0.0
    return np.max(vals)


def _plot_min_positive(arr):
    vals = _positive_finite_values(arr)
    if vals.size == 0:
        return None
    return np.min(vals)


def _within_factor(values, factor):
    values = [v for v in values if v is not None and v > 0]
    if len(values) <= 1:
        return True
    return max(values) / min(values) <= factor


def _make_log_norm(arrays):
    mins = []
    maxs = []

    for arr in arrays:
        amin = _plot_min_positive(arr)
        amax = _plot_max(arr)
        if amin is not None and amax > 0:
            mins.append(amin)
            maxs.append(amax)

    if not mins or not maxs:
        return None

    vmin = min(mins)
    vmax = max(maxs)

    if vmin <= 0 or vmax <= 0:
        return None

    return LogNorm(vmin=vmin, vmax=vmax)

def robust_percentile(arr, q=99):
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return 0.0
    return np.percentile(vals, q)


def choose_error_norms(
    dg_l2_err_plot,
    siac_l2_err_plot,
    dg_nm_err_plot,
    siac_nm_err_plot,
    factor=10**1.5,
):
    """
    Decide shared/pairwise/separate log scales for the four error plots.
    Returns:
        norms : dict
        mode  : str
    """
    m_dg_l2 = _plot_max(dg_l2_err_plot)
    m_siac_l2 = _plot_max(siac_l2_err_plot)
    m_dg_nm = _plot_max(dg_nm_err_plot)
    m_siac_nm = _plot_max(siac_nm_err_plot)

    if _within_factor([m_dg_l2, m_siac_l2, m_dg_nm, m_siac_nm], factor):
        shared = _make_log_norm([
            dg_l2_err_plot, siac_l2_err_plot, dg_nm_err_plot, siac_nm_err_plot
        ])
        return {
            "dg_l2": shared,
            "siac_l2": shared,
            "dg_nm": shared,
            "siac_nm": shared,
        }, "all_shared"

    l2_ok = _within_factor([m_dg_l2, m_siac_l2], factor)
    nm_ok = _within_factor([m_dg_nm, m_siac_nm], factor)

    if l2_ok and nm_ok:
        l2_norm = _make_log_norm([dg_l2_err_plot, siac_l2_err_plot])
        nm_norm = _make_log_norm([dg_nm_err_plot, siac_nm_err_plot])
        return {
            "dg_l2": l2_norm,
            "siac_l2": l2_norm,
            "dg_nm": nm_norm,
            "siac_nm": nm_norm,
        }, "pairwise_shared"

    return {
        "dg_l2": _make_log_norm([dg_l2_err_plot]),
        "siac_l2": _make_log_norm([siac_l2_err_plot]),
        "dg_nm": _make_log_norm([dg_nm_err_plot]),
        "siac_nm": _make_log_norm([siac_nm_err_plot]),
    }, "separate"

def plot_l2_vs_nodal_modal_experiment_2d(results, log_errors=False):
    """
    Plot exact, DG, SIAC, and interior error fields for the
    L2-projection vs nodal->modal comparison experiment.
    """
    xlim = results["params"]["xlim"]
    ylim = results["params"]["ylim"]
    trim = results["params"]["trim"]

    U_exact = results["fields"]["exact"]
    U_dg_l2 = results["fields"]["dg_l2"]
    U_siac_l2 = results["fields"]["siac_l2"]
    U_dg_nm = results["fields"]["dg_nm"]
    U_siac_nm = results["fields"]["siac_nm"]

    err_dg_l2 = results["errors"]["dg_l2"]
    err_siac_l2 = results["errors"]["siac_l2"]
    err_dg_nm = results["errors"]["dg_nm"]
    err_siac_nm = results["errors"]["siac_nm"]

    dg_l2_err_plot = np.full_like(U_exact, np.nan, dtype=float)
    siac_l2_err_plot = np.full_like(U_exact, np.nan, dtype=float)
    dg_nm_err_plot = np.full_like(U_exact, np.nan, dtype=float)
    siac_nm_err_plot = np.full_like(U_exact, np.nan, dtype=float)

    if trim == 0:
        dg_l2_err_plot[:] = np.abs(err_dg_l2)
        siac_l2_err_plot[:] = np.abs(err_siac_l2)
        dg_nm_err_plot[:] = np.abs(err_dg_nm)
        siac_nm_err_plot[:] = np.abs(err_siac_nm)
    else:
        sl = slice(trim, -trim)
        dg_l2_err_plot[sl, sl] = np.abs(err_dg_l2)
        siac_l2_err_plot[sl, sl] = np.abs(err_siac_l2)
        dg_nm_err_plot[sl, sl] = np.abs(err_dg_nm)
        siac_nm_err_plot[sl, sl] = np.abs(err_siac_nm)

    # --------------------------------------------------
    # L2 pipeline fields
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    im0 = axes[0].imshow(
        U_exact,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        aspect="equal",
    )
    axes[0].set_title("Exact")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        U_dg_l2,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        aspect="equal",
    )
    axes[1].set_title("DG (L2 projection)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        U_siac_l2,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        aspect="equal",
    )
    axes[2].set_title("SIAC (L2 projection)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2])

    plt.show()

    # --------------------------------------------------
    # Nodal->modal pipeline fields
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    im0 = axes[0].imshow(
        U_exact,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        aspect="equal",
    )
    axes[0].set_title("Exact")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        U_dg_nm,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        aspect="equal",
    )
    axes[1].set_title("DG (Nodal→Modal)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        U_siac_nm,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        aspect="equal",
    )
    axes[2].set_title("SIAC (Nodal→Modal)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2])

    plt.show()

    # --------------------------------------------------
    # Error plots only
    # --------------------------------------------------
    if log_errors:
        norms, scale_mode = choose_error_norms(
            dg_l2_err_plot,
            siac_l2_err_plot,
            dg_nm_err_plot,
            siac_nm_err_plot,
            factor=10**1.5,
        )
        print("Error color scale mode:", scale_mode)
    else:
        norms = {
            "dg_l2": None,
            "siac_l2": None,
            "dg_nm": None,
            "siac_nm": None,
        }
    if not log_errors:
        # same decision logic but for linear scaling
        m_dg_l2 = robust_percentile(dg_l2_err_plot, q=99)
        m_siac_l2 = robust_percentile(siac_l2_err_plot, q=99)
        m_dg_nm = robust_percentile(dg_nm_err_plot, q=99)
        m_siac_nm = robust_percentile(siac_nm_err_plot, q=99)

        def within(vals, factor=10**1.5):
            vals = [v for v in vals if v > 0]
            if len(vals) <= 1:
                return True
            return max(vals) / min(vals) <= factor

        if within([m_dg_l2, m_siac_l2, m_dg_nm, m_siac_nm]):
            vmax_all = max(m_dg_l2, m_siac_l2, m_dg_nm, m_siac_nm)
            vlims = {
                "dg_l2": (0, vmax_all),
                "siac_l2": (0, vmax_all),
                "dg_nm": (0, vmax_all),
                "siac_nm": (0, vmax_all),
            }
        elif within([m_dg_l2, m_siac_l2]) and within([m_dg_nm, m_siac_nm]):
            vmax_l2 = max(m_dg_l2, m_siac_l2)
            vmax_nm = max(m_dg_nm, m_siac_nm)
            vlims = {
                "dg_l2": (0, vmax_l2),
                "siac_l2": (0, vmax_l2),
                "dg_nm": (0, vmax_nm),
                "siac_nm": (0, vmax_nm),
            }
        else:
            vlims = {
                "dg_l2": (0, m_dg_l2),
                "siac_l2": (0, m_siac_l2),
                "dg_nm": (0, m_dg_nm),
                "siac_nm": (0, m_siac_nm),
            }
    if log_errors:
        print("Error color scale mode:", scale_mode)

    # -----------------------------
    # L2 pipeline errors
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    im3 = axes[0].imshow(
        dg_l2_err_plot,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        aspect="equal",
        norm=norms["dg_l2"] if log_errors else None,
        vmin=vlims["dg_l2"][0] if not log_errors else None,
        vmax=vlims["dg_l2"][1] if not log_errors else None,
    )
    axes[0].set_title("|DG error| (L2, interior)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im3, ax=axes[0])

    im4 = axes[1].imshow(
        siac_l2_err_plot,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        aspect="equal",
        norm=norms["siac_l2"] if log_errors else None,
        vmin=vlims["siac_l2"][0] if not log_errors else None,
        vmax=vlims["siac_l2"][1] if not log_errors else None,
    )
    axes[1].set_title("|SIAC error| (L2, interior)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im4, ax=axes[1])

    plt.show()

    # -----------------------------
    # Nodal->modal pipeline errors
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    im3 = axes[0].imshow(
        dg_nm_err_plot,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        aspect="equal",
        norm=norms["dg_nm"] if log_errors else None,
        vmin=vlims["dg_nm"][0] if not log_errors else None,
        vmax=vlims["dg_nm"][1] if not log_errors else None,
    )
    axes[0].set_title("|DG error| (Nodal→Modal, interior)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im3, ax=axes[0])

    im4 = axes[1].imshow(
        siac_nm_err_plot,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        aspect="equal",
        norm=norms["siac_nm"] if log_errors else None,
        vmin=vlims["siac_nm"][0] if not log_errors else None,
        vmax=vlims["siac_nm"][1] if not log_errors else None,
    )
    axes[1].set_title("|SIAC error| (Nodal→Modal, interior)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im4, ax=axes[1])

    plt.show()
    
    
def add_rel_gauss_noise(Unode, rel_level=0.05, seed=62, return_noise=False):
    """
    Add additive Gaussian noise to a nodal DG array, with noise level
    defined relative to the RMS magnitude of the signal.

    Parameters
    ----------
    Unode : ndarray
        Array of nodal values, e.g. shape (Ky, Kx, order, order).
    rel_level : float
        Relative noise level. For example:
            0.01 = 1% noise
            0.05 = 5% noise
            0.10 = 10% noise
    seed : int or None
        Random seed for reproducibility.
    return_noise : bool
        If True, also return the generated noise array and sigma.

    Returns
    -------
    Unode_noisy : ndarray
        Noisy nodal data.
    noise : ndarray, optional
        The additive noise field.
    sigma : float, optional
        Standard deviation used for the Gaussian noise.
    """
    Unode = np.asarray(Unode, dtype=float)
    rng = np.random.default_rng(seed)

    signal_rms = np.sqrt(np.mean(Unode**2))
    sigma = rel_level * signal_rms

    noise = rng.standard_normal(size=Unode.shape) * sigma
    Unode_noisy = Unode + noise

    if return_noise:
        return Unode_noisy, noise, sigma
    return Unode_noisy

def compute_relative_l2_and_linf(U_num, U_exact):
    """
    Compute relative L2 and relative Linf errors on a sampled grid.
    """
    err = U_num - U_exact

    exact_l2 = np.linalg.norm(U_exact.ravel(), ord=2)
    exact_linf = np.max(np.abs(U_exact))

    rel_l2 = np.linalg.norm(err.ravel(), ord=2) / exact_l2

    if exact_linf > 0:
        rel_linf = np.max(np.abs(err)) / exact_linf
    else:
        rel_linf = np.max(np.abs(err))

    return {
        "abs_err": err,
        "rel_l2": rel_l2,
        "rel_linf": rel_linf,
    }
    
import numpy as np

def add_rel_gauss_noise(Unode, rel_level=0.05, seed=62, return_noise=False):
    """
    Add additive Gaussian noise to a nodal DG array, with noise level
    defined relative to the RMS magnitude of the signal.
    """
    Unode = np.asarray(Unode, dtype=float)
    rng = np.random.default_rng(seed)

    signal_rms = np.sqrt(np.mean(Unode**2))
    sigma = rel_level * signal_rms

    noise = rng.standard_normal(size=Unode.shape) * sigma
    Unode_noisy = Unode + noise

    if return_noise:
        return Unode_noisy, noise, sigma
    return Unode_noisy


def compute_relative_l2_and_linf(U_num, U_exact):
    """
    Compute relative L2 and relative Linf errors on a sampled grid.
    """
    U_num = np.asarray(U_num, dtype=float)
    U_exact = np.asarray(U_exact, dtype=float)

    err = U_num - U_exact

    exact_l2 = np.linalg.norm(U_exact.ravel(), ord=2)
    exact_linf = np.max(np.abs(U_exact))

    if exact_l2 > 0:
        rel_l2 = np.linalg.norm(err.ravel(), ord=2) / exact_l2
    else:
        rel_l2 = np.linalg.norm(err.ravel(), ord=2)

    if exact_linf > 0:
        rel_linf = np.max(np.abs(err)) / exact_linf
    else:
        rel_linf = np.max(np.abs(err))

    return {
        "abs_err": err,
        "rel_l2": rel_l2,
        "rel_linf": rel_linf,
    }


def run_noise_trial_2d(
    exact_func,
    Unode_clean,
    mesh,
    p,
    moments,
    BSorder,
    noise_level=0.0,
    seed=None,
    fine_factor=10,
    return_full=True,
):
    """
    Run one noisy-data DG/SIAC experiment in 2D.
    """
    Unode_clean = np.asarray(Unode_clean, dtype=float)
    Ky, Kx, order, _ = Unode_clean.shape

    # Fine evaluation grid
    n_eval_fine = fine_factor * order
    nodes_fine = local_cell_center_nodes_1d(n_eval_fine)

    Xf, Yf = build_grid_from_local_nodes_2d(
        mesh=mesh,
        eval_nodes=nodes_fine
    )

    # Exact field on fine grid
    U_exact = exact_func(Xf, Yf)

    # Add noise to nodal data
    Unode_noisy, noise, sigma = add_rel_gauss_noise(
        Unode_clean,
        rel_level=noise_level,
        seed=seed,
        return_noise=True,
    )

    # Nodal -> modal DG
    dg = nodal_to_modal_2d(Unode_noisy, mesh=mesh, p=p)

    # Evaluate DG and SIAC on fine grid
    U_dg = eval_dg_on_local_nodes_2d(
        dg,
        eval_nodes=nodes_fine
    )

    U_siac = apply_siac_modal_dg_2d(
        dg,
        moments=moments,
        BSorder=BSorder,
        eval_nodes=nodes_fine,
    )

    # Trim to common valid SIAC region
    U_exact_trim, trim = trim_valid_siac_region_2d(
        U_exact,
        n_eval=n_eval_fine,
        moments=moments,
        BSorder=BSorder,
        return_trim=True,
    )

    U_dg_trim = trim_valid_siac_region_2d(
        U_dg,
        n_eval=n_eval_fine,
        moments=moments,
        BSorder=BSorder,
    )

    U_siac_trim = trim_valid_siac_region_2d(
        U_siac,
        n_eval=n_eval_fine,
        moments=moments,
        BSorder=BSorder,
    )

    # Error metrics
    dg_metrics = compute_relative_l2_and_linf(U_dg_trim, U_exact_trim)
    siac_metrics = compute_relative_l2_and_linf(U_siac_trim, U_exact_trim)

    result = {
        "meta": {
            "noise_level": noise_level,
            "seed": seed,
            "sigma": sigma,
            "fine_factor": fine_factor,
            "n_eval_fine": n_eval_fine,
            "trim": trim,
        },
        "metrics": {
            "dg": {
                "rel_l2": dg_metrics["rel_l2"],
                "rel_linf": dg_metrics["rel_linf"],
                "abs_linf": np.max(np.abs(dg_metrics["abs_err"])),
            },
            "siac": {
                "rel_l2": siac_metrics["rel_l2"],
                "rel_linf": siac_metrics["rel_linf"],
                "abs_linf": np.max(np.abs(siac_metrics["abs_err"])),
            },
        },
    }

    if return_full:
        result["fields"] = {
            "Unode_clean": Unode_clean,
            "Unode_noisy": Unode_noisy,
            "noise": noise,
            "exact_full": U_exact,
            "dg_full": U_dg,
            "siac_full": U_siac,
            "exact_trim": U_exact_trim,
            "dg_trim": U_dg_trim,
            "siac_trim": U_siac_trim,
            "err_dg_trim": dg_metrics["abs_err"],
            "err_siac_trim": siac_metrics["abs_err"],
        }

    return result


def run_noise_sweep_2d(
    exact_func,
    Unode_clean,
    mesh,
    p,
    moments,
    BSorder,
    noise_levels,
    seed=0,
    fine_factor=10,
):
    """
    Run one trial for each noise level and collect summary metrics.
    """
    records = []

    for noise_level in noise_levels:
        res = run_noise_trial_2d(
            exact_func=exact_func,
            Unode_clean=Unode_clean,
            mesh=mesh,
            p=p,
            moments=moments,
            BSorder=BSorder,
            noise_level=noise_level,
            seed=seed,
            fine_factor=fine_factor,
            return_full=False,
        )

        records.append({
            "noise_level": noise_level,
            "sigma": res["meta"]["sigma"],
            "dg_rel_l2": res["metrics"]["dg"]["rel_l2"],
            "dg_rel_linf": res["metrics"]["dg"]["rel_linf"],
            "dg_abs_linf": res["metrics"]["dg"]["abs_linf"],
            "siac_rel_l2": res["metrics"]["siac"]["rel_l2"],
            "siac_rel_linf": res["metrics"]["siac"]["rel_linf"],
            "siac_abs_linf": res["metrics"]["siac"]["abs_linf"],
        })

    return records


def run_noise_monte_carlo_2d(
    exact_func,
    Unode_clean,
    mesh,
    p,
    moments,
    BSorder,
    noise_levels,
    n_trials=20,
    base_seed=0,
    fine_factor=10,
):
    """
    Run Monte Carlo study over multiple noise levels.

    Returns
    -------
    results : dict
        {
            "raw": DataFrame with all trials,
            "summary": DataFrame with mean/std per noise level
        }
    """
    records = []

    for i_level, noise_level in enumerate(noise_levels):
        for trial in range(n_trials):
            seed = base_seed + 10000 * i_level + trial

            res = run_noise_trial_2d(
                exact_func=exact_func,
                Unode_clean=Unode_clean,
                mesh=mesh,
                p=p,
                moments=moments,
                BSorder=BSorder,
                noise_level=noise_level,
                seed=seed,
                fine_factor=fine_factor,
                return_full=False,
            )

            records.append({
                "noise_level": noise_level,
                "trial": trial,
                "seed": seed,
                "dg_rel_l2": res["metrics"]["dg"]["rel_l2"],
                "dg_rel_linf": res["metrics"]["dg"]["rel_linf"],
                "dg_abs_linf": res["metrics"]["dg"]["abs_linf"],
                "siac_rel_l2": res["metrics"]["siac"]["rel_l2"],
                "siac_rel_linf": res["metrics"]["siac"]["rel_linf"],
                "siac_abs_linf": res["metrics"]["siac"]["abs_linf"],
            })

    df = pd.DataFrame(records)

    summary = (
        df.groupby("noise_level")
        .agg(
            dg_rel_l2_mean=("dg_rel_l2", "mean"),
            dg_rel_l2_std=("dg_rel_l2", "std"),
            dg_abs_linf_mean=("dg_abs_linf", "mean"),
            dg_abs_linf_std=("dg_abs_linf", "std"),
            siac_rel_l2_mean=("siac_rel_l2", "mean"),
            siac_rel_l2_std=("siac_rel_l2", "std"),
            siac_abs_linf_mean=("siac_abs_linf", "mean"),
            siac_abs_linf_std=("siac_abs_linf", "std"),
        )
        .reset_index()
    )

    return {
        "raw": df,
        "summary": summary,
    }


def plot_noise_trial_quad(results, noise_levels):
    """
    One row per noise level:
    [DG solution | SIAC solution | DG error | SIAC error]
    """
    nrows = len(results)

    fig, axes = plt.subplots(
        nrows, 4,
        figsize=(16, 4 * nrows),
        constrained_layout=True,
        squeeze=False
    )

    # Collect solution arrays for common solution color scaling
    dg_all = [res["fields"]["dg_trim"] for res in results]
    siac_all = [res["fields"]["siac_trim"] for res in results]
    sol_all = dg_all + siac_all

    sol_vmin = min(np.min(a) for a in sol_all)
    sol_vmax = max(np.max(a) for a in sol_all)

    # Collect error arrays for common symmetric error scaling
    dg_err_all = [res["fields"]["err_dg_trim"] for res in results]
    siac_err_all = [res["fields"]["err_siac_trim"] for res in results]
    err_all = dg_err_all + siac_err_all

    err_absmax = max(np.max(np.abs(a)) for a in err_all)

    im_sol = None
    im_err = None

    for i, (res, noise) in enumerate(zip(results, noise_levels)):
        dg = res["fields"]["dg_trim"]
        siac = res["fields"]["siac_trim"]
        err_dg = res["fields"]["err_dg_trim"]
        err_siac = res["fields"]["err_siac_trim"]
        
        eps = 1e-14

        log_err_dg = np.log10(np.abs(err_dg) + eps)
        log_err_siac = np.log10(np.abs(err_siac) + eps)
        
        log_err_all = [
            np.log10(np.abs(a) + eps)
            for a in err_all
        ]

        vmin = min(np.min(a) for a in log_err_all)
        vmax = max(np.max(a) for a in log_err_all)
        
        im_sol = axes[i, 0].imshow(
            dg,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=sol_vmin,
            vmax=sol_vmax,
        )

        axes[i, 1].imshow(
            siac,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=sol_vmin,
            vmax=sol_vmax,
        )

        im_err = axes[i, 2].imshow(
            log_err_dg,
            origin="lower",
            cmap="viridis",
            aspect="auto",
            vmin=vmin, 
            vmax=vmax
        )

        axes[i, 3].imshow(
            log_err_siac,
            origin="lower",
            cmap="viridis",
            aspect="auto",
            vmin=vmin, 
            vmax=vmax
        )

        # Row label on the left
        axes[i, 0].set_ylabel(f"noise = {noise:.2f}", fontsize=11)

        for j in range(4):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    # Column titles only on top row
    axes[0, 0].set_title("DG solution")
    axes[0, 1].set_title("SIAC solution")
    axes[0, 2].set_title("DG error")
    axes[0, 3].set_title("SIAC error")

    # Shared colorbars
    fig.colorbar(
        im_sol,
        ax=axes[:, 0:2],
        location="right",
        shrink=0.9,
        label="Solution value",
    )

    fig.colorbar(
        im_err,
        ax=axes[:, 2:4],
        location="right",
        shrink=0.9,
        label=r"$\log_{10}(|\mathrm{error}|)$",
    )

    plt.show()