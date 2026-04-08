import numpy as np


def compute_rates(errors, hs):
    """
    Compute observed convergence rates.

    Parameters
    ----------
    errors : array_like
        Error values.
    hs : array_like
        Mesh sizes.

    Returns
    -------
    rates : ndarray
        Observed rates, with np.nan for the first entry.
    """
    errors = np.asarray(errors, dtype=float)
    hs = np.asarray(hs, dtype=float)

    rates = np.full(len(errors), np.nan, dtype=float)

    for i in range(1, len(errors)):
        if errors[i] > 0.0 and errors[i - 1] > 0.0:
            rates[i] = np.log(errors[i - 1] / errors[i]) / np.log(hs[i - 1] / hs[i])

    return rates


def run_convergence_study_1d(
    exact_func,
    K_list,
    p,
    xlim=(-1, 1),
    poly_deg=None,
    moments=None,
    BSorder=None,
    n_eval_per_mode_factor=50,
    quad_order=None,
    apply_noise=False,
    relative_error=True,
    build_uniform_mesh_1d=None,
    local_cell_center_nodes_1d=None,
    build_grid_from_local_nodes_1d=None,
    l2_project_exact_func_to_dg_1d=None,
    eval_dg_on_local_nodes_1d=None,
    apply_siac_to_modal_dg_1d=None,
    trim_valid_siac_region_1d=None,
):
    """
    Run a 1D convergence study for DG and SIAC using exact L2 projection.

    Parameters
    ----------
    exact_func : callable
        Exact function with signature exact_func(x, xlim=..., degree=...).
    K_list : iterable of int
        Mesh resolutions.
    p : int
        DG polynomial degree.
    xlim : tuple
        Domain limits.
    poly_deg : int or None
        Degree passed to the exact polynomial test function.
    moments : int or None
        SIAC moments. Defaults to 2*p.
    BSorder : int or None
        SIAC B-spline order. Defaults to p+1.
    n_eval_per_mode_factor : int
        Number of local evaluation nodes per quadrature-based scale factor.
    quad_order : int or None
        Quadrature order for projection.
    apply_noise : bool
        Passed into the exact projection routine.
    relative_error : bool
        If True, compute relative L2 errors, else absolute L2 errors.

    Returns
    -------
    results : dict
        Dictionary containing K, h, DG/SIAC errors, and rates.
    """
    if moments is None:
        moments = 2 * p
    if BSorder is None:
        BSorder = p + 1

    if quad_order is None:
        if poly_deg is None:
            quad_order = max(2 * p + 4, 8)
        else:
            quad_order = int(np.ceil((poly_deg + p + 1) / 2))

    n_eval = n_eval_per_mode_factor * quad_order
    nodes_fine = local_cell_center_nodes_1d(n_eval)
    
    K_max = max(K_list)
    base_n_eval = n_eval_per_mode_factor * quad_order

    K_vals = []
    h_vals = []
    trims = []
    dg_max_err = []
    dg_rel_l2_err = []
    siac_max_err = []
    siac_rel_l2_err = []

    for_loop_lenght = len(K_list)
    for iter, K in enumerate(K_list):
        print(f"Iteration {iter+1} of {for_loop_lenght}, K = {K} ")
        
        if K_max % K != 0:
            raise ValueError("Each K must divide K_max to keep a common global evaluation density.")
        K_factor = int(K_max / K)
        n_eval = base_n_eval * K_factor
        nodes_fine = local_cell_center_nodes_1d(n_eval)
        
        mesh = build_uniform_mesh_1d(K=K, p=p, domain=xlim)

        dg = l2_project_exact_func_to_dg_1d(
            func=exact_func,
            mesh=mesh,
            poly_max_deg=poly_deg,
            quad_order=quad_order,
            add_noise=apply_noise,
        )

        grid_fine = build_grid_from_local_nodes_1d(
            mesh=dg["mesh"],
            eval_nodes=nodes_fine,
        )

        U_exact = exact_func(
            x=grid_fine
        )

        U_dg = eval_dg_on_local_nodes_1d(
            dg,
            eval_nodes=nodes_fine,
        )

        U_siac = apply_siac_to_modal_dg_1d(
            dg,
            moments=moments,
            BSorder=BSorder,
            eval_nodes=nodes_fine,
        )

        exact_trim, trim = trim_valid_siac_region_1d(
            U_exact,
            n_eval=n_eval,
            moments=moments,
            BSorder=BSorder,
            return_trim=True,
        )

        dg_trim = trim_valid_siac_region_1d(
            U_dg,
            n_eval=n_eval,
            moments=moments,
            BSorder=BSorder,
        )

        siac_trim = trim_valid_siac_region_1d(
            U_siac,
            n_eval=n_eval,
            moments=moments,
            BSorder=BSorder,
        )

        err_dg = dg_trim - exact_trim
        err_siac = siac_trim - exact_trim

        dg_max_val = np.max(np.abs(err_dg))
        siac_max_val = np.max(np.abs(err_siac))

        denom = np.linalg.norm(exact_trim)
        dg_rel_l2_val = np.linalg.norm(err_dg) / denom
        siac_rel_l2_val = np.linalg.norm(err_siac) / denom

        K_vals.append(K)
        h_vals.append(mesh["h"])
        
        dg_max_err.append(dg_max_val)
        dg_rel_l2_err.append(dg_rel_l2_val)
        siac_max_err.append(siac_max_val)
        siac_rel_l2_err.append(siac_rel_l2_val)
        
        trims.append(trim)

    K_vals = np.asarray(K_vals, dtype=int)
    h_vals = np.asarray(h_vals, dtype=float)
    
    dg_max_err = np.asarray(dg_max_err)
    dg_rel_l2_err = np.asarray(dg_rel_l2_err)
    siac_max_err = np.asarray(siac_max_err)
    siac_rel_l2_err = np.asarray(siac_rel_l2_err)

    return {
        "K": K_vals,
        "h": h_vals,
        "dg_max": dg_max_err,
        "siac_max": siac_max_err,
        "dg_rel_l2": dg_rel_l2_err,
        "siac_rel_l2": siac_rel_l2_err,
        "dg_max_rate": compute_rates(dg_max_err, h_vals),
        "siac_max_rate": compute_rates(siac_max_err, h_vals),
        "dg_rel_l2_rate": compute_rates(dg_rel_l2_err, h_vals),
        "siac_rel_l2_rate": compute_rates(siac_rel_l2_err, h_vals),
        "p": p,
        "poly_deg": poly_deg,
        "moments": moments,
        "BSorder": BSorder,
        "quad_order": quad_order,
        "n_eval": n_eval,
        "trims": np.asarray(trims, dtype=int),
        "apply_noise": apply_noise,
    }


def run_convergence_study_2d(
    exact_func,
    K_list,
    p,
    xlim=(-1, 1),
    ylim=(-1, 1),
    poly_deg=None,
    moments=None,
    BSorder=None,
    n_eval_per_mode_factor=10,
    quad_order=None,
    apply_noise=False,
    relative_error=True,
    build_uniform_mesh_2d=None,
    local_cell_center_nodes_1d=None,
    build_grid_from_local_nodes_2d=None,
    l2_project_exact_func_to_dg_2d=None,
    eval_dg_on_local_nodes_2d=None,
    apply_siac_modal_dg_2d=None,
    trim_valid_siac_region_2d=None,
):
    """
    Run a 2D convergence study for DG and SIAC using exact L2 projection.

    Parameters
    ----------
    exact_func : callable
        Exact function with signature exact_func(x, y, xlim=..., ylim=..., degree=...).
    K_list : iterable of int
        Mesh resolutions. Uses Kx = Ky = K.
    p : int
        DG polynomial degree.
    xlim, ylim : tuple
        Domain limits.
    poly_deg : int or None
        Degree passed to the exact polynomial test function.
    moments : int or None
        SIAC moments. Defaults to 2*p.
    BSorder : int or None
        SIAC B-spline order. Defaults to p+1.
    n_eval_per_mode_factor : int
        Number of local evaluation nodes per quadrature-based scale factor.
    quad_order : int or None
        Quadrature order for projection.
    apply_noise : bool
        Passed into the exact projection routine.
    relative_error : bool
        If True, compute relative L2 errors, else absolute L2 errors.

    Returns
    -------
    results : dict
        Dictionary containing K, h, DG/SIAC errors, and rates.
    """
    if moments is None:
        moments = 2 * p
    if BSorder is None:
        BSorder = p + 1

    if quad_order is None:
        if poly_deg is None:
            quad_order = max(2 * p + 4, 8)
        else:
            quad_order = int(np.ceil((poly_deg + p + 1) / 2))

    K_max = max(K_list)
    base_n_eval = n_eval_per_mode_factor * quad_order

    K_vals = []
    h_vals = []
    trims = []
    dg_max_err = []
    dg_rel_l2_err = []
    siac_max_err = []
    siac_rel_l2_err = []

    for_loop_lenght = len(K_list)
    for iter, K in enumerate(K_list):
        print(f"Iteration {iter+1} of {for_loop_lenght}, K = {K} ")
        
        if K_max % K != 0:
            raise ValueError("Each K must divide K_max to keep a common global evaluation density.")
        K_factor = int(K_max / K)
        n_eval = base_n_eval * K_factor
        nodes_fine = local_cell_center_nodes_1d(n_eval)
        
        mesh = build_uniform_mesh_2d(Kx=K, Ky=K, p=p, xlim=xlim, ylim=ylim)

        dg = l2_project_exact_func_to_dg_2d(
            func=exact_func,
            mesh=mesh,
            poly_max_deg=poly_deg,
            quad_order=quad_order,
            add_noise=apply_noise,
        )

        X_fine, Y_fine = build_grid_from_local_nodes_2d(
            dg["mesh"],
            nodes_fine,
        )

        U_exact = exact_func(
            x=X_fine,
            y=Y_fine
        )

        U_dg = eval_dg_on_local_nodes_2d(
            dg,
            eval_nodes=nodes_fine,
        )

        U_siac = apply_siac_modal_dg_2d(
            dg,
            moments=moments,
            BSorder=BSorder,
            eval_nodes=nodes_fine,
        )

        exact_trim, trim = trim_valid_siac_region_2d(
            U_exact,
            n_eval=n_eval,
            moments=moments,
            BSorder=BSorder,
            return_trim=True,
        )

        dg_trim = trim_valid_siac_region_2d(
            U_dg,
            n_eval=n_eval,
            moments=moments,
            BSorder=BSorder,
        )

        siac_trim = trim_valid_siac_region_2d(
            U_siac,
            n_eval=n_eval,
            moments=moments,
            BSorder=BSorder,
        )

        err_dg = dg_trim - exact_trim
        err_siac = siac_trim - exact_trim

        dg_max_val = np.max(np.abs(err_dg))
        siac_max_val = np.max(np.abs(err_siac))

        denom = np.linalg.norm(exact_trim)
        dg_rel_l2_val = np.linalg.norm(err_dg) / denom
        siac_rel_l2_val = np.linalg.norm(err_siac) / denom

        K_vals.append(K)
        
        h = (1/np.sqrt(2)) * np.sqrt(mesh["hx"]**2 + mesh["hy"]**2)
        h_vals.append(h)
        
        dg_max_err.append(dg_max_val)
        dg_rel_l2_err.append(dg_rel_l2_val)
        siac_max_err.append(siac_max_val)
        siac_rel_l2_err.append(siac_rel_l2_val)
        
        trims.append(trim)

    K_vals = np.asarray(K_vals, dtype=int)
    h_vals = np.asarray(h_vals, dtype=float)
    
    dg_max_err = np.asarray(dg_max_err)
    dg_rel_l2_err = np.asarray(dg_rel_l2_err)
    siac_max_err = np.asarray(siac_max_err)
    siac_rel_l2_err = np.asarray(siac_rel_l2_err)

    return {
        "K": K_vals,
        "h": h_vals,
        "dg_max": dg_max_err,
        "siac_max": siac_max_err,
        "dg_rel_l2": dg_rel_l2_err,
        "siac_rel_l2": siac_rel_l2_err,
        "dg_max_rate": compute_rates(dg_max_err, h_vals),
        "siac_max_rate": compute_rates(siac_max_err, h_vals),
        "dg_rel_l2_rate": compute_rates(dg_rel_l2_err, h_vals),
        "siac_rel_l2_rate": compute_rates(siac_rel_l2_err, h_vals),
        "p": p,
        "poly_deg": poly_deg,
        "moments": moments,
        "BSorder": BSorder,
        "quad_order": quad_order,
        "n_eval": n_eval,
        "trims": np.asarray(trims, dtype=int),
        "apply_noise": apply_noise,
    }
