import numpy as np
import pandas as pd
import jax

import gc

jax.config.update("jax_enable_x64", True)

from optimization_core import (
    optimizer_grad_desc_penalty,
    optimizer_basinhop_constrained,
    optimizer_grad_desc_penalty_norm_ramp,
    optimizer_box_grad_desc_penalty_norm_ramp,
)

# ---------------------------------------------------------------
#          Penalty ramping gradient descent
# ---------------------------------------------------------------


def penalty_continuation_grad_desc(
    ini_seed,
    jsa,
    lambda0=1e-2,
    lambda_factor=2,
    tol_constraint=1e-6,  # Final desired constraint precision
    max_outer_iters=40,
    max_inner_iters=1000,
):
    lam = lambda0
    f = ini_seed.copy()
    all_outer_info = []

    # Parameters for accelerating and stopping earlier if tolerance is reached
    const_threshold = 1e-2  # Error to tighten the penalty parameter
    tighten_lambda_factor = 5  # 5.0
    obj_tol = 1e-6  # Stop if Objective function reaches this precision
    init_obj = float("inf")

    # 3-phase continuation schedule to avoid false convergence in the discovery phase
    lambda_min_required = 1e3  # minimum λ before stopping is allowed
    entered_tightening_phase = False  # state flag

    for outer in range(max_outer_iters):
        print(f"\n=== Outer {outer} | λ = {lam:.2e} ===")

        f_opt, fun_val, inner_hist, converged, exit_code = optimizer_grad_desc_penalty(
            initial_seed=f, jsa=jsa, lambda_penalty=lam, max_iters=max_inner_iters
        )

        # --- handle failure safely ---
        if (not converged) or (f_opt is None):
            print(f"[Warning] Inner optimization failed at outer {outer}, λ={lam:.2e}.")
            print("Keeping previous f and continuing.")

            # Compute violation using previous f
            f_re, f_im = np.split(f, 2)
            f_comp = f_re + 1j * f_im
            violation = abs(np.linalg.norm(f_comp) - 1)

            all_outer_info.append(
                {
                    "lambda": lam,
                    "violation": violation,
                    "fun": None,
                    # "inner_history": inner_hist["objective"] if inner_hist else [],
                    "converged": False,
                    "exit_code": exit_code,
                }
            )

            lam *= lambda_factor
            continue

        # --- compute violation ---
        f_re, f_im = np.split(f_opt, 2)
        f_comp = f_re + 1j * f_im
        violation = abs(np.linalg.norm(f_comp) - 1)

        print(f"Constraint violation: {violation:.3e}")
        print(f"Objective function value: {fun_val:.6e}")

        all_outer_info.append(
            {
                "lambda": lam,
                "violation": violation,
                "fun": fun_val,
                # "inner_history": inner_hist["objective"],
                "converged": converged,
                "exit_code": exit_code,
            }
        )

        # Convergence checks
        objective_precision = abs(init_obj - fun_val)

        # ============================================================
        #   ROBUST STOPPING CONDITION (prevents false convergence)
        # ============================================================

        if (
            entered_tightening_phase
            and lam >= lambda_min_required
            and violation < tol_constraint
            and objective_precision < obj_tol
        ):
            print("Constraint satisfied and objective function converged.")
            return f_opt, fun_val, all_outer_info

        # Hybrid Update Logic
        if violation > const_threshold:
            # Phase 1: Discovery (Stay stable)
            lam *= lambda_factor
        else:
            # Phase 2: Tightening (Finish fast)
            lam *= tighten_lambda_factor
            entered_tightening_phase = True

        # Update init_obj for convergence checks in next iterations
        init_obj = fun_val

        f = f_opt.copy()

    print("Warning: maximum outer iterations reached.")
    return f_opt, fun_val, all_outer_info


def constraint_violation(f):
    f_re, f_im = np.split(f, 2)
    f_comp = f_re + 1j * f_im
    return abs(np.linalg.norm(f_comp) - 1)


def run_comparison(
    f_opt_init,
    jsa,
    K=10,
    bh_kwargs=None,  # Basin hopping parameters (temp, step_size, n_iter, displ)
    **kwargs,
):
    results = []

    # Optimal seed
    f0 = f_opt_init.copy()

    f_pen, fun_pen, info_pen = penalty_continuation_grad_desc(f0, jsa, **kwargs)
    if f_pen is None:
        # Penalty method failed → store failure cleanly
        results.append(
            {
                "method": "penalty",
                "seed_type": "u_init",
                "outer_iters": len(info_pen) if info_pen is not None else None,
                "final_fun": None,
                "final_violation": None,
                "f_opt": None,
                # "convergence_path": None,
                "lambdas": None,
                "outer_info": info_pen,
                "all_outer_converged": False,
            }
        )
    else:
        # Success → store normally
        results.append(
            {
                "method": "penalty",
                "seed_type": "u_init",
                "outer_iters": len(info_pen),
                "final_fun": float(fun_pen),
                "final_violation": float(info_pen[-1]["violation"]),
                "f_opt": f_pen.copy(),
                # "convergence_path": [h["inner_history"] for h in info_pen],
                "lambdas": [h["lambda"] for h in info_pen],
                "outer_info": info_pen,
            }
        )

    # ---------- Trust-constr (optimal seed) ----------

    bh_kwargs = bh_kwargs or {}
    # Basinhopping defaults
    bh_defaults = {
        "n_iter": 1000,
        "temp": 1.0,
        "step_size": 0.5,
        "displ": False,
    }

    # merge user overrides
    bh_params = {**bh_defaults, **bh_kwargs}

    # unpack
    n_iter = bh_params["n_iter"]
    temp = bh_params["temp"]
    step_size = bh_params["step_size"]
    displ = bh_params["displ"]

    f_tr, fun_tr, info_tr, success_tr = optimizer_basinhop_constrained(
        f_opt_init,
        jsa,
        p_norm=2,
        n_iter=n_iter,
        temp=temp,
        step_size=step_size,
        displ=displ,
    )

    results.append(
        {
            "method": "trust",
            "seed_type": "u_init",
            "outer_iters": None,
            "final_fun": float(fun_tr),
            "final_violation": constraint_violation(f_tr),
            "f_opt": f_tr.copy(),
            # "convergence_path": None,
            "lambdas": None,
            "outer_info": None,
            "trust_success": bool(info_tr.lowest_optimization_result.success),
            "trust_status": int(info_tr.lowest_optimization_result.status),
            "trust_message": str(info_tr.lowest_optimization_result.message),
        }
    )

    # ---------- Random seeds ----------

    for k in range(K):
        f_rand = np.random.randn(len(f0))
        f_rand /= np.linalg.norm(f_rand)

        f_pen, fun_pen, info_pen = penalty_continuation_grad_desc(f_rand, jsa, **kwargs)
        if f_pen is None:
            results.append(
                {
                    "method": "penalty",
                    "seed_type": "random",
                    "outer_iters": len(info_pen) if info_pen is not None else None,
                    "final_fun": None,
                    "final_violation": None,
                    "f_opt": None,
                    # "convergence_path": None,
                    "lambdas": None,
                    "outer_info": info_pen,
                    "all_outer_converged": False,
                }
            )
        else:
            results.append(
                {
                    "method": "penalty",
                    "seed_type": "random",
                    "outer_iters": len(info_pen),
                    "final_fun": float(fun_pen),
                    "final_violation": float(info_pen[-1]["violation"]),
                    "f_opt": f_pen.copy(),
                    # "convergence_path": [h["inner_history"] for h in info_pen],
                    "lambdas": [h["lambda"] for h in info_pen],
                    "outer_info": info_pen,
                }
            )

        f_tr, fun_tr, info_tr, success_tr = optimizer_basinhop_constrained(
            f_opt_init,
            jsa,
            p_norm=2,
            n_iter=n_iter,
            temp=temp,
            step_size=step_size,
            displ=displ,
        )

        results.append(
            {
                "method": "trust",
                "seed_type": "random",
                "outer_iters": None,
                "final_fun": float(fun_tr),
                "final_violation": constraint_violation(f_tr),
                "f_opt": f_tr.copy(),
                # "convergence_path": None,
                "lambdas": None,
                "outer_info": None,
                "trust_success": bool(info_tr.lowest_optimization_result.success),
                "trust_status": int(info_tr.lowest_optimization_result.status),
                "trust_message": str(info_tr.lowest_optimization_result.message),
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------
#          Penalty ramping runner
# ---------------------------------------------------------------


def run_penalty_only(f_opt_init, jsa, seeds, **kwargs):
    results = []

    # Optimal seed
    f0 = f_opt_init.copy()
    f_pen, fun_pen, info_pen = penalty_continuation_grad_desc(f0, jsa, **kwargs)

    results.append(
        {
            "method": "penalty",
            "seed_type": "u_init",
            "seed_index": None,
            "outer_iters": len(info_pen) if info_pen is not None else None,
            "final_fun": float(fun_pen) if f_pen is not None else None,
            "final_violation": (
                float(info_pen[-1]["violation"]) if f_pen is not None else None
            ),
            "f_opt": f_pen.copy() if f_pen is not None else None,
            "lambdas": (
                [h["lambda"] for h in info_pen] if info_pen is not None else None
            ),
            "outer_info": info_pen,
        }
    )

    # Random seeds
    for idx, s in enumerate(seeds):
        rng = np.random.default_rng(s)
        f_rand = rng.normal(size=len(f0))
        f_rand /= np.linalg.norm(f_rand)

        f_pen, fun_pen, info_pen = penalty_continuation_grad_desc(f_rand, jsa, **kwargs)

        results.append(
            {
                "method": "penalty",
                "seed_type": "random",
                "seed_index": idx,
                "outer_iters": len(info_pen) if info_pen is not None else None,
                "final_fun": float(fun_pen) if f_pen is not None else None,
                "final_violation": (
                    float(info_pen[-1]["violation"]) if f_pen is not None else None
                ),
                "f_opt": f_pen.copy() if f_pen is not None else None,
                "lambdas": (
                    [h["lambda"] for h in info_pen] if info_pen is not None else None
                ),
                "outer_info": info_pen,
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------
#          Trust constraint  runner
# ---------------------------------------------------------------


def run_trust_only(f_opt_init, jsa, seeds, bh_kwargs=None):
    results = []

    # Basinhopping defaults
    bh_defaults = {
        "n_iter": 1000,
        "temp": 1.0,
        "step_size": 0.5,
        "displ": False,
    }
    bh_params = {**bh_defaults, **(bh_kwargs or {})}

    # Optimal seed
    f_tr, fun_tr, info_tr, success_tr = optimizer_basinhop_constrained(
        f_opt_init, jsa, p_norm=2, **bh_params
    )

    results.append(
        {
            "method": "trust",
            "seed_type": "u_init",
            "seed_index": None,
            "final_fun": float(fun_tr),
            "final_violation": constraint_violation(f_tr),
            "f_opt": f_tr.copy(),
            "trust_success": bool(info_tr.lowest_optimization_result.success),
            "trust_status": int(info_tr.lowest_optimization_result.status),
            "trust_message": str(info_tr.lowest_optimization_result.message),
        }
    )

    # Random seeds
    for idx, s in enumerate(seeds):
        rng = np.random.default_rng(s)
        f_rand = rng.normal(size=len(f_opt_init))
        f_rand /= np.linalg.norm(f_rand)

        f_tr, fun_tr, info_tr, success_tr = optimizer_basinhop_constrained(
            f_rand, jsa, p_norm=2, **bh_params
        )

        results.append(
            {
                "method": "trust",
                "seed_type": "random",
                "seed_index": idx,
                "final_fun": float(fun_tr),
                "final_violation": constraint_violation(f_tr),
                "f_opt": f_tr.copy(),
                "trust_success": bool(info_tr.lowest_optimization_result.success),
                "trust_status": int(info_tr.lowest_optimization_result.status),
                "trust_message": str(info_tr.lowest_optimization_result.message),
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------
#          Penalty and norm relaxation/homotopy continuation runners 
# ---------------------------------------------------------------


def penalty_norm_relaxation_grad_desc(
    ini_seed,
    jsa,
    p_schedule,
    lambda0=1e-2,
    lambda_factor=2,
    tol_constraint=1e-6,  # Final desired constraint precision
    max_outer_iters=40,
    max_inner_iters=1000,
):
    lam = lambda0
    f = ini_seed.copy()
    all_outer_info = []

    # Parameters for accelerating and stopping earlier if tolerance is reached
    const_threshold = 1e-2  # Error to tighten the penalty parameter
    tighten_lambda_factor = 5.0
    obj_tol = 1e-6  # Stop if Objective function reaches this precision
    init_obj = float("inf")

    for outer in range(max_outer_iters):
        p_idx = min(
            outer, len(p_schedule) - 1
        )  # In case the schedule is shorter than max_outer_iters-> So we always reach p=2
        p_k = p_schedule[p_idx]

        print(f"\n=== Outer {outer} | p = {p_k:.3f} | λ = {lam:.2e} ===")

        f_opt, fun_val, inner_hist, converged, exit_code = (
            optimizer_grad_desc_penalty_norm_ramp(
                initial_seed=f,
                jsa=jsa,
                p=p_k,
                lambda_penalty=lam,
                max_iters=max_inner_iters,
            )
        )

        # --- handle failure safely ---
        if (not converged) or (f_opt is None):
            print(
                f"[Warning] Inner optimization failed at outer {outer}, λ={lam:.2e}, p = {p_k:.3f}."
            )
            print("Keeping previous f and continuing.")

            # Compute violation using previous f
            f_re, f_im = np.split(f, 2)
            f_comp = f_re + 1j * f_im
            violation = abs(np.linalg.norm(f_comp) - 1)

            all_outer_info.append(
                {
                    "p": p_k,
                    "lambda": lam,
                    "violation": violation,
                    "fun": None,
                    # "inner_history": inner_hist["objective"] if inner_hist else [],
                    "converged": False,
                    "exit_code": exit_code,
                }
            )

            lam *= lambda_factor
            continue

        f_re, f_im = np.split(f_opt, 2)
        f_comp = f_re + 1j * f_im
        violation = abs(np.linalg.norm(f_comp) - 1)

        print(f"Constraint violation: {violation:.3e}")
        print(f"Objective function value: {fun_val:.6e}")

        all_outer_info.append(
            {
                "p": p_k,
                "lambda": lam,
                "violation": violation,
                "fun": fun_val,
                # "inner_history": inner_hist["objective"],
                "converged": converged,
                "exit_code": exit_code,
            }
        )

        # Convergence checks
        objective_precision = abs(init_obj - fun_val)

        # if violation < tol_constraint and objective_precision < obj_tol and p_k <= 2.0001:
        if violation < tol_constraint and objective_precision < obj_tol:
            print("Constraint satisfied and objective function converged  at p≈2..")
            return f_opt, fun_val, all_outer_info

        # Hybrid Update Logic
        if violation > const_threshold:
            # Phase 1: Discovery (Stay stable)
            lam *= lambda_factor
        else:
            # Phase 2: Tightening (Finish fast)
            lam *= tighten_lambda_factor

        # Update init_obj for convergence checks in next iterations
        init_obj = fun_val

        f = f_opt.copy()

        del inner_hist  # Free memory from inner history if stored
        jax.clear_caches()  # Clear JAX caches to prevent memory issues during long runs
        gc.collect()

    print("Warning: maximum outer iterations reached.")
    return f_opt, fun_val, all_outer_info


def run_penalty_norm_relaxation(f_opt_init, jsa, seeds, p_schedule, **kwargs):
    results = []

    # Optimal seed
    f0 = f_opt_init.copy()
    f_pen, fun_pen, info_pen = penalty_norm_relaxation_grad_desc(
        f0, jsa, p_schedule, **kwargs
    )

    results.append(
        {
            "method": "penalty+p-norm",
            "seed_type": "u_init",
            "seed_index": None,
            "outer_iters": len(info_pen) if info_pen is not None else None,
            "final_fun": float(fun_pen) if f_pen is not None else None,
            "final_violation": (
                float(info_pen[-1]["violation"]) if f_pen is not None else None
            ),
            "f_opt": f_pen.copy() if f_pen is not None else None,
            "lambdas": (
                [h["lambda"] for h in info_pen] if info_pen is not None else None
            ),
            "p_values": [h["p"] for h in info_pen] if info_pen is not None else None,
            "outer_info": info_pen,
        }
    )

    # Random seeds
    for idx, s in enumerate(seeds):
        rng = np.random.default_rng(s)
        f_rand = rng.normal(size=len(f0))
        f_rand /= np.linalg.norm(f_rand)

        f_pen, fun_pen, info_pen = penalty_norm_relaxation_grad_desc(
            f_rand, jsa, p_schedule, **kwargs
        )

        results.append(
            {
                "method": "penalty+p-norm",
                "seed_type": "random",
                "seed_index": idx,
                "outer_iters": len(info_pen) if info_pen is not None else None,
                "final_fun": float(fun_pen) if f_pen is not None else None,
                "final_violation": (
                    float(info_pen[-1]["violation"]) if f_pen is not None else None
                ),
                "f_opt": f_pen.copy() if f_pen is not None else None,
                "lambdas": (
                    [h["lambda"] for h in info_pen] if info_pen is not None else None
                ),
                "p_values": (
                    [h["p"] for h in info_pen] if info_pen is not None else None
                ),
                "outer_info": info_pen,
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------
#          Penalty and norm relaxation/homotopy continuation runners with box bounds
# ---------------------------------------------------------------


def penalty_norm_relaxation_box_grad_desc(
    ini_seed,
    jsa,
    p_schedule,
    lambda0=1e-2,
    lambda_factor=2,
    tol_constraint=1e-6,  # Final desired constraint precision
    max_outer_iters=40,
    max_inner_iters=1000,
):
    lam = lambda0
    f = ini_seed.copy()
    all_outer_info = []

    # Parameters for accelerating and stopping earlier if tolerance is reached
    const_threshold = 1e-2  # Error to tighten the penalty parameter
    tighten_lambda_factor = 5.0
    obj_tol = 1e-6  # Stop if Objective function reaches this precision
    init_obj = float("inf")

    for outer in range(max_outer_iters):
        p_idx = min(
            outer, len(p_schedule) - 1
        )  # In case the schedule is shorter than max_outer_iters-> So we always reach p=2
        p_k = p_schedule[p_idx]

        print(f"\n=== Outer {outer} | p = {p_k:.3f} | λ = {lam:.2e} ===")

        f_opt, fun_val, inner_hist, converged, exit_code = (
            optimizer_box_grad_desc_penalty_norm_ramp(
                initial_seed=f,
                jsa=jsa,
                p=p_k,
                lambda_penalty=lam,
                max_iters=max_inner_iters,
            )
        )

        # --- handle failure safely ---
        if (not converged) or (f_opt is None):
            print(
                f"[Warning] Inner optimization failed at outer {outer}, λ={lam:.2e}, p = {p_k:.3f}."
            )
            print("Keeping previous f and continuing.")

            # Compute violation using previous f
            f_re, f_im = np.split(f, 2)
            f_comp = f_re + 1j * f_im
            violation = abs(np.linalg.norm(f_comp) - 1)

            all_outer_info.append(
                {
                    "p": p_k,
                    "lambda": lam,
                    "violation": violation,
                    "fun": None,
                    # "inner_history": inner_hist["objective"] if inner_hist else [],
                    "converged": False,
                    "exit_code": exit_code,
                }
            )

            lam *= lambda_factor
            continue

        f_re, f_im = np.split(f_opt, 2)
        f_comp = f_re + 1j * f_im
        violation = abs(np.linalg.norm(f_comp) - 1)

        print(f"Constraint violation: {violation:.3e}")
        print(f"Objective function value: {fun_val:.6e}")

        all_outer_info.append(
            {
                "p": p_k,
                "lambda": lam,
                "violation": violation,
                "fun": fun_val,
                # "inner_history": inner_hist["objective"],
                "converged": converged,
                "exit_code": exit_code,
            }
        )

        # Convergence checks
        objective_precision = abs(init_obj - fun_val)

        # if violation < tol_constraint and objective_precision < obj_tol and p_k <= 2.0001:
        if violation < tol_constraint and objective_precision < obj_tol:
            print("Constraint satisfied and objective function converged  at p≈2..")
            return f_opt, fun_val, all_outer_info

        # Hybrid Update Logic
        if violation > const_threshold:
            # Phase 1: Discovery (Stay stable)
            lam *= lambda_factor
        else:
            # Phase 2: Tightening (Finish fast)
            lam *= tighten_lambda_factor

        # Update init_obj for convergence checks in next iterations
        init_obj = fun_val

        f = f_opt.copy()

        del inner_hist  # Free memory from inner history if stored
        jax.clear_caches()  # Clear JAX caches to prevent memory issues during long runs
        gc.collect()

    print("Warning: maximum outer iterations reached.")
    return f_opt, fun_val, all_outer_info


def run_penalty_norm_relaxation_box(f_opt_init, jsa, seeds, p_schedule, **kwargs):
    results = []

    # Optimal seed
    f0 = f_opt_init.copy()
    f_pen, fun_pen, info_pen = penalty_norm_relaxation_box_grad_desc(
        f0, jsa, p_schedule, **kwargs
    )

    results.append(
        {
            "method": "penalty+p-norm",
            "seed_type": "u_init",
            "seed_index": None,
            "outer_iters": len(info_pen) if info_pen is not None else None,
            "final_fun": float(fun_pen) if f_pen is not None else None,
            "final_violation": (
                float(info_pen[-1]["violation"]) if f_pen is not None else None
            ),
            "f_opt": f_pen.copy() if f_pen is not None else None,
            "lambdas": (
                [h["lambda"] for h in info_pen] if info_pen is not None else None
            ),
            "p_values": [h["p"] for h in info_pen] if info_pen is not None else None,
            "outer_info": info_pen,
        }
    )

    # Random seeds
    for idx, s in enumerate(seeds):
        rng = np.random.default_rng(s)
        f_rand = rng.normal(size=len(f0))
        f_rand /= np.linalg.norm(f_rand)

        f_pen, fun_pen, info_pen = penalty_norm_relaxation_box_grad_desc(
            f_rand, jsa, p_schedule, **kwargs
        )

        results.append(
            {
                "method": "penalty+p-norm",
                "seed_type": "random",
                "seed_index": idx,
                "outer_iters": len(info_pen) if info_pen is not None else None,
                "final_fun": float(fun_pen) if f_pen is not None else None,
                "final_violation": (
                    float(info_pen[-1]["violation"]) if f_pen is not None else None
                ),
                "f_opt": f_pen.copy() if f_pen is not None else None,
                "lambdas": (
                    [h["lambda"] for h in info_pen] if info_pen is not None else None
                ),
                "p_values": (
                    [h["p"] for h in info_pen] if info_pen is not None else None
                ),
                "outer_info": info_pen,
            }
        )

    return pd.DataFrame(results)
