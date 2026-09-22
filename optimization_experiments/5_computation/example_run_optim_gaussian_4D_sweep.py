import numpy as np
import matplotlib.pyplot as plt
import string
import json
import warnings

plt.rcParams.update(
    {
        "font.size": 11,  # base font size (tick labels)
        "axes.labelsize": 13,  # axis labels
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 300,
    }
)


import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices("cpu")[0])

# Import your optimization and plotting utilities
from optimization_core import (
    build_jsa_grid,
    scaled_gaussian_jsa_vectorized_rsvd,
    penalty_norm_relaxation_box_grad_desc,
    save_results,
    load_results,
    to_complex,
)

from tensorrsvd import ho_rsvd

# ---------------------------------------------------------
# 1. Experiment parameters
# ---------------------------------------------------------

# Parameters for the multivariate normal distribution
n = 50  # Number of points in the frequency grid nxnxnxn
d = 4  # Dimensionality of the problem
mean = jnp.zeros(d)  # Mean
x_lim = 4  # Limits in x and y ax0is for plotting

# # Sweep parameters
n_steps = 71  # Number of steps in the sweep
rho_values = jnp.linspace(
    -0.33, 0.99, n_steps
)  # rho_values is analogous to the correlation coefficients in the correlation matrix

chunks = jnp.array_split(rho_values, 7)


# ---------------------------------------------------------
# 2. Parameters for the optimization
# ---------------------------------------------------------

# Penalty parameters
lambda0 = 1e-2
lambda_factor = 2
tol_constraint = 1e-6
max_outer_iters = 80
max_inner_iters = 500


# P-norm ramping schedule
num_stages = 30  # I will determine the appropriate number of stages based on how quickly the optimization converges typically.
p_schedule = np.linspace(2 * d, 2.0, num_stages)


def run_single_optimization(rho_value):

    # ---------------------------------------------------------
    # 1. Build JSA grid and initial mode
    # ---------------------------------------------------------
    _, jsa_data = build_jsa_grid(n, mean, rho_value, x_lim)

    # Random SVD
    f = scaled_gaussian_jsa_vectorized_rsvd(r=rho_value, mean=mean, x_lim=x_lim)
    rank = 6
    U_list, S_list = ho_rsvd(
        tensor=f,
        tensor_shape=(n,) * d,
        dtype=jnp.float64,
        rank=rank,
        num_oversamples=10,
        num_power_iterations=2,
        num_idxs=d,
        backend="jax",
    )

    u = U_list[0]

    # ---------------------------------------------------------
    # 2. Load lower bound for this rho
    # ---------------------------------------------------------
    bounds_data = np.load(f"bounds_gaussian_{d}D.npz")
    rho_arr = bounds_data["rho"]
    lower_bound_arr = bounds_data["lower_bound"]

    idx = np.argmin((rho_arr - rho_value) ** 2)
    lower_bound = lower_bound_arr[idx]

    # ---------------------------------------------------------
    # 3. Try each of the first `rank` singular vectors as seeds
    # ---------------------------------------------------------
    best_f_pen = None
    best_fun_pen = -np.inf
    seed_log = []

    for k in range(rank):
        u_init = u[:, k] / jnp.linalg.norm(u[:, k])
        u_init_normal = u_init / jnp.linalg.norm(u_init)
        f_init = jnp.concatenate([u_init_normal, jnp.zeros_like(u_init_normal)])

        f_pen, fun_pen, _ = penalty_norm_relaxation_box_grad_desc(
            ini_seed=f_init,
            jsa=jsa_data,
            p_schedule=p_schedule,
            lambda0=lambda0,
            lambda_factor=lambda_factor,
            tol_constraint=tol_constraint,
            max_outer_iters=max_outer_iters,
            max_inner_iters=max_inner_iters,
        )

        accepted = bool(fun_pen >= lower_bound)

        seed_log.append(
            {
                "seed_index": k,
                "fun_pen": float(fun_pen),
                "accepted": accepted,
            }
        )

        if accepted and fun_pen > best_fun_pen:
            best_fun_pen = fun_pen
            best_f_pen = f_pen

    # ---------------------------------------------------------
    # 4. Save results (best result) + lightweight run log
    # ---------------------------------------------------------
    log_filename = f"run_log_{d}D_r_{rho_value:.2f}.json"
    log_data = {
        "d": d,
        "rho": float(rho_value),
        "lower_bound": float(lower_bound),
        "best_fun_pen": float(best_fun_pen) if best_f_pen is not None else None,
        "seeds": seed_log,
    }
    with open(log_filename, "w") as fh:
        json.dump(log_data, fh, indent=2)

    if best_f_pen is None:
        warnings.warn(
            f"All {rank} seed initializations were rejected for rho={rho_value:.4f} "
            f"(none exceeded lower_bound={lower_bound:.6g}). "
            f"Skipping save of results pickle. See log file '{log_filename}' for details."
        )
        return best_f_pen, best_fun_pen

    filename = f"eigen_eval_results_{d}D_20260615_r_{rho_value:.2f}.pkl"
    save_results(
        {"optim_lo": best_f_pen, "eta": best_fun_pen, "rho": rho_value},
        None,
        None,
        filename,
    )

    return best_f_pen, best_fun_pen


total = len(rho_values)
processed = 0


for i, chunk in enumerate(chunks):
    print(f"Running chunk {i+1}/{len(chunks)}")

    for rho_value in chunk:
        f_pen_vals, fun_pen_vals = run_single_optimization(rho_value)

        # Update progress
        processed += len(chunk)
        percent = 100 * processed / total
        print(f"Progress: {percent:.1f}% completed\n")

    print()  # newline after each chunk
