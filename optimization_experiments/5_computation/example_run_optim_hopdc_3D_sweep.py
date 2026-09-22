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
# jax.config.update("jax_default_device", jax.devices("cpu")[0])

# Import your optimization and plotting utilities
from optimization_core import (
    scaled_spdc_jsa_vectorized,
    scaled_spdc_jsa_vectorized_rsvd,
    penalty_norm_relaxation_box_grad_desc,
    save_results,
    load_results,
    to_complex,
)

from tensorrsvd import ho_rsvd

# ---------------------------------------------------------
# 1. Experiment parameters
# ---------------------------------------------------------

# # Sweep parameters
n_steps = 80  # Number of steps in the sweep
# taup=jnp.linspace(-0.49, 0.99 , n_steps) #b is analogous to the correlation coefficients in the correlation matrix

# Dispersion parameters
ldelk01 = 0  # Phase mismatch at the central frequencies
tau11 = 0  # Group-velocity matching
# taup1=1 # \beta_P << \beta_F -->\beta_P negligible in comparison to \beta_F
taup = jnp.linspace(
    0, 9, n_steps
)  # We can also sweep over taup1, which is the parameter that controls the pump bandwidth. This allows us to explore different regimes of pump bandwidth in relation to the phase-matching function.
tauf1 = 1.5  # \beta_F >> \beta_P
T0 = tauf1 / jnp.sqrt(jnp.pi)  # Pump bandwidth
T01 = 0  # Ultrafast pump approximation

# chunks = jnp.array_split(taup, 12)
# Compute chunk boundaries
n_chunks = 20
indices = jnp.linspace(0, n_steps, n_chunks + 1, dtype=int)

chunks = [taup[indices[i] : indices[i + 1]] for i in range(n_chunks)]

# Geometrical parameters
n_res = 30  # Number of points inside the bandwidth of the phase-matching function in the frequency grid nxnxn
d = 3  # Dimensionality of the problem

# Spectral resolution
sigmapm = (
    jnp.sqrt(jnp.pi) / tauf1
)  # Approximate HWHM of the phase-matching function, measured in terms of the sinc's main lobe width
delw = (
    sigmapm / n_res
)  # The spectral resolution is defined such that a minimum of n discrete points are sampled within each frequency vector over the bandwidth of the phase-matching function, regardless of the specific width of the evaluation function.

# Spectral window
domg = (
    4 * sigmapm
)  # We set the half-width of the frequency window wide enough to sample correctly both the jsa and reduced density matrix
omf = jnp.arange(-domg, domg, delw)
n = omf.size
print(
    n
)  # Note: Always check the frequency vector size, because later we are going to build a 3D grid with it, which greatly affects  the memory usage.

# Create N‑dimensional meshgrid
grid = jnp.meshgrid(*([omf] * d), indexing="ij")


# ---------------------------------------------------------
# 2. Parameters for the optimization
# ---------------------------------------------------------

# Penalty parameters
lambda0 = 1e-2
lambda_factor = 1.5
tol_constraint = 1e-6
max_outer_iters = 80
max_inner_iters = 500


# P-norm ramping schedule
num_stages = 20  # I will determine the appropriate number of stages based on how quickly the optimization converges typically.
p_schedule = np.linspace(2 * d, 2.0, num_stages)


def run_single_optimization(taup_value):

    # ---------------------------------------------------------
    # 1. Build JSA grid and initial mode
    # ---------------------------------------------------------
    jsa = scaled_spdc_jsa_vectorized(
        ldelk0=ldelk01, tau1=tau11, taup=taup_value, tauf=tauf1, T0=T01
    )
    jsa_data = jsa(*grid)
    jsa_norm = np.linalg.norm(jsa_data)
    jsa_data /= jsa_norm

    # Random SVD
    f = scaled_spdc_jsa_vectorized_rsvd(
        ldelk0=ldelk01, tau1=tau11, taup=taup_value, tauf=tauf1, domg=domg, T0=T01
    )
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
    s = S_list[0] / jsa_norm
    u = U_list[0]

    # ---------------------------------------------------------
    # 2. Load lower bound for this (taup, tauf) pair
    # ---------------------------------------------------------
    bounds_data = np.load(f"bounds_hopdc_{d}D.npz")
    taup_arr = bounds_data["taup"]
    tauf_arr = bounds_data["tauf"]
    lower_bound_arr = bounds_data["lower_bound"]

    # Find the closest matching (taup, tauf) entry in the bounds file
    idx = np.argmin((taup_arr - taup_value) ** 2 + (tauf_arr - tauf1) ** 2)
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
    log_filename = f"run_log_hopdc_{d}D_taup_{taup_value:.2f}_tauf_{tauf1:.2f}.json"
    log_data = {
        "d": d,
        "taup": float(taup_value),
        "tauf": float(tauf1),
        "lower_bound": float(lower_bound),
        "best_fun_pen": float(best_fun_pen) if best_f_pen is not None else None,
        "seeds": seed_log,
    }
    with open(log_filename, "w") as fh:
        json.dump(log_data, fh, indent=2)

    if best_f_pen is None:
        warnings.warn(
            f"All {rank} seed initializations were rejected for taup={taup_value:.4f}, "
            f"tauf={tauf1:.4f} (none exceeded lower_bound={lower_bound:.6g}). "
            f"Skipping save of results pickle. See log file '{log_filename}' for details."
        )
        return best_f_pen, best_fun_pen

    filename = f"eigen_eval_results_hopdc_{d}D_20260620_taup_{taup_value:.2f}_tauf_{tauf1:.2f}.pkl"
    save_results(
        {
            "optim_lo": best_f_pen,
            "eta": best_fun_pen,
            "taup": taup_value,
            "tauf": tauf1,
        },
        None,
        None,
        filename,
    )

    return best_f_pen, best_fun_pen


# # Run single optimization
# taup_value=5.276
# f_pen_vals, fun_pen_vals = run_single_optimization(taup_value)


# Run optimization sweep on taup

total = len(taup)
processed = 0
for i, chunk in enumerate(chunks):
    print(f"Running chunk {i+1}/{len(chunks)}")

    for taup_value in chunk:
        f_pen_vals, fun_pen_vals = run_single_optimization(taup_value)

    # Update progress
    processed += len(chunk)
    percent = 100 * processed / total
    print(f"Progress: {percent:.1f}% completed\n")

    print()  # newline after each chunk
