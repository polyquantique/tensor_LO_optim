import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp

import matplotlib.pyplot as plt
import jax.numpy as jnp
import itertools

import numpy as np

import math


def plot_modes(u, s, n_modes=None):
    weighted_u = u * s

    if n_modes is None:
        n_modes = u.shape[-1]

    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Weighted singular vectors
    for i in range(n_modes):
        axes[0].plot(weighted_u[:, i], label=f"Mode {i}")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Weighted Left Singular Vectors")
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Singular values (Scree plot)
    axes[1].bar(jnp.arange(len(s)), s)
    axes[1].set_xlim(-1, 20)
    axes[1].set_xlabel("Mode Index")
    axes[1].set_ylabel("Singular Value")
    axes[1].set_title("Singular Value Spectrum")
    axes[1].grid(axis="y")

    plt.tight_layout()  # Prevents overlapping labels
    plt.show()


def plot_init_mode(u_init, u_0):
    plt.plot(u_init, label="u_init")
    plt.plot(u_0, label=f"Mode 0-SVD")
    plt.legend()
    plt.title("Initial seed vs first SVD mode")
    plt.show()


def is_totally_symmetric(T, tol=1e-12):
    """
    Check if a tensor T is totally symmetric under all index permutations.
    Works for tensors of any order.
    """
    T = jnp.asarray(T)
    order = T.ndim
    axes = list(range(order))

    # Generate all permutations of the axes
    for perm in itertools.permutations(axes):
        if not jnp.allclose(T, jnp.transpose(T, perm), atol=tol):
            return False
    return True


def plot_2d_projection(jsa_vals):
    """
    Plot a 2D projection of a multivariate PDF f by varying two coordinates
    and fixing the rest at their mean values.

    f: vectorized PDF function created by make_mvn_pdf_vectorized
    mean: (d,) array
    idx: tuple (i,j) selecting which coordinates to vary
    n: grid resolution
    lim: plot range [-lim, lim]
    """
    d = jnp.ndim(jsa_vals)

    jsa_vals /= jnp.linalg.norm(jsa_vals)

    # Square the JSA
    jsa_sq = jsa_vals**2

    # Sum over axes 2..d-1
    axes_to_sum = tuple(range(2, d))
    S = jnp.sum(jsa_sq, axis=axes_to_sum)

    plt.figure(figsize=(6, 5))

    plt.imshow(S, origin="lower", cmap="viridis", alpha=0.85)
    # plt.contour(axis, axis, S, colors="white", linewidths=0.7)

    plt.colorbar(label="Intensity")
    plt.xlabel(r"i")
    plt.ylabel(r"j")
    plt.title("2D Projection of Joint spectral intensity")

    plt.show()


def plot_outer_iterations(df):
    # Accept both penalty-only and penalty+p-norm
    df_pen = df[df["method"].isin(["penalty", "penalty+p-norm"])]

    plt.figure(figsize=(7, 4))
    sns.boxplot(
        data=df_pen,
        x="seed_type",
        y="outer_iters",
        hue="seed_type",
        dodge=False,
        legend=False,
        palette="Set2",
    )

    plt.title("Penalty Method: Outer Iterations (Optimal vs Random)")
    plt.ylabel("Outer iterations")
    plt.xlabel("Seed type")
    plt.grid(axis="y", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_final_objective(df):
    plt.figure(figsize=(7, 4))
    sns.stripplot(
        data=df, x="method", y="final_fun", hue="seed_type", dodge=True, palette="Set1"
    )

    plt.title("Final Objective Values")
    plt.ylabel("Objective")
    plt.xlabel("Method")
    plt.grid(axis="y", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_constraint_violation(df):
    plt.figure(figsize=(7, 4))
    sns.boxplot(
        data=df, x="method", y="final_violation", hue="seed_type", palette="Set3"
    )

    plt.yscale("log")
    plt.title("Final Constraint Violation")
    plt.ylabel("|‖f‖ - 1|")
    plt.xlabel("Method")
    plt.grid(axis="y", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_convergence_from_results(df, index, save=None):
    """
    df: DataFrame returned by run_comparison
    index: row index of the run to plot
    """

    row = df.iloc[index]
    path_nested = row["convergence_path"]

    # Ensure it's a list of lists
    if isinstance(path_nested, float):
        raise ValueError(
            "convergence_path is a float — history was not stored correctly."
        )

    # Flatten inner histories
    full_path = []
    for inner in path_nested:
        full_path.extend(inner)

    plt.figure(figsize=(7, 4))
    plt.plot(full_path, lw=2)
    plt.xlabel("Iteration")
    plt.ylabel("Penalized objective")
    plt.title(f"Convergence Path ({row['method']} - {row['seed_type']})")
    plt.grid(True, ls="--", alpha=0.5)

    if save:
        plt.savefig(save, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_all_penalty_runs(df, title="All penalty convergence paths", save=None):
    """
    df: DataFrame returned by run_comparison
    Plots all convergence paths from all penalty runs.
    """

    # Filter only penalty rows
    df_pen = df[df["method"] == "penalty"]

    if df_pen.empty:
        raise ValueError("No penalty runs found in the DataFrame.")

    plt.figure(figsize=(10, 6))

    for idx, row in df_pen.iterrows():
        path_nested = row["convergence_path"]

        if path_nested is None:
            continue  # skip trust-constr or missing data

        # Retrieve lambdas if stored, else use indices
        lambdas = row.get("lambdas", list(range(len(path_nested))))

        # Plot each λ curve
        for lam, inner_hist in zip(lambdas, path_nested):
            plt.plot(
                inner_hist, lw=1.5, alpha=0.7, label=f"λ={lam:g} ({row['seed_type']})"
            )

    plt.xlabel("Inner iteration")
    plt.ylabel("Penalized objective")
    plt.title(title)
    plt.grid(True, ls="--", alpha=0.5)

    # Avoid duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), title="Penalty λ and seed")

    if save:
        plt.savefig(save, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_subplots_per_lambda(df, save=None):
    df_pen = df[df["method"] == "penalty"]

    # Collect all lambdas across all runs
    all_lambdas = set()
    for _, row in df_pen.iterrows():
        lambdas = row.get("lambdas", list(range(len(row["convergence_path"]))))
        all_lambdas.update(lambdas)
    all_lambdas = sorted(all_lambdas)

    n = len(all_lambdas)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes = axes.flatten()

    for ax, lam in zip(axes, all_lambdas):
        for _, row in df_pen.iterrows():
            lambdas = row.get("lambdas", list(range(len(row["convergence_path"]))))
            if lam in lambdas:
                idx = lambdas.index(lam)
                inner_hist = row["convergence_path"][idx]
                ax.plot(inner_hist, lw=1.5, label=row["seed_type"])

        ax.set_title(f"λ = {lam:g}")
        ax.set_xlabel("Inner iteration")
        ax.set_ylabel("Objective")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_subplots_per_seed(df, save=None):
    df_pen = df[df["method"] == "penalty"]
    seed_types = df_pen["seed_type"].unique()

    fig, axes = plt.subplots(1, len(seed_types), figsize=(7 * len(seed_types), 5))

    if len(seed_types) == 1:
        axes = [axes]

    for ax, seed in zip(axes, seed_types):
        df_seed = df_pen[df_pen["seed_type"] == seed]

        for _, row in df_seed.iterrows():
            lambdas = row.get("lambdas", list(range(len(row["convergence_path"]))))
            for lam, inner_hist in zip(lambdas, row["convergence_path"]):
                ax.plot(inner_hist, lw=1.5, label=f"λ={lam:g}")

        ax.set_title(f"Seed type: {seed}")
        ax.set_xlabel("Inner iteration")
        ax.set_ylabel("Objective")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend()

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_color_by_seed(df, save=None):
    df_pen = df[df["method"] == "penalty"]

    colors = {"u_init": "tab:blue", "random": "tab:orange"}

    plt.figure(figsize=(10, 6))

    for _, row in df_pen.iterrows():
        seed = row["seed_type"]
        lambdas = row.get("lambdas", list(range(len(row["convergence_path"]))))

        for lam, inner_hist in zip(lambdas, row["convergence_path"]):
            plt.plot(
                inner_hist,
                lw=1.5,
                alpha=0.8,
                color=colors.get(seed, "gray"),
                label=f"{seed} (λ={lam:g})",
            )

    plt.xlabel("Inner iteration")
    plt.ylabel("Penalized objective")
    plt.title("All penalty convergence paths (colored by seed type)")
    plt.grid(True, ls="--", alpha=0.5)

    # Deduplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())

    if save:
        plt.savefig(save, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_constraint_violation_vs_lambda(df, index, save=None):
    row = df.iloc[index]

    if row["method"] != "penalty":
        raise ValueError("This plot is only for penalty continuation.")

    info = row["outer_info"] if "outer_info" in row else None
    if info is None:
        raise ValueError("outer_info not stored in df. Add it in run_comparison.")

    lambdas = [d["lambda"] for d in info]
    violations = [d["violation"] for d in info]

    plt.figure(figsize=(7, 5))
    plt.plot(lambdas, violations, marker="o", lw=2)
    plt.xlabel("Penalty parameter λ")
    plt.ylabel("Constraint violation")
    plt.title(f"Constraint violation vs λ ({row['seed_type']})")
    plt.grid(True, ls="--", alpha=0.5)

    if save:
        plt.savefig(save, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_outer_objective_vs_lambda(df, index, save=None):
    row = df.iloc[index]

    if row["method"] != "penalty":
        raise ValueError("This plot is only for penalty continuation.")

    info = row["outer_info"]
    lambdas = [d["lambda"] for d in info]
    fun_vals = [d["fun"] for d in info]

    plt.figure(figsize=(7, 5))
    plt.plot(lambdas, fun_vals, marker="o", lw=2)
    plt.xlabel("Penalty parameter λ")
    plt.ylabel("Penalized objective")
    plt.title(f"Outer-loop objective vs λ ({row['seed_type']})")
    plt.grid(True, ls="--", alpha=0.5)

    if save:
        plt.savefig(save, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_outer_objective_both_seeds(df, title="Outer-loop objective vs λ", save=None):
    """
    df: DataFrame returned by run_comparison
    Produces a single plot with:
      - objective vs λ for u_init seed
      - objective vs λ for random seed
    """

    # Accept both penalty-only and penalty+p-norm
    df_pen = df[df["method"].isin(["penalty", "penalty+p-norm"])]

    if df_pen.empty:
        raise ValueError("No penalty runs found in the DataFrame.")

    # Separate optimal and random runs
    df_opt = df_pen[df_pen["seed_type"] == "u_init"]
    df_rand = df_pen[df_pen["seed_type"] == "random"]

    plt.figure(figsize=(8, 5))

    # Plot optimal seed curve(s)
    for _, row in df_opt.iterrows():
        lambdas = [d["lambda"] for d in row["outer_info"]]
        fun_vals = [d["fun"] for d in row["outer_info"]]
        plt.loglog(
            lambdas, fun_vals, marker="s", lw=2, label="u_init seed", color="tab:blue"
        )

    # Plot random seed curve(s)
    for _, row in df_rand.iterrows():
        lambdas = [d["lambda"] for d in row["outer_info"]]
        fun_vals = [d["fun"] for d in row["outer_info"]]
        plt.loglog(
            lambdas,
            fun_vals,
            marker="o",
            lw=2,
            label="Random seed",
            linestyle="--",
            color="black",
        )

    plt.xlabel("Penalty parameter λ")
    plt.ylabel("Final objective at end of outer iteration")
    plt.title(title)
    plt.grid(True, ls="--", alpha=0.5)

    # Deduplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())

    if save:
        plt.savefig(save, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_final_distributions(df, method_name):

    rows = df[df["method"] == method_name]

    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n))

    if n == 1:
        axes = np.array([axes])

    for i, (_, row) in enumerate(rows.iterrows()):
        f = row["f_opt"]
        f_re, f_im = np.split(f, 2)

        axes[i, 0].plot(f_re, marker="o")
        axes[i, 0].set_title(f"{method_name} | {row['seed_type']} | Real")

        axes[i, 1].plot(f_im, marker="o")
        axes[i, 1].set_title(f"{method_name} | {row['seed_type']} | Imag")

    plt.tight_layout()
    plt.show()


def plot_all_amplitudes(df):

    plt.figure(figsize=(12, 6))

    for _, row in df.iterrows():
        f = row["f_opt"]
        if f is None:
            continue

        # Convert to complex
        f_re, f_im = np.split(f, 2)
        f_comp = f_re + 1j * f_im

        amp = np.abs(f_comp)

        label = f"{row['method']} | {row['seed_type']}"
        plt.plot(amp, marker="o", alpha=0.7, label=label)

    plt.title("Amplitude of final distributions | Penalty vs Trust")
    plt.xlabel("Index")
    plt.ylabel("|f|")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
