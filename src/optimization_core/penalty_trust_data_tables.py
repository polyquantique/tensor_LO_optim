import numpy as np
import pandas as pd


def phase_align(f, g):
    phase = np.vdot(f, g)
    if phase == 0:
        return g
    return g * (phase / abs(phase))


def cosine_similarity(f, g):
    return np.real(np.vdot(f, g)) / (np.linalg.norm(f) * np.linalg.norm(g))


def compute_similarity_table(df):
    rows = []

    for seed in df["seed_type"].unique():
        pen = df[(df["method"] == "penalty") & (df["seed_type"] == seed)]
        tr = df[(df["method"] == "trust") & (df["seed_type"] == seed)]

        if len(pen) == 0 or len(tr) == 0:
            continue

        f_pen = pen.iloc[0]["f_opt"]
        f_tr = tr.iloc[0]["f_opt"]

        # phase align trust to penalty
        f_tr_aligned = phase_align(f_pen, f_tr)

        rows.append(
            {
                "seed_type": seed,
                "cosine_similarity": cosine_similarity(f_pen, f_tr_aligned),
                "euclidean_distance": np.linalg.norm(f_pen - f_tr_aligned),
            }
        )

    return pd.DataFrame(rows)


def compute_penalty_vs_trust_similarity(df):
    rows = []

    # group by seed_type
    seed_types = df["seed_type"].unique()

    for seed in seed_types:
        pen_row = df[(df["method"] == "penalty") & (df["seed_type"] == seed)]
        tr_row = df[(df["method"] == "trust") & (df["seed_type"] == seed)]

        if len(pen_row) == 0 or len(tr_row) == 0:
            continue

        f_pen = pen_row.iloc[0]["f_opt"]
        f_tr = tr_row.iloc[0]["f_opt"]

        # convert to complex
        f_pen = f_pen[: len(f_pen) // 2] + 1j * f_pen[len(f_pen) // 2 :]
        f_tr = f_tr[: len(f_tr) // 2] + 1j * f_tr[len(f_tr) // 2 :]

        # phase align trust to penalty
        f_tr_aligned = phase_align(f_pen, f_tr)

        rows.append(
            {
                "seed_type": seed,
                "cosine_similarity": cosine_similarity(f_pen, f_tr_aligned),
            }
        )

    return pd.DataFrame(rows)


def summarize_outer_results(df):
    rows = []

    for _, row in df.iterrows():
        info = row["outer_info"]
        if info is None or len(info) == 0:
            rows.append(
                {
                    "method": row["method"],
                    "seed_type": row["seed_type"],
                    "seed_index": row["seed_index"],
                    "converged": None,
                    "final_fun": None,
                    "final_violation": None,
                    "final_lambda": None,
                    "final_p": None,
                    "exit_code": None,
                }
            )
            continue

        last = info[-1]

        rows.append(
            {
                "method": row["method"],
                "seed_type": row["seed_type"],
                "seed_index": row["seed_index"],
                "converged": last.get("converged", None),
                "final_fun": last.get("fun", None),
                "final_violation": last.get("violation", None),
                "final_lambda": last.get("lambda", None),
                "final_p": last.get("p", None),  # present only for p‑norm runs
                "exit_code": last.get("exit_code", None),
            }
        )

    return pd.DataFrame(rows)
