import numpy as np
import pandas as pd
import os


# random seeds for reproducibility
def generate_seeds(K, seed_init):
    rng = np.random.default_rng(seed_init)
    return rng.integers(0, 2**32 - 1, size=K)


def save_results(df, params, seeds, filename):
    # Build full path inside results_data folder
    folder = os.path.join("results_data")
    os.makedirs(folder, exist_ok=True)  # Create folder if missing

    full_path = os.path.join(folder, filename)

    df_clean = df.copy()

    package = {
        "df": df_clean,
        "params": params,
        "seeds": seeds,
    }

    pd.to_pickle(package, full_path)
    print(f"Saved to {full_path}")


def load_results(filename):
    folder = os.path.join("results_data")
    full_path = os.path.join(folder, filename)

    package = pd.read_pickle(full_path)
    return package["df"], package["params"], package["seeds"]


def to_complex(x):
    x = x.copy()
    n = len(x) // 2
    return x[:n] + 1j * x[n:]
