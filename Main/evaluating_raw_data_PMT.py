import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Fast monotonicity (O(n log n)) ────────────────────────────────────────────
def monotonicity_fast(feature_k, time_k):
    n = len(feature_k)

    idx = np.argsort(feature_k)
    t_sorted = time_k[idx]

    prefix_t = np.cumsum(t_sorted)

    Num = 0.0
    for j in range(n):
        sum_t = prefix_t[j - 1] if j > 0 else 0.0
        Num += j * t_sorted[j] - sum_t

    # fast denominator
    prefix = np.cumsum(time_k)
    Den = 0.0
    for j in range(n):
        Den += j * time_k[j] - (prefix[j - 1] if j > 0 else 0.0)

    return Num / Den if Den != 0 else 0.0


# ── Main function ─────────────────────────────────────────────────────────────
def PMT_all_features(folder, time_col, feature_cols, coefficients):
    a, b, c = coefficients

    files = sorted(Path(folder).glob("*.parquet"))
    n_samples = len(files)

    # storage per feature
    Monotonicity = {col: [] for col in feature_cols}
    initial_vals = {col: [] for col in feature_cols}
    final_vals   = {col: [] for col in feature_cols}
    z_vals       = {col: [] for col in feature_cols}

    for f in tqdm(files, desc="Processing files"):
        df = pd.read_parquet(f)

        time_k = df[time_col].to_numpy(dtype=np.float64)

        for col in feature_cols:
            feature_k = df[col].to_numpy(dtype=np.float64)

            n = len(feature_k)
            if n < 2:
                continue

            # ── Monotonicity ──
            m = monotonicity_fast(feature_k, time_k)
            Monotonicity[col].append(m)

            # ── Prognosability prep ──
            initial_vals[col].append(feature_k[0])
            final_vals[col].append(feature_k[-1])

            # ── Trendability ──
            if n > 2:
                dt1   = time_k[1:] - time_k[:-1]
                dydt  = (feature_k[1:] - feature_k[:-1]) / dt1

                dt2    = time_k[2:] - time_k[:-2]
                dy2dt2 = (feature_k[2:] - 2 * feature_k[1:-1] + feature_k[:-2]) / (dt2 / 2) ** 2

                n1d = np.sum(dydt > 0)
                n2d = np.sum(dy2dt2 > 0)

                z = n1d / (n - 1) + n2d / (n - 2)
            else:
                z = 0.0

            z_vals[col].append(z)

    # ── Final aggregation per feature ─────────────────────────────────────────
    results = {}

    for col in feature_cols:
        M = np.mean(Monotonicity[col])

        init = np.array(initial_vals[col])
        final = np.array(final_vals[col])

        P = np.exp(-np.std(final) / np.mean(np.abs(init - final)))

        T = 1 - np.std(z_vals[col])

        fitness = a * M + b * P + c * T

        results[col] = {
            "Monotonicity": M,
            "Prognosability": P,
            "Trendability": T,
            "Fitness": fitness
        }

    return results

feature_cols = [
    "A", "R", "THR", "E", "NoTRAI", "TRAI",
    "CSS", "CCNT", "RMS", "PCTA", "PCTD",
    "D", "CHIT", "SS", "CNTS"
]

results = PMT_all_features(
    folder="Main/cleaned_trimmed_data",
    time_col="t",
    feature_cols=feature_cols,
    coefficients=[0.6, 0.4, 0.00001]
)

# print nicely
for col, vals in results.items():
    print(f"\n{col}")
    for k, v in vals.items():
        print(f"  {k}: {v:.4f}")
