"""
analyze.py
Statistical analysis and plotting over results/summary.csv, which is
accumulated by evaluate.py across all models and conditions.

Produces:
    graphs/accuracy_vs_severity_<corruption>.png  — one plot per corruption type
    results/mce_summary.csv                        — mean Corruption Error per model

Computes:
    - Friedman test: do the 4 models differ in accuracy across corrupted
      conditions (a non-parametric analogue of repeated-measures ANOVA)?
    - Nemenyi post-hoc test (if the Friedman test is significant) — which
      pairs of models differ significantly;
    - mCE (mean Corruption Error), following the protocol of
      Hendrycks & Dietterich (2019), normalized against ResNet-50 as the
      baseline model.

Usage:
    python analyze.py --summary results/summary.csv --out graphs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare

try:
    import scikit_posthocs as sp
    HAS_POSTHOC = True
except ImportError:
    HAS_POSTHOC = False


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce")
    return df


def plot_accuracy_vs_severity(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    degraded = df[df["corruption"] != "clean"].dropna(subset=["severity"])

    for corruption, sub in degraded.groupby("corruption"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for model_name, model_sub in sub.groupby("model"):
            model_sub = model_sub.sort_values("severity")
            ax.plot(model_sub["severity"], model_sub["accuracy"], marker="o", label=model_name)
        ax.set_xlabel("Severity level")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy vs severity — {corruption}")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"accuracy_vs_severity_{corruption}.png", dpi=150)
        plt.close(fig)
        print(f"Saved plot: {out_dir / f'accuracy_vs_severity_{corruption}.png'}")


def compute_mce(df: pd.DataFrame, baseline_model: str = "resnet50") -> pd.DataFrame:
    """mCE following Hendrycks & Dietterich (2019): for each (model, corruption)
    pair, average the error rate over severity 1..5, then normalize against
    the baseline model.
    """
    degraded = df[df["corruption"] != "clean"].dropna(subset=["severity"]).copy()
    degraded["error"] = 1 - degraded["accuracy"]

    ce = degraded.groupby(["model", "corruption"])["error"].mean().reset_index()
    ce = ce.rename(columns={"error": "mean_error"})

    baseline = ce[ce["model"] == baseline_model].set_index("corruption")["mean_error"]
    if baseline.empty:
        print(f"WARNING: baseline model '{baseline_model}' not found in the data; mCE will not be normalized.")
        ce["CE"] = ce["mean_error"]
    else:
        ce["CE"] = ce.apply(
            lambda row: row["mean_error"] / baseline.get(row["corruption"], float("nan")),
            axis=1,
        )

    mce = ce.groupby("model")["CE"].mean().reset_index().rename(columns={"CE": "mCE"})
    return mce.sort_values("mCE")


def friedman_test(df: pd.DataFrame) -> None:
    """Friedman test: compares models by accuracy across identical conditions
    (corruption x severity), where each condition is treated as a "block".
    """
    degraded = df[df["corruption"] != "clean"].dropna(subset=["severity"]).copy()
    pivot = degraded.pivot_table(
        index=["corruption", "severity"], columns="model", values="accuracy",
    )
    pivot = pivot.dropna()  # only keep conditions where every model has a result

    if pivot.shape[0] < 3 or pivot.shape[1] < 3:
        print("Not enough data for a Friedman test (need >=3 conditions and >=3 models).")
        return

    stat, p_value = friedmanchisquare(*[pivot[col] for col in pivot.columns])
    print(f"\nFriedman test: statistic={stat:.4f}, p-value={p_value:.6f}")
    if p_value < 0.05:
        print("=> Significant differences between models detected (p < 0.05).")
        if HAS_POSTHOC:
            posthoc = sp.posthoc_nemenyi_friedman(pivot.values)
            posthoc.columns = pivot.columns
            posthoc.index = pivot.columns
            print("\nPost-hoc Nemenyi test (pairwise p-values between models):")
            print(posthoc.round(4))
        else:
            print("Install scikit-posthocs for the post-hoc analysis: pip install scikit-posthocs")
    else:
        print("=> No statistically significant difference between models detected (p >= 0.05).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statistical analysis of robustness-evaluation results")
    parser.add_argument("--summary", type=str, default="results/summary.csv")
    parser.add_argument("--out", type=str, default="graphs")
    parser.add_argument("--baseline_model", type=str, default="resnet50")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = load_summary(Path(args.summary))

    print(f"Loaded {len(df)} rows from {args.summary}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Corruption types: {sorted(df['corruption'].unique())}")

    plot_accuracy_vs_severity(df, Path(args.out))

    mce = compute_mce(df, baseline_model=args.baseline_model)
    print(f"\nMean Corruption Error (mCE), normalized against {args.baseline_model}:")
    print(mce.to_string(index=False))
    mce.to_csv(Path(args.out).parent / "results" / "mce_summary.csv", index=False)

    friedman_test(df)
