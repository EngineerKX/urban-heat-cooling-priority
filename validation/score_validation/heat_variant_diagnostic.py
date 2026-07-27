"""Heat-variant diagnostic (from diagnostic_heat_variants.ipynb): investigates
whether `lst_bicubic10` showing a LOWER RMSE against held-out NEA data than
`lst_native30`/`lst_regress10` (a counterintuitive result for a "dumb"
interpolation baseline) is explained by a uniform offset (not concerning) or
heat-dependent smoothing (concerning — would mean cubic interpolation is
eating the hottest signal, a bad property for a heat-priority tool
specifically).
"""

from pathlib import Path

import pandas as pd
from scipy.stats import linregress, pearsonr


def pairwise_differences(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["diff_bicubic"] = df["lst_native30"] - df["lst_bicubic10"]
    df["diff_regress"] = df["lst_native30"] - df["lst_regress10"]
    return df


def gap_vs_heat_level(df: pd.DataFrame):
    """Correlates the (native30 - bicubic10) gap against native30's own
    value. A strong positive correlation means the gap is LARGER where
    native30 is hotter — bicubic10 is specifically flattening peak heat,
    not applying a constant offset."""
    r, p = pearsonr(df["lst_native30"], df["diff_bicubic"])
    slope, intercept, r_value, p_value, std_err = linregress(df["lst_native30"], df["diff_bicubic"])
    print(f"Pearson correlation (native30 vs diff_bicubic gap): r={r:.3f}, p={p:.4f}")
    print(f"Linear fit: diff_bicubic ≈ {slope:.3f} * lst_native30 + {intercept:.3f} (R²={r_value**2:.3f})")

    if r > 0.4 and p < 0.05:
        print("⚠️  Gap GROWS at hotter subzones — consistent with cubic interpolation smoothing/undershooting "
              "peak heat. Treat bicubic10's lower held-out RMSE with caution: flag the heat-dependent gap "
              "explicitly rather than reporting 'bicubic10 is more accurate' at face value.")
    else:
        print("✅ Gap looks closer to a uniform offset — less concerning, though still worth a one-line note "
              "on why the offset exists.")
    return r, p, slope, intercept


def plot_diagnostic(df: pd.DataFrame, slope: float, intercept: float, r: float, out_path: Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.scatter(df["lst_native30"], df["lst_bicubic10"], alpha=0.5, s=15)
    lims = [df[["lst_native30", "lst_bicubic10"]].min().min(), df[["lst_native30", "lst_bicubic10"]].max().max()]
    ax.plot(lims, lims, "r--", label="1:1 line")
    ax.set_xlabel("lst_native30 (°C)"); ax.set_ylabel("lst_bicubic10 (°C)")
    ax.set_title("bicubic10 vs native30"); ax.legend()

    ax = axes[1]
    ax.scatter(df["lst_native30"], df["lst_regress10"], alpha=0.5, s=15, color="green")
    lims2 = [df[["lst_native30", "lst_regress10"]].min().min(), df[["lst_native30", "lst_regress10"]].max().max()]
    ax.plot(lims2, lims2, "r--", label="1:1 line")
    ax.set_xlabel("lst_native30 (°C)"); ax.set_ylabel("lst_regress10 (°C)")
    ax.set_title("regress10 vs native30 (sanity check)"); ax.legend()

    ax = axes[2]
    ax.scatter(df["lst_native30"], df["diff_bicubic"], alpha=0.5, s=15, color="orange")
    ax.axhline(0, color="gray", linestyle=":")
    fit_x = [df["lst_native30"].min(), df["lst_native30"].max()]
    fit_y = [slope * x + intercept for x in fit_x]
    ax.plot(fit_x, fit_y, "r-", label=f"fit (r={r:.2f})")
    ax.set_xlabel("lst_native30 (°C)"); ax.set_ylabel("native30 - bicubic10 (°C)")
    ax.set_title("Gap vs heat level (the key diagnostic)"); ax.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
