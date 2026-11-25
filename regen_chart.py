"""
regen_chart.py
--------------
Regenerates the feature importance chart using the new field-notebook palette.
Does NOT touch any ML code or train.py — just re-plots from the saved pipeline.

Run with: python regen_chart.py
"""
import os, sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_PATH, CHART_PATH

# ---- Field-notebook palette ----
BG        = "#F6F3EC"
GREEN     = "#2F4F3E"
CLAY      = "#B5654A"
MUTED     = "#6B6560"
TINT      = "#EAE7DE"
TEXT      = "#222222"

def regen_chart():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model not found at {MODEL_PATH}. Run src/train.py first.")
        sys.exit(1)

    pipeline = joblib.load(MODEL_PATH)
    preprocessor = pipeline.named_steps["preprocessor"]
    model         = pipeline.named_steps["model"]

    cat_cols = preprocessor.transformers_[1][2]
    num_cols = preprocessor.transformers_[0][2]
    ohe      = preprocessor.named_transformers_["cat"]
    ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
    all_names = num_cols + ohe_names

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        print("No importances available.")
        sys.exit(1)

    fi = pd.DataFrame({"feature": all_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=True).tail(15)

    def prettify(name):
        if "_" in name:
            parts = name.split("_", 1)
            base  = parts[0].replace("_", " ").title()
            value = parts[1].replace("_", " ")
            return f"{base}: {value}"
        return name.replace("_", " ").title()

    fi["label"] = fi["feature"].apply(prettify)

    # ---- Plot ----
    rcParams.update({"font.family": "DejaVu Sans"})

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    n = len(fi)
    colors = [CLAY if i == n - 1 else GREEN for i in range(n)]

    bars = ax.barh(fi["label"], fi["importance"], color=colors,
                   edgecolor="none", height=0.55)

    # Direct value labels on bars
    for bar, val in zip(bars, fi["importance"]):
        ax.text(bar.get_width() + fi["importance"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                fontsize=7.5, color=MUTED)

    ax.set_xlabel("Gain (feature importance)", color=MUTED, fontsize=9, labelpad=8)
    ax.set_title("Top 15 features by importance — XGBoost",
                 color=GREEN, fontsize=12, fontweight="normal",
                 pad=14, loc="left")

    # Minimal style
    ax.tick_params(axis="y", colors=TEXT, labelsize=8.5)
    ax.tick_params(axis="x", colors=MUTED, labelsize=7.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(TINT)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.set_xlim(0, fi["importance"].max() * 1.22)

    # Subtle grid on x only
    ax.xaxis.grid(True, color=TINT, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)

    fig.text(0.01, -0.02,
             "Gain = average improvement in prediction accuracy per split on this feature.",
             fontsize=7, color=MUTED, ha="left", va="top")

    plt.tight_layout(pad=1.5)
    os.makedirs(os.path.dirname(CHART_PATH), exist_ok=True)
    plt.savefig(CHART_PATH, dpi=160, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print(f"Chart saved to: {CHART_PATH}")

if __name__ == "__main__":
    regen_chart()
