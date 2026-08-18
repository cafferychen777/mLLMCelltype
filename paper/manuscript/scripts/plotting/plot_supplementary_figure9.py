#!/usr/bin/env python3
"""
Supplementary Figure 9: Per-dataset annotation accuracy across five methods.

Panel a — horizontal dot plot showing per-dataset accuracy for mLLMCelltype,
          GPTCelltype, SingleR, scType, and naive baseline.
Panel b — mean accuracy summary across methods.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path("manuscript")
SUPPL_DATA = BASE_DIR / "Supplementary_Data_1.xlsx"
NAIVE_DATA = Path("results/benchmark/"
                  "reference_comparison/2_evaluation/naive_baseline_comparison.csv")
SINGLER_SCTYPE_DATA = Path("results/benchmark/"
                           "singler_sctype_batch/"
                           "singler_sctype_co_matched_results.csv")
OUTPUT_DIR = BASE_DIR / "figures"

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
# mLLMCelltype & GPTCelltype (from Supplementary Data 1)
raw = pd.read_excel(SUPPL_DATA, sheet_name="Figure 4a Data", header=1)
raw = raw.dropna(how="all").reset_index(drop=True).iloc[1:].reset_index(drop=True)
raw.columns = [
    "Dataset", "Total_Entries",
    "mLLM_Full", "mLLM_Partial", "mLLM_Acc",
    "GPT_Full", "GPT_Partial", "GPT_Acc",
]
for col in raw.columns[1:]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")
raw = raw.dropna(subset=["Dataset"]).copy()


def normalize(name):
    """Canonical key for cross-file dataset matching."""
    n = str(name).strip().replace("_", " ").lower()
    n = n.replace("crc", "colon")
    n = n.replace("(literature (gtex))", "(literature)(gtex)")
    return " ".join(n.split())


# Naive baseline
naive_df = pd.read_csv(NAIVE_DATA)
naive_dict = {normalize(r["Dataset"]): r["Naive_Accuracy"]
              for _, r in naive_df.iterrows()}
raw["Naive_Acc"] = raw["Dataset"].apply(
    lambda d: naive_dict.get(normalize(d), np.nan))

# SingleR & scType
ss_df = pd.read_csv(SINGLER_SCTYPE_DATA)
singler_agg, sctype_agg = {}, {}
for _, r in ss_df.iterrows():
    key = normalize(r["dataset"])
    if "esophagus mucosa" in key or "esophagus muscularis" in key:
        key = (key.replace("esophagus mucosa", "esophagus")
                  .replace("esophagus muscularis", "esophagus"))
    singler_agg.setdefault(key, []).append(r.get("singler_accuracy", np.nan))
    sctype_agg.setdefault(key, []).append(r.get("sctype_accuracy", np.nan))
singler_dict = {k: np.nanmean(v) for k, v in singler_agg.items()}
sctype_dict = {k: np.nanmean(v) for k, v in sctype_agg.items()}
raw["SingleR_Acc"] = raw["Dataset"].apply(
    lambda d: singler_dict.get(normalize(d), np.nan))
raw["scType_Acc"] = raw["Dataset"].apply(
    lambda d: sctype_dict.get(normalize(d), np.nan))

# ---------------------------------------------------------------------------
# 2. Categorise, sort, shorten names
# ---------------------------------------------------------------------------
def dataset_category(name):
    if "(Azimuth)" in name:
        return "Azimuth"
    if "(TS)" in name:
        return "Tabula Sapiens"
    if "(GTEx)" in name and "Literature" not in name:
        return "GTEx (DE)"
    if "(Literature)" in name or "Literature" in name:
        return "GTEx (Literature)"
    if "(Cancer)" in name:
        return "Cancer"
    return "Other"


CAT_ORDER = ["Azimuth", "Tabula Sapiens", "GTEx (DE)",
             "GTEx (Literature)", "Cancer", "Other"]

raw["Category"] = raw["Dataset"].apply(dataset_category)
raw["cat_rank"] = raw["Category"].map({c: i for i, c in enumerate(CAT_ORDER)})
raw = (raw.sort_values(["cat_rank", "mLLM_Acc"], ascending=[True, False])
          .reset_index(drop=True))

SHORTEN_MAP = [
    ("(Literature)(GTEx)", "(Lit.)(GTEx)"),
    ("(Literature (GTEx))", "(Lit.)(GTEx)"),
    ("Skeletal Muscle", "Sk. Muscle"),
    ("Fetal Development", "Fetal Dev."),
    ("Small Intestine", "Sm. Intestine"),
    ("Large Intestine", "Lg. Intestine"),
    ("Salivary Gland", "Sal. Gland"),
    ("Bone Marrow", "BM"),
]


def shorten(name):
    for old, new in SHORTEN_MAP:
        name = name.replace(old, new)
    # Ensure first letter is capitalised
    return name[0].upper() + name[1:] if name else name


raw["Label"] = raw["Dataset"].apply(shorten)

# ---------------------------------------------------------------------------
# 3. Compute y positions with inter-category gaps
# ---------------------------------------------------------------------------
CAT_GAP = 1.5

y_positions = []
cat_spans = []          # (y_start, y_end, category_name)
y = 0
prev_cat = None
cat_start = 0

for _, row in raw.iterrows():
    cat = row["Category"]
    if cat != prev_cat:
        if prev_cat is not None:
            cat_spans.append((cat_start, y - 1, prev_cat))
            y += CAT_GAP
        cat_start = y
        prev_cat = cat
    y_positions.append(y)
    y += 1

cat_spans.append((cat_start, y - 1, prev_cat))
raw["y"] = y_positions
y_max = max(y_positions)

# ---------------------------------------------------------------------------
# 4. Style constants
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# (label, column, colour, marker, size, zorder)
METHODS = [
    ("mLLMCelltype", "mLLM_Acc",    "#2f5597", "o", 36, 5),
    ("GPTCelltype",  "GPT_Acc",     "#8faadc", "o", 24, 4),
    ("SingleR",      "SingleR_Acc", "#548235", "D", 20, 3),
    ("scType",       "scType_Acc",  "#ed7d31", "^", 22, 3),
    ("Naive",        "Naive_Acc",   "#bfbfbf", "s", 12, 2),
]

# ---------------------------------------------------------------------------
# 5. Figure layout
# ---------------------------------------------------------------------------
ROW_HEIGHT = 0.145          # inches per y-unit in the dot plot
PANEL_B_HEIGHT = 1.4        # inches
VSPACE = 0.85               # inches between panels

panel_a_h = (y_max + 1) * ROW_HEIGHT
fig_w = 7.2
fig_h = panel_a_h + PANEL_B_HEIGHT + VSPACE + 1.0   # +margins

fig = plt.figure(figsize=(fig_w, fig_h))
gs = fig.add_gridspec(
    2, 1,
    height_ratios=[panel_a_h, PANEL_B_HEIGHT],
    hspace=VSPACE / (panel_a_h + PANEL_B_HEIGHT),
)

# ---------------------------------------------------------------------------
# 6. Panel a — horizontal dot plot
# ---------------------------------------------------------------------------
ax = fig.add_subplot(gs[0])

# Category shading (alternating bands)
for idx, (ys, ye, cat) in enumerate(cat_spans):
    if idx % 2 == 0:
        ax.axhspan(ys - 0.5, ye + 0.5, color="#f5f5f5", zorder=0)
    # Category label at right margin
    mid = (ys + ye) / 2
    ax.text(1.02, mid, cat,
            transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=6, fontweight="bold",
            color="#555555", clip_on=False)

# Thin horizontal guide per dataset
for yp in y_positions:
    ax.axhline(yp, color="#ebebeb", linewidth=0.4, zorder=1)

# Range connector (min→max) per dataset
for _, row in raw.iterrows():
    vals = [row[col] for _, col, *_ in METHODS if pd.notna(row[col])]
    if len(vals) >= 2:
        ax.plot([min(vals), max(vals)], [row["y"], row["y"]],
                color="#dcdcdc", linewidth=1.0, zorder=2,
                solid_capstyle="round")

# Dots
for label, col, color, marker, size, zorder in METHODS:
    mask = raw[col].notna()
    ax.scatter(raw.loc[mask, col], raw.loc[mask, "y"],
               s=size, c=color, marker=marker, zorder=zorder,
               edgecolors="white", linewidths=0.3, label=label)

# Axes formatting
ax.set_xlim(-0.02, 1.05)
ax.set_ylim(y_max + 0.8, -0.8)          # top-to-bottom
ax.set_yticks(y_positions)
ax.set_yticklabels(raw["Label"], fontsize=5.5)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Annotation accuracy")
ax.xaxis.grid(True, linestyle="--", alpha=0.2, color="gray", zorder=0)
ax.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.set_title("a", fontsize=10, fontweight="bold", loc="left", pad=6)

# Legend (lower-right of the dot plot)
leg = ax.legend(
    loc="lower right", frameon=True, framealpha=0.95,
    edgecolor="#cccccc", fontsize=6, handlelength=1.2,
    borderpad=0.5, labelspacing=0.35, scatterpoints=1,
    handletextpad=0.4,
)
leg.get_frame().set_linewidth(0.5)

# ---------------------------------------------------------------------------
# 7. Panel b — summary bar chart
# ---------------------------------------------------------------------------
ax_b = fig.add_subplot(gs[1])

n_all = len(raw)
n_sr = raw["SingleR_Acc"].notna().sum()
n_sc = raw["scType_Acc"].notna().sum()

labels_b = [
    f"mLLMCelltype (n={n_all})",
    f"GPTCelltype (n={n_all})",
    f"SingleR (n={n_sr})",
    f"scType (n={n_sc})",
    f"Naive (n={n_all})",
]
means_b = [
    raw["mLLM_Acc"].mean(),
    raw["GPT_Acc"].mean(),
    raw["SingleR_Acc"].dropna().mean(),
    raw["scType_Acc"].dropna().mean(),
    raw["Naive_Acc"].mean(),
]
colors_b = [m[2] for m in METHODS]

by = np.arange(len(labels_b))
bars = ax_b.barh(by, means_b, color=colors_b, edgecolor="white",
                 linewidth=0.3, height=0.55, zorder=3)

for score, bar in zip(means_b, bars):
    ax_b.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
              f"{score:.1%}", va="center", ha="left",
              fontsize=6.5, color="#333333")

ax_b.set_yticks(by)
ax_b.set_yticklabels(labels_b, fontsize=7)
ax_b.set_xlim(0, 0.95)
ax_b.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax_b.set_xlabel("Mean annotation accuracy")
ax_b.invert_yaxis()
ax_b.set_title("b", fontsize=10, fontweight="bold", loc="left", pad=6)
for sp in ["top", "right"]:
    ax_b.spines[sp].set_visible(False)
ax_b.xaxis.grid(True, linestyle="--", alpha=0.2, color="gray", zorder=0)
ax_b.set_axisbelow(True)

# ---------------------------------------------------------------------------
# 8. Save
# ---------------------------------------------------------------------------
out_pdf = OUTPUT_DIR / "supplementary_figure9.pdf"
out_png = OUTPUT_DIR / "supplementary_figure9.png"
fig.savefig(out_pdf, bbox_inches="tight", format="pdf")
fig.savefig(out_png, bbox_inches="tight", format="png", dpi=300)
plt.close(fig)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
