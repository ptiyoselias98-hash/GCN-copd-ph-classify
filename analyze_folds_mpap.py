"""Per-fold mPAP analysis: link case_ids in each fold's val set back to mPAP
from the patient excel, and check whether folds 4/5 contain a heavier load of
borderline (mPAP ≈ 20) cases — the COPD-PH gold-standard threshold.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

EXCEL = r"C:\Users\cheng\Desktop\图卷积-肺小血管演化规律探索\copd-ph患者113例0331.xlsx"
SPLITS_DIR = Path("temp_splits")
OUT_DIR = Path(r"C:\Users\cheng\Desktop\图卷积-肺小血管演化规律探索\outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PER_FOLD_AUC = {1: 0.96, 2: 1.00, 3: 1.00, 4: 0.78, 5: 0.78}


def case_to_pinyin(case_id: str) -> str:
    parts = case_id.split("_")
    if len(parts) >= 3 and parts[0] in ("nonph", "ph"):
        return parts[1].lower()
    return case_id.lower()


def main() -> int:
    df = pd.read_excel(EXCEL, sheet_name="Sheet1")
    df["_pinyin"] = df["ct文件名"].astype(str).str.split("_").str[0].str.lower()
    pinyin_to_mpap = dict(zip(df["_pinyin"], df["mPAP"]))
    pinyin_to_label = dict(zip(df["_pinyin"], df["PH"].map({"是": 1, "/": 0})))

    rows = []
    for k in range(1, 6):
        val_ids = [c.strip() for c in (SPLITS_DIR / f"fold{k}_val.txt").read_text().splitlines() if c.strip()]
        for cid in val_ids:
            py = case_to_pinyin(cid)
            mpap = pinyin_to_mpap.get(py)
            label_csv = 1 if cid.startswith("ph_") else 0
            rows.append({
                "fold": k,
                "case_id": cid,
                "pinyin": py,
                "label_from_id": label_csv,
                "label_from_xlsx": pinyin_to_label.get(py),
                "mPAP": mpap,
                "borderline_18_22": (mpap is not None) and (18 <= mpap <= 22),
            })
    rec = pd.DataFrame(rows)
    matched = rec["mPAP"].notna().sum()
    print(f"matched mPAP: {matched}/{len(rec)} val cases")

    # ----- per-fold summary -----
    summary = []
    for k in range(1, 6):
        sub = rec[rec["fold"] == k].dropna(subset=["mPAP"])
        ph = sub[sub["label_from_id"] == 1]["mPAP"]
        non = sub[sub["label_from_id"] == 0]["mPAP"]
        n_border = sub["borderline_18_22"].sum()
        # closest to threshold gap (smaller = harder)
        ph_min = ph.min() if len(ph) else float("nan")
        non_max = non.max() if len(non) else float("nan")
        gap = (ph_min - non_max) if (len(ph) and len(non)) else float("nan")
        summary.append({
            "fold": k,
            "val_AUC": PER_FOLD_AUC[k],
            "n_val": len(sub),
            "n_PH": len(ph),
            "n_nonPH": len(non),
            "PH_mPAP_min": round(float(ph_min), 1) if len(ph) else None,
            "PH_mPAP_median": round(float(ph.median()), 1) if len(ph) else None,
            "nonPH_mPAP_max": round(float(non_max), 1) if len(non) else None,
            "nonPH_mPAP_median": round(float(non.median()), 1) if len(non) else None,
            "n_borderline_18_22": int(n_border),
            "PHmin_minus_nonPHmax": round(float(gap), 1) if not np.isnan(gap) else None,
        })
    sum_df = pd.DataFrame(summary)
    print("\n=== per-fold val-set mPAP summary ===")
    print(sum_df.to_string(index=False))

    # write excel report
    xlsx_path = OUT_DIR / "fold_mpap_audit.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        sum_df.to_excel(w, sheet_name="per_fold_summary", index=False)
        rec.sort_values(["fold", "label_from_id", "mPAP"]).to_excel(
            w, sheet_name="per_case", index=False)
    print(f"\nwrote {xlsx_path}")

    # ----- plot -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # left: per-case scatter, colored by label, x=fold
    ax = axes[0]
    rng = np.random.default_rng(0)
    for k in range(1, 6):
        sub = rec[rec["fold"] == k].dropna(subset=["mPAP"])
        for lab, color, marker in [(0, "#1f77b4", "o"), (1, "#d62728", "^")]:
            s = sub[sub["label_from_id"] == lab]
            jitter = rng.normal(0, 0.08, size=len(s))
            ax.scatter(np.full(len(s), k) + jitter, s["mPAP"],
                       c=color, marker=marker, s=42, alpha=0.75,
                       edgecolors="white", linewidths=0.5,
                       label=("non-PH" if lab == 0 else "PH") if k == 1 else None)
    ax.axhline(20, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.text(0.55, 20.4, "mPAP = 20 (gold-standard threshold)",
            fontsize=9, color="gray")
    # AUC annotation per fold
    for k in range(1, 6):
        ax.text(k, 66, f"AUC\n{PER_FOLD_AUC[k]:.2f}", ha="center",
                fontsize=9, color="#444")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Fold (val set)")
    ax.set_ylabel("mPAP (mmHg)")
    ax.set_title("mPAP distribution per CV fold (val set)", fontweight="bold")
    ax.set_ylim(5, 75)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    # right: borderline-count + gap
    ax = axes[1]
    folds = sum_df["fold"].values
    n_border = sum_df["n_borderline_18_22"].values
    gaps = sum_df["PHmin_minus_nonPHmax"].fillna(0).values

    bars = ax.bar(folds - 0.18, n_border, width=0.36, color="#FFA94D",
                  label="# borderline cases (mPAP ∈ [18,22])")
    ax.set_xlabel("Fold")
    ax.set_ylabel("# borderline cases", color="#B86A00")
    ax.tick_params(axis="y", labelcolor="#B86A00")
    ax.set_xticks(folds)
    ax.set_ylim(0, max(n_border.max() + 2, 5))
    for x, n in zip(folds, n_border):
        ax.text(x - 0.18, n + 0.1, str(int(n)), ha="center",
                fontsize=9, color="#B86A00")

    ax2 = ax.twinx()
    ax2.plot(folds, gaps, marker="o", color="#2E7D32", linewidth=2,
             label="PHmin − nonPHmax gap")
    ax2.axhline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.set_ylabel("PH-min minus nonPH-max  (mmHg)", color="#2E7D32")
    ax2.tick_params(axis="y", labelcolor="#2E7D32")
    for x, g in zip(folds, gaps):
        ax2.text(x + 0.05, g + 0.4, f"{g:+.1f}", color="#2E7D32",
                 fontsize=9)

    # AUC annotation
    for x in folds:
        ax.text(x, ax.get_ylim()[1] * 0.95, f"AUC {PER_FOLD_AUC[x]:.2f}",
                ha="center", fontsize=8, color="#444")

    ax.set_title("Borderline load + class-separation gap vs val AUC",
                 fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)

    fig.suptitle("Per-fold mPAP audit — why do folds 4 & 5 underperform?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    png = OUT_DIR / "fold_mpap_audit.png"
    plt.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
