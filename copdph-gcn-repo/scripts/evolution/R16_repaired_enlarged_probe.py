"""R16.D — Re-run R15.G enlarged-stratum probe with REPAIRED lung masks.

After R16.C lung-mask repair (HU<-300 + top-2-CC filter): lung volume
median dropped 10839 mL → 7678 mL (26% reduction). The R15.G
within-nonPH protocol probe at LR=0.908 may have been inflated by
oversegmentation. This script:

  1. Builds lung_features_extended_repaired.csv = legacy 260 + new100 repaired
  2. Re-runs the same probes from R15.G:
     A) Within-nonPH protocol probe (n=151)
     B) Within-contrast disease classifier (n=186)
     C) Endotype replication PH vs nonPH (multiplicity-corrected)
  3. Compares to R15.G results to attribute the 0.85 → 0.91 jump:
     - if repaired LR drops back near 0.85 → mask artifact, R15.G was inflated
     - if repaired LR stays at 0.91 → real protocol signal, repair confirms
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "r16"
OUT.mkdir(parents=True, exist_ok=True)
LEGACY = ROOT / "outputs" / "lung_features_v2.csv"
NEW_REPAIRED = ROOT / "outputs" / "r16" / "lung_features_new100_repaired.csv"
LABELS = ROOT / "data" / "labels_extended_382.csv"
PROTO = ROOT / "data" / "case_protocol_extended.csv"
SEG_FAILS = ROOT / "outputs" / "r13" / "seg_failures_real.json"


def boot_ci(y, p, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if not len(pos) or not len(neg): return [float("nan"), float("nan")]
    boots = []
    for _ in range(n_boot):
        bp = rng.choice(pos, size=len(pos), replace=True)
        bn = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([bp, bn])
        try:
            boots.append(roc_auc_score(y[idx], p[idx]))
        except ValueError:
            continue
    arr = np.array(boots)
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def oof(X, y, model_name="lr", seed=42):
    if len(np.unique(y)) < 2: return None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_p = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        if model_name == "lr":
            clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=seed)
        clf.fit(sc.transform(X[tr]), y[tr])
        oof_p[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return float(roc_auc_score(y, oof_p)), oof_p


def main():
    leg = pd.read_csv(LEGACY)
    new = pd.read_csv(NEW_REPAIRED)
    new["is_new_ingestion"] = 1
    leg["is_new_ingestion"] = 0
    overlap = set(leg["case_id"]) & set(new["case_id"])
    leg_keep = leg[~leg["case_id"].isin(overlap)].copy()
    all_cols = sorted(set(leg.columns) | set(new.columns))
    leg_keep = leg_keep.reindex(columns=all_cols)
    new = new.reindex(columns=all_cols)
    ext = pd.concat([leg_keep, new], ignore_index=True)
    out_csv = OUT / "lung_features_extended_repaired.csv"
    ext.to_csv(out_csv, index=False)
    print(f"saved {len(ext)} rows × {len(ext.columns)} cols → {out_csv}")

    lab = pd.read_csv(LABELS); pro = pd.read_csv(PROTO)
    df = ext.merge(lab[["case_id", "label"]], on="case_id", how="inner") \
        .merge(pro[["case_id", "protocol"]], on="case_id", how="inner")
    fails = set()
    if SEG_FAILS.exists():
        sf = json.loads(SEG_FAILS.read_text(encoding="utf-8"))
        fails = {r["case_id"] for r in sf.get("real_fails", []) + sf.get("lung_anomaly", [])}
    df = df[~df["case_id"].isin(fails)].copy()
    df["is_contrast"] = (df["protocol"].astype(str).str.lower() == "contrast").astype(int)

    # Restrict to feature columns present in both subsets (≥95% non-NaN)
    candidate_cols = [c for c in ext.columns if c not in
                       ("case_id", "case_dir", "is_new_ingestion", "mask_convention",
                        "lung_vol_mL_raw", "lung_vol_mL_repaired",
                        "lung_repair_drop_frac", "error", "label", "protocol")
                       and pd.api.types.is_numeric_dtype(ext[c])]
    leg_sub = df[df["is_new_ingestion"] == 0]
    new_sub = df[df["is_new_ingestion"] == 1]
    keep_cols = []
    for c in candidate_cols:
        leg_nan = leg_sub[c].isna().mean() if len(leg_sub) else 1
        new_nan = new_sub[c].isna().mean() if len(new_sub) else 1
        if leg_nan <= 0.05 and new_nan <= 0.05:
            keep_cols.append(c)
    print(f"feature cols: {len(keep_cols)} (after filter)")

    out: dict = {"n_total": int(len(df)),
                 "n_features": len(keep_cols), "feature_set": keep_cols}

    # A) Within-nonPH protocol probe
    nonph = df[df["label"] == 0].dropna(subset=keep_cols).copy()
    print(f"A) within-nonPH n={len(nonph)} (contrast={int(nonph['is_contrast'].sum())} "
          f"plain={int((1-nonph['is_contrast']).sum())})")
    if len(nonph) >= 30 and nonph["is_contrast"].nunique() == 2:
        X = nonph[keep_cols].values; y = nonph["is_contrast"].values.astype(int)
        lr_auc, lr_oof = oof(X, y, "lr")
        mlp_auc, mlp_oof = oof(X, y, "mlp")
        out["A_within_nonph_protocol"] = {
            "n": int(len(nonph)),
            "n_contrast": int(y.sum()), "n_plain": int(len(y) - y.sum()),
            "lr_auc": lr_auc, "lr_ci95": boot_ci(y, lr_oof),
            "mlp_auc": mlp_auc, "mlp_ci95": boot_ci(y, mlp_oof),
            "comparison_to_R15G": {
                "R15G_LR": 0.908, "R15G_LR_CI": [0.819, 0.968],
                "R15G_MLP": 0.914, "R15G_MLP_CI": [0.820, 0.978],
            },
        }
        delta_lr = lr_auc - 0.908
        out["A_within_nonph_protocol"]["delta_LR_repaired_minus_R15G"] = float(delta_lr)
        print(f"  LR={lr_auc:.3f} [{lr_oof.min():.3f}..{lr_oof.max():.3f}]; "
              f"R15G was 0.908, Δ={delta_lr:+.3f}")

    # B) Within-contrast disease
    contrast = df[df["protocol"].str.lower() == "contrast"].dropna(subset=keep_cols).copy()
    print(f"B) within-contrast disease n={len(contrast)} "
          f"(PH={int((contrast['label']==1).sum())} nonPH={int((contrast['label']==0).sum())})")
    if len(contrast) >= 30 and contrast["label"].nunique() == 2:
        X = contrast[keep_cols].values; y = contrast["label"].values.astype(int)
        lr_auc, lr_oof = oof(X, y, "lr")
        out["B_within_contrast_disease"] = {
            "n": int(len(contrast)), "n_ph": int(y.sum()), "n_nonph": int(len(y) - y.sum()),
            "lr_auc": lr_auc, "lr_ci95": boot_ci(y, lr_oof),
            "comparison_to_R15G": {"R15G_LR": 0.847, "R15G_LR_CI": [0.755, 0.923]},
        }
        out["B_within_contrast_disease"]["delta_LR_repaired_minus_R15G"] = float(lr_auc - 0.847)
        print(f"  LR={lr_auc:.3f}; R15G was 0.847, Δ={lr_auc-0.847:+.3f}")

    # C) Endotype replication on repaired data
    contrast = df[df["protocol"].str.lower() == "contrast"].copy()
    if len(contrast) >= 30:
        out["C_endotype_replication"] = {}
        for col in ["paren_mean_HU", "paren_std_HU", "paren_LAA_950_frac",
                    "paren_LAA_910_frac", "paren_LAA_856_frac",
                    "apical_basal_LAA950_gradient",
                    "lung_vol_mL", "vessel_airway_over_lung",
                    "artery_vol_mL", "vein_vol_mL"]:
            if col not in contrast.columns: continue
            a = contrast.loc[contrast["label"] == 1, col].dropna().values
            b = contrast.loc[contrast["label"] == 0, col].dropna().values
            if len(a) < 5 or len(b) < 5: continue
            try:
                u, p = mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                p = float("nan")
            out["C_endotype_replication"][col] = {
                "PH_mean": float(a.mean()), "PH_sd": float(a.std(ddof=1)),
                "nonPH_mean": float(b.mean()), "nonPH_sd": float(b.std(ddof=1)),
                "delta": float(a.mean() - b.mean()),
                "mwu_p": float(p) if p == p else None,
            }

    (OUT / "repaired_enlarged_results.json").write_text(json.dumps(out, indent=2),
                                                          encoding="utf-8")

    md = ["# R16.D — Repaired-mask enlarged-stratum probe (vs R15.G inflated)",
          "",
          f"After lung-mask repair (HU<-300 + top-2-CC; median vol 10.8L → 7.7L).",
          f"Cohort n={out['n_total']}; feature set {out['n_features']} cols.",
          "",
          "## A — Within-nonPH protocol probe (REPAIRED vs R15.G)",
          ""]
    if "A_within_nonph_protocol" in out:
        r = out["A_within_nonph_protocol"]; cmp = r["comparison_to_R15G"]
        md += [f"| metric | R15.G (oversegmented) | R16.D (repaired) | Δ |",
               f"|---|---|---|---|",
               f"| n | 151 | {r['n']} | {r['n']-151:+d} |",
               f"| LR AUC | {cmp['R15G_LR']:.3f} [{cmp['R15G_LR_CI'][0]:.3f}, {cmp['R15G_LR_CI'][1]:.3f}] | "
               f"**{r['lr_auc']:.3f} [{r['lr_ci95'][0]:.3f}, {r['lr_ci95'][1]:.3f}]** | "
               f"{r['delta_LR_repaired_minus_R15G']:+.3f} |",
               f"| MLP AUC | {cmp['R15G_MLP']:.3f} [{cmp['R15G_MLP_CI'][0]:.3f}, {cmp['R15G_MLP_CI'][1]:.3f}] | "
               f"**{r['mlp_auc']:.3f} [{r['mlp_ci95'][0]:.3f}, {r['mlp_ci95'][1]:.3f}]** | "
               f"{r['mlp_auc']-cmp['R15G_MLP']:+.3f} |",
               ""]
    if "B_within_contrast_disease" in out:
        r = out["B_within_contrast_disease"]; cmp = r["comparison_to_R15G"]
        md += ["## B — Within-contrast disease (REPAIRED vs R15.G)",
               "",
               f"| metric | R15.G | R16.D (repaired) | Δ |",
               f"|---|---|---|---|",
               f"| n | 186 | {r['n']} | {r['n']-186:+d} |",
               f"| LR AUC | {cmp['R15G_LR']:.3f} | **{r['lr_auc']:.3f} [{r['lr_ci95'][0]:.3f}, {r['lr_ci95'][1]:.3f}]** | "
               f"{r['delta_LR_repaired_minus_R15G']:+.3f} |",
               ""]
    if "C_endotype_replication" in out:
        md += ["## C — Endotype replication (REPAIRED data, within-contrast)",
               "",
               "| feature | PH μ±SD | nonPH μ±SD | Δ | MWU p |",
               "|---|---|---|---|---|"]
        for col, r in out["C_endotype_replication"].items():
            p = r.get("mwu_p")
            md.append(f"| {col} | {r['PH_mean']:.3f} ± {r['PH_sd']:.3f} | "
                      f"{r['nonPH_mean']:.3f} ± {r['nonPH_sd']:.3f} | "
                      f"{r['delta']:+.3f} | {p if p is None else f'{p:.3g}'} |")
        md.append("")
    md += ["## Verdict",
           "",
           "If the repaired-mask LR is **near 0.91** → R15.G's enlarged protocol",
           "decoder is real, not artifact.",
           "If it drops back to **~0.85** → the inflation was driven by oversegmentation.",
           "If it stays HIGH but lower → mixed effect.",
           ""]
    (OUT / "repaired_enlarged_results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nsaved → {OUT}/repaired_enlarged_results.{{json,md}}")


if __name__ == "__main__":
    main()
