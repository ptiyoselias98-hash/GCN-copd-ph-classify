"""R15.G — Enlarged-stratum analyses on the 360-case extended panel.

Three analyses, all using ONLY lung_features_extended.csv (no v2 graph
yet for new 100):

  A) Within-nonPH protocol probe (LR + MLP) at n=190 (vs old n=80):
     does the protocol leak survive in the bigger stratum?

  B) Within-contrast disease classifier on lung-features at n=187
     (no protocol confound), reported with 95% CI. Compare to R14 n=184.

  C) Apical-basal LAA-950 gradient + paren_mean_HU contrast PH vs nonPH:
     does the dense-lung PH signature replicate at scale?

Outputs: outputs/r15/enlarged_lung_results.{json,md}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "r15"
LUNG = OUT / "lung_features_extended.csv"
LABELS = ROOT / "data" / "labels_extended_382.csv"
PROTO = ROOT / "data" / "case_protocol_extended.csv"
SEG_FAILS = ROOT / "outputs" / "r13" / "seg_failures_real.json"


def boot_ci(y, p, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if not len(pos) or not len(neg):
        return [float("nan"), float("nan")]
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
            clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000,
                                  random_state=seed)
        clf.fit(sc.transform(X[tr]), y[tr])
        oof_p[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return float(roc_auc_score(y, oof_p)), oof_p


def main():
    lung = pd.read_csv(LUNG)
    lab = pd.read_csv(LABELS); pro = pd.read_csv(PROTO)
    df = lung.merge(lab[["case_id", "label"]], on="case_id", how="inner") \
        .merge(pro[["case_id", "protocol"]], on="case_id", how="inner")
    fails = set()
    if SEG_FAILS.exists():
        sf = json.loads(SEG_FAILS.read_text(encoding="utf-8"))
        fails = {r["case_id"] for r in sf.get("real_fails", []) + sf.get("lung_anomaly", [])}
    df = df[~df["case_id"].isin(fails)].copy()
    print(f"after seg-failure exclusion: {len(df)} cases")

    lung_cols = [c for c in lung.columns if c not in ("case_id", "case_dir",
                                                          "is_new_ingestion",
                                                          "mask_convention",
                                                          "error", "label", "protocol")
                  and pd.api.types.is_numeric_dtype(lung[c])]
    df["is_contrast"] = (df["protocol"].astype(str).str.lower() == "contrast").astype(int)
    # Restrict to columns that have ≥95% non-NaN in BOTH legacy and new ingestion subsets
    leg_sub = df[df["is_new_ingestion"] == 0]
    new_sub = df[df["is_new_ingestion"] == 1]
    keep_cols = []
    for c in lung_cols:
        leg_nan = leg_sub[c].isna().mean() if len(leg_sub) > 0 else 1.0
        new_nan = new_sub[c].isna().mean() if len(new_sub) > 0 else 1.0
        if leg_nan <= 0.05 and new_nan <= 0.05:
            keep_cols.append(c)
    print(f"feature cols (raw {len(lung_cols)} → common {len(keep_cols)})")
    lung_cols = keep_cols

    out = {"n_total": int(len(df)),
           "feature_set": lung_cols, "n_features": len(lung_cols)}

    # A) Within-nonPH protocol probe (n=190 vs old 80)
    nonph = df[df["label"] == 0].dropna(subset=lung_cols).copy()
    print(f"A) within-nonPH stratum: {len(nonph)} cases "
          f"(contrast={int(nonph['is_contrast'].sum())}, plain={int((1-nonph['is_contrast']).sum())})")
    if len(nonph) >= 30 and nonph["is_contrast"].nunique() == 2:
        X = nonph[lung_cols].values
        y = nonph["is_contrast"].values.astype(int)
        lr_auc, lr_oof = oof(X, y, "lr")
        mlp_auc, mlp_oof = oof(X, y, "mlp")
        out["A_within_nonph_protocol"] = {
            "n": int(len(nonph)),
            "n_contrast": int(y.sum()), "n_plain": int(len(y) - y.sum()),
            "lr_auc": lr_auc, "lr_ci95": boot_ci(y, lr_oof),
            "mlp_auc": mlp_auc, "mlp_ci95": boot_ci(y, mlp_oof),
            "comparison_to_R12": "R12 reported n=80 LR=0.853 [0.722, 0.942]; new n is "
                                 + f"{len(nonph)}",
        }

    # B) Within-contrast disease classifier
    contrast = df[df["protocol"].str.lower() == "contrast"].dropna(subset=lung_cols).copy()
    print(f"B) within-contrast disease: {len(contrast)} cases "
          f"(PH={int((contrast['label']==1).sum())} nonPH={int((contrast['label']==0).sum())})")
    if len(contrast) >= 30 and contrast["label"].nunique() == 2:
        X = contrast[lung_cols].values
        y = contrast["label"].values.astype(int)
        lr_auc, lr_oof = oof(X, y, "lr")
        out["B_within_contrast_disease"] = {
            "n": int(len(contrast)),
            "n_ph": int(y.sum()), "n_nonph": int(len(y) - y.sum()),
            "lr_auc": lr_auc, "lr_ci95": boot_ci(y, lr_oof),
            "comparison_to_R14": "R14 reported lung-only AUC 0.844 [0.754, 0.917] at n=184",
        }

    # C) Apical-basal gradient + paren_mean_HU PH vs nonPH (within contrast)
    if len(contrast) >= 30:
        ph = contrast[contrast["label"] == 1]
        nph = contrast[contrast["label"] == 0]
        out["C_endotype_replication"] = {}
        for col in ["paren_mean_HU", "apical_basal_LAA950_gradient",
                    "paren_LAA_950_frac", "vessel_airway_over_lung",
                    "lung_vol_mL", "artery_vol_mL", "vein_vol_mL"]:
            if col not in contrast.columns: continue
            a = ph[col].dropna().values; b = nph[col].dropna().values
            if len(a) < 5 or len(b) < 5: continue
            try:
                u, p = mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                u, p = float("nan"), float("nan")
            out["C_endotype_replication"][col] = {
                "PH_mean": float(a.mean()), "PH_sd": float(a.std(ddof=1)),
                "nonPH_mean": float(b.mean()), "nonPH_sd": float(b.std(ddof=1)),
                "delta": float(a.mean() - b.mean()),
                "mwu_p": float(p) if p == p else None,
            }

    # Compare new vs legacy ingestion for the 100 new cases
    new_only = df[df["is_new_ingestion"] == 1]
    if len(new_only) > 0:
        out["new_ingestion_check"] = {
            "n_new": int(len(new_only)),
            "all_label0": bool((new_only["label"] == 0).all()),
            "all_plain": bool((new_only["protocol"].str.lower() == "plain_scan").all()),
            "paren_mean_HU_mean": float(new_only["paren_mean_HU"].dropna().mean()),
            "paren_LAA_950_frac_mean": float(new_only["paren_LAA_950_frac"].dropna().mean()),
        }

    (OUT / "enlarged_lung_results.json").write_text(json.dumps(out, indent=2),
                                                       encoding="utf-8")

    md = ["# R15.G — Enlarged-stratum lung-feature analyses (n=360 ingested cohort)",
          "",
          f"After seg-failure exclusion: {out['n_total']} cases. "
          f"Feature set: {out['n_features']} lung-parenchyma features.",
          ""]
    if "A_within_nonph_protocol" in out:
        r = out["A_within_nonph_protocol"]
        md += ["## A — Within-nonPH protocol probe (lung features, n=" + str(r["n"]) + ")",
               "",
               f"Contrast: {r['n_contrast']}, Plain-scan: {r['n_plain']}",
               "",
               f"- **LR AUC = {r['lr_auc']:.3f} [{r['lr_ci95'][0]:.3f}, {r['lr_ci95'][1]:.3f}]**",
               f"- **MLP AUC = {r['mlp_auc']:.3f} [{r['mlp_ci95'][0]:.3f}, {r['mlp_ci95'][1]:.3f}]**",
               "",
               f"_R12 baseline (n=80): LR=0.853 [0.722, 0.942]_",
               ""]
    if "B_within_contrast_disease" in out:
        r = out["B_within_contrast_disease"]
        md += ["## B — Within-contrast disease classifier (lung-only, n=" + str(r["n"]) + ")",
               "",
               f"PH: {r['n_ph']}, nonPH: {r['n_nonph']}",
               "",
               f"- **LR AUC = {r['lr_auc']:.3f} [{r['lr_ci95'][0]:.3f}, {r['lr_ci95'][1]:.3f}]**",
               "",
               f"_R14 baseline (n=184): LR=0.844 [0.754, 0.917]_",
               ""]
    if "C_endotype_replication" in out:
        md += ["## C — Endotype replication (within-contrast PH vs nonPH)",
               "",
               "| feature | PH (μ±SD) | nonPH (μ±SD) | Δ | MWU p |",
               "|---|---|---|---|---|"]
        for col, r in out["C_endotype_replication"].items():
            md.append(f"| {col} | {r['PH_mean']:.3f} ± {r['PH_sd']:.3f} | "
                      f"{r['nonPH_mean']:.3f} ± {r['nonPH_sd']:.3f} | "
                      f"{r['delta']:+.3f} | {r.get('mwu_p', 'NA')} |"
                      if r.get("mwu_p") is not None
                      else f"| {col} | {r['PH_mean']:.3f} | {r['nonPH_mean']:.3f} | {r['delta']:+.3f} | NA |")
        md.append("")
    if "new_ingestion_check" in out:
        r = out["new_ingestion_check"]
        md += ["## New-ingestion sanity check",
               "",
               f"- {r['n_new']} new cases ingested",
               f"- All label=0: {r['all_label0']}",
               f"- All plain_scan: {r['all_plain']}",
               f"- Mean paren_mean_HU: {r['paren_mean_HU_mean']:.1f}",
               f"- Mean paren_LAA_950_frac: {r['paren_LAA_950_frac_mean']:.4f}",
               ""]

    (OUT / "enlarged_lung_results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Saved {OUT}/enlarged_lung_results.{{json,md}}")


if __name__ == "__main__":
    main()
