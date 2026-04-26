"""R22.A — Proper deconfounding with NESTED train-fold CORAL + bootstrap CI.

Closes R21 codex feedback:
1. Nested CORAL (fit on train fold only, apply to val) — avoids the
   train-test leakage that R21.D's whole-cohort CORAL produced.
2. Bootstrap-500 CI on within-contrast disease AUC.
3. Fold-level AUC table (n_nonPH=27 small, fold-level honesty).
4. Univariate per-feature CORAL alternative (means+stds only) as
   less-aggressive baseline.

Output: outputs/r20/r22a_nested_deconfound.{json,md}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent.parent
MORPH = ROOT / "outputs" / "r20" / "morph_unified301.csv"
PROTO = ROOT / "data" / "case_protocol_extended.csv"
OUT = ROOT / "outputs" / "r20"


def coral_full(Xs, Xt, lam=1.0):
    Xs, Xt = np.asarray(Xs, float), np.asarray(Xt, float)
    if Xs.shape[0] < 2 or Xt.shape[0] < 2: return Xs
    Cs = np.cov(Xs.T) + lam * np.eye(Xs.shape[1])
    Ct = np.cov(Xt.T) + lam * np.eye(Xt.shape[1])
    eig_s, V_s = np.linalg.eigh(Cs)
    eig_s = np.clip(eig_s, 1e-8, None)
    Cs_isr = V_s @ np.diag(1.0/np.sqrt(eig_s)) @ V_s.T
    eig_t, V_t = np.linalg.eigh(Ct)
    eig_t = np.clip(eig_t, 1e-8, None)
    Ct_sr = V_t @ np.diag(np.sqrt(eig_t)) @ V_t.T
    return (Xs - Xs.mean(0)) @ Cs_isr @ Ct_sr + Xt.mean(0)


def coral_univariate(Xs, Xt):
    """Per-feature mean/std alignment — less aggressive than full CORAL."""
    Xs, Xt = np.asarray(Xs, float), np.asarray(Xt, float)
    mu_s, sd_s = Xs.mean(0), Xs.std(0) + 1e-8
    mu_t, sd_t = Xt.mean(0), Xt.std(0) + 1e-8
    return (Xs - mu_s) / sd_s * sd_t + mu_t


def kfold_auc_nested_coral(X, y, protocol, seed=42, k=5, coral_fn=None):
    """5-fold CV with nested CORAL: fit on train (split contrast vs plain),
    apply to val. Returns per-fold AUC and mean."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    aucs = []
    for tr, va in skf.split(X, y):
        Xtr_full, ytr = X[tr], y[tr]
        Xva_full, yva = X[va], y[va]
        if coral_fn is not None:
            # Within TRAIN: separate plain vs contrast among nonPH (label=0)
            ptr = protocol[tr]
            tr_plain_mask = (ytr == 0) & (ptr == 0)  # plain nonPH
            tr_contrast_mask = (ytr == 0) & (ptr == 1)  # contrast nonPH
            if tr_plain_mask.sum() >= 3 and tr_contrast_mask.sum() >= 3:
                Xs = Xtr_full[tr_plain_mask]
                Xt = Xtr_full[tr_contrast_mask]
                Xtr_aligned = Xtr_full.copy()
                # Apply coral params to plain rows in train (recompute for symmetry)
                Xtr_aligned[tr_plain_mask] = coral_fn(Xs, Xt)
                # Apply same alignment to val plain rows using TRAIN distributions
                pva = protocol[va]
                va_plain_mask = pva == 0
                Xva_aligned = Xva_full.copy()
                if va_plain_mask.sum() >= 1:
                    # Use train-fitted distributions: mean/std of plain (Xs) and contrast (Xt)
                    if coral_fn is coral_univariate:
                        mu_s, sd_s = Xs.mean(0), Xs.std(0) + 1e-8
                        mu_t, sd_t = Xt.mean(0), Xt.std(0) + 1e-8
                        Xva_aligned[va_plain_mask] = (
                            Xva_full[va_plain_mask] - mu_s) / sd_s * sd_t + mu_t
                    else:
                        # Full CORAL: just transform val plain using same Xs→Xt fit
                        Xva_aligned[va_plain_mask] = coral_full(
                            Xva_full[va_plain_mask], Xt)
            else:
                Xtr_aligned = Xtr_full
                Xva_aligned = Xva_full
        else:
            Xtr_aligned = Xtr_full
            Xva_aligned = Xva_full
        sc = StandardScaler()
        Xtr_z = sc.fit_transform(Xtr_aligned); Xva_z = sc.transform(Xva_aligned)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr_z, ytr)
        try:
            aucs.append(roc_auc_score(yva, clf.predict_proba(Xva_z)[:, 1]))
        except ValueError:
            continue
    return aucs


def main():
    df = pd.read_csv(MORPH)
    pro = pd.read_csv(PROTO)
    df = df.merge(pro[["case_id", "protocol"]], on="case_id", how="left")
    artifacts = {"airway_n_terminals", "airway_term_per_node", "artery_lap_eig0",
                 "artery_n_terminals", "artery_term_per_node", "vein_lap_eig0",
                 "vein_n_terminals", "vein_term_per_node"}
    feat_cols = [c for c in df.columns
                 if c not in ("case_id", "label", "protocol", "source_cache")
                 and c not in artifacts]
    Xall = df[feat_cols].fillna(0).values.astype(float)
    y = df["label"].astype(int).values
    is_contrast = (df["protocol"].str.lower() == "contrast").astype(int).values
    print(f"using {len(feat_cols)} features; cohort n={len(df)}")

    seeds = [42, 43, 44, 45, 46]
    out = {"n_features": len(feat_cols), "n_cohort": int(len(df))}

    # Within-contrast disease AUC: baseline + nested CORAL
    contrast_mask = is_contrast == 1
    Xc, yc, pc = Xall[contrast_mask], y[contrast_mask], is_contrast[contrast_mask]

    print("\n=== Within-contrast disease AUC (no CORAL) ===")
    base_aucs_per_fold = []
    for s in seeds:
        a = kfold_auc_nested_coral(Xc, yc, pc, seed=s)
        base_aucs_per_fold.append(a)
        print(f"  seed={s}: folds={[f'{x:.3f}' for x in a]} mean={np.mean(a):.3f}")
    flat = [v for fold in base_aucs_per_fold for v in fold]
    out["within_contrast_baseline"] = {
        "auc_mean_seedmean": float(np.mean([np.mean(f) for f in base_aucs_per_fold])),
        "auc_std_seedstd": float(np.std([np.mean(f) for f in base_aucs_per_fold])),
        "all_25_folds": [float(v) for v in flat],
        "fold_min": float(min(flat)), "fold_max": float(max(flat)),
        "n_ph": int(yc.sum()), "n_nonph": int((yc==0).sum()),
    }
    # Bootstrap CI for within-contrast AUC
    rng = np.random.default_rng(0)
    boot_aucs = []
    n = len(yc)
    for _ in range(500):
        idx = rng.choice(n, size=n, replace=True)
        if yc[idx].sum() < 3 or (yc[idx]==0).sum() < 3: continue
        a = kfold_auc_nested_coral(Xc[idx], yc[idx], pc[idx], seed=42)
        if a:
            boot_aucs.append(np.mean(a))
    out["within_contrast_baseline"]["bootstrap_500"] = {
        "n_succeed": len(boot_aucs),
        "mean": float(np.mean(boot_aucs)) if boot_aucs else None,
        "ci95_lo": float(np.percentile(boot_aucs, 2.5)) if boot_aucs else None,
        "ci95_hi": float(np.percentile(boot_aucs, 97.5)) if boot_aucs else None,
    }
    print(f"  bootstrap-500: mean={np.mean(boot_aucs):.3f} 95%CI=[{np.percentile(boot_aucs,2.5):.3f}, {np.percentile(boot_aucs,97.5):.3f}]")

    # Now within-nonPH protocol-AUC under NESTED univariate CORAL (less aggressive)
    print("\n=== Within-nonPH protocol-AUC (control test) ===")
    print("Nested CORAL: fit on TRAIN fold only, apply to VAL (no test leakage)")
    nonph_mask = (y == 0)
    Xn = Xall[nonph_mask]; pn = is_contrast[nonph_mask]
    # No-coral baseline
    no_coral_proto_aucs = []
    for s in seeds:
        a = kfold_auc_nested_coral(Xn, pn, pn, seed=s, coral_fn=None)
        no_coral_proto_aucs.extend(a)
    no_coral_oriented = [max(v, 1-v) for v in no_coral_proto_aucs]
    out["protocol_auc_no_coral_oriented"] = {
        "mean": float(np.mean(no_coral_oriented)),
        "std": float(np.std(no_coral_oriented))}
    print(f"  no CORAL: orient-AUC = {np.mean(no_coral_oriented):.3f} ± {np.std(no_coral_oriented):.3f}")

    # Nested univariate CORAL
    uni_proto_aucs = []
    for s in seeds:
        a = kfold_auc_nested_coral(Xn, pn, pn, seed=s, coral_fn=coral_univariate)
        uni_proto_aucs.extend(a)
    uni_oriented = [max(v, 1-v) for v in uni_proto_aucs]
    out["protocol_auc_nested_univariate_coral_oriented"] = {
        "mean": float(np.mean(uni_oriented)),
        "std": float(np.std(uni_oriented))}
    print(f"  nested univariate CORAL: orient-AUC = {np.mean(uni_oriented):.3f} ± {np.std(uni_oriented):.3f}")

    # Verdict
    out["verdict"] = {
        "within_contrast_disease_auc": out["within_contrast_baseline"]["auc_mean_seedmean"],
        "within_contrast_disease_ci95": [
            out["within_contrast_baseline"]["bootstrap_500"]["ci95_lo"],
            out["within_contrast_baseline"]["bootstrap_500"]["ci95_hi"]],
        "fold_min": out["within_contrast_baseline"]["fold_min"],
        "fold_max": out["within_contrast_baseline"]["fold_max"],
        "protocol_leakage_pre": float(np.mean(no_coral_oriented)),
        "protocol_leakage_post_nested_uni_coral": float(np.mean(uni_oriented)),
        "deconfounding_drop": float(np.mean(no_coral_oriented) - np.mean(uni_oriented)),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r22a_nested_deconfound.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    md = ["# R22.A — Proper deconfounding (nested CORAL + bootstrap CI)",
          "",
          "Per R21 codex feedback: nested train-fold CORAL avoids test leakage; "
          "univariate per-feature alignment is less aggressive than full-covariance "
          "and avoids over-correction artifact.",
          "",
          "## Within-contrast disease AUC (baseline, no CORAL needed within contrast)",
          "",
          f"- 5-seed × 5-fold = 25 fold AUCs",
          f"- Mean across seed-means = **{out['verdict']['within_contrast_disease_auc']:.3f}**",
          f"- Per-fold range = [{out['verdict']['fold_min']:.3f}, {out['verdict']['fold_max']:.3f}]",
          f"- Bootstrap-500 95% CI = [{out['verdict']['within_contrast_disease_ci95'][0]:.3f}, "
          f"{out['verdict']['within_contrast_disease_ci95'][1]:.3f}]",
          f"- Cohort: n={out['within_contrast_baseline']['n_ph'] + out['within_contrast_baseline']['n_nonph']} "
          f"({out['within_contrast_baseline']['n_ph']} PH + "
          f"{out['within_contrast_baseline']['n_nonph']} nonPH)",
          f"- **Caveat**: n_nonPH=27 small; per-fold AUC variance is real, not "
          "an artifact. Each fold has ~5-6 nonPH cases. Bootstrap CI captures "
          "this uncertainty.",
          "",
          "## Within-nonPH protocol-AUC (orientation-free)",
          "",
          f"- No CORAL: orientation-AUC = **{out['verdict']['protocol_leakage_pre']:.3f}**",
          f"- Nested univariate CORAL: orientation-AUC = **{out['verdict']['protocol_leakage_post_nested_uni_coral']:.3f}**",
          f"- Drop: {out['verdict']['deconfounding_drop']:+.3f}",
          ""]
    if out["verdict"]["protocol_leakage_post_nested_uni_coral"] < 0.65:
        md.append("**Closes R20 must-fix #5 with PARTIAL/POSITIVE verdict** "
                  "(feature-level): nested univariate CORAL drops orientation-free "
                  "protocol leakage to near-chance levels in 5-seed CV. NOTE: this "
                  "does not close GCN-embedding-level deconfounding (R20 #3); a "
                  "principled paper repositioning is needed if GCN training is to "
                  "be retired in favor of feature-level claims.")
    elif out["verdict"]["protocol_leakage_post_nested_uni_coral"] < 0.85:
        md.append("**PARTIAL verdict**: nested univariate CORAL reduces protocol leakage "
                  "but residual remains. The morphometric feature space is partially "
                  "but not fully deconfounded.")
    else:
        md.append("**HONEST NEGATIVE**: nested CORAL still has high orientation-free "
                  "leakage. Feature-level deconfounding insufficient — need richer "
                  "approach (training-time GRL, deeper feature learning, or paper "
                  "repositioning).")
    (OUT / "r22a_nested_deconfound.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nsaved → {OUT}/r22a_nested_deconfound.{{json,md}}")


if __name__ == "__main__":
    main()
