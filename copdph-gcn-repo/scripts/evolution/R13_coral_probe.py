"""R13.4 — Within-nonPH protocol probe on CORAL embeddings + seg-failure exclusion.

Pulls per-fold CORAL embeddings from remote (4 lambdas × seed 42), excludes
the 34 real-seg-failure cases identified in R13.2b, runs the same LR + MLP
within-nonPH protocol probe used in R11/R12, and compares against the
corrected-GRL R11 baseline.

Outputs:
  outputs/r13/coral_probe.{json,md}
  outputs/r13/coral_vs_grl_comparison.md

Key correction vs R11/R12:
  Before:  effective within-nonPH n=80 (all in-cache nonPH-plain + nonPH-contrast)
  R13:     n_effective = 80 - (real-seg-failures in nonph_plain ∩ in-cache)
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

REMOTE = "imss@10.60.147.117"
REMOTE_BASE = "/home/imss/cw/GCN copdnoph copdph/outputs"
ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "r13"
OUT.mkdir(parents=True, exist_ok=True)
CORAL_LOCAL = OUT / "coral_embeddings"
CORAL_LOCAL.mkdir(exist_ok=True)
SPLITS = ROOT / "data" / "splits_expanded_282"
CACHE_LIST = ROOT / "outputs" / "r5" / "cache_v2_tri_flat_list.txt"
PROTO = ROOT / "data" / "case_protocol.csv"
LABELS = ROOT / "data" / "labels_expanded_282.csv"
SEG_FAILS = OUT / "seg_failures_real.json"

LAMBDAS = [0.0, 1.0, 5.0, 10.0]
SEEDS = [42]


def fetch_coral_emb(lam: float, seed: int) -> Path | None:
    """Pull all 5 fold embeddings for one (lambda, seed) config."""
    out_dir = CORAL_LOCAL / f"l{lam}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = f"{REMOTE_BASE}/sprint6_arm_a_coral_l{lam}_s{seed}/embeddings"
    for k in range(1, 6):
        local_f = out_dir / f"emb_gcn_only_rep1_fold{k}.npz"
        if local_f.exists():
            continue
        cmd = ["scp", "-q", f"{REMOTE}:{remote_dir}/emb_gcn_only_rep1_fold{k}.npz",
               str(local_f)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[skip] l{lam}_s{seed} fold{k}: {r.stderr.strip()[:100]}")
            return None
    return out_dir


def load_emb_with_ids(emb_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Re-attach case_ids by reading the per-fold val.txt."""
    cached = set(c.strip() for c in CACHE_LIST.read_text(encoding="utf-8").splitlines() if c.strip())
    all_e, all_y, all_c = [], [], []
    for k in range(1, 6):
        f = emb_dir / f"emb_gcn_only_rep1_fold{k}.npz"
        if not f.exists():
            return None
        d = np.load(f)
        val_ids = [c.strip() for c in (SPLITS / f"fold_{k}" / "val.txt").read_text().splitlines() if c.strip()]
        val_ids = [c for c in val_ids if c in cached]
        if len(val_ids) != len(d["embeddings"]):
            print(f"  fold {k}: id-count mismatch ({len(val_ids)} vs {len(d['embeddings'])})")
            return None
        all_e.append(d["embeddings"])
        all_y.extend(d["y_true"].tolist())
        all_c.extend(val_ids)
    return np.concatenate(all_e), np.array(all_y, int), all_c


def oof_lr_mlp(X: np.ndarray, y: np.ndarray, seed: int = 42):
    if len(np.unique(y)) < 2:
        return None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_lr = np.zeros(len(y))
    oof_mlp = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        lr = LogisticRegression(max_iter=5000, class_weight="balanced")
        lr.fit(Xtr, y[tr])
        oof_lr[te] = lr.predict_proba(Xte)[:, 1]
        mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=seed)
        mlp.fit(Xtr, y[tr])
        oof_mlp[te] = mlp.predict_proba(Xte)[:, 1]
    auc_lr = float(roc_auc_score(y, oof_lr))
    auc_mlp = float(roc_auc_score(y, oof_mlp))
    return auc_lr, auc_mlp, oof_lr, oof_mlp


def boot_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
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


def main():
    proto = pd.read_csv(PROTO)
    proto["is_contrast"] = (proto["protocol"].astype(str).str.lower() == "contrast").astype(int)
    proto_lookup = dict(zip(proto["case_id"], proto["is_contrast"]))
    label_df = pd.read_csv(LABELS)
    lbl_lookup = dict(zip(label_df["case_id"], label_df["label"]))

    # Load seg-failure exclusion list
    seg_fail_ids: set[str] = set()
    if SEG_FAILS.exists():
        sf = json.loads(SEG_FAILS.read_text(encoding="utf-8"))
        for r in sf.get("real_fails", []) + sf.get("lung_anomaly", []):
            seg_fail_ids.add(r["case_id"])
    print(f"[exclusion] {len(seg_fail_ids)} cases on seg-failure list")

    summary: dict = {"per_lambda": {}, "n_excluded_seg_fails": len(seg_fail_ids)}
    for lam in LAMBDAS:
        for seed in SEEDS:
            emb_dir = fetch_coral_emb(lam, seed)
            if emb_dir is None:
                continue
            res = load_emb_with_ids(emb_dir)
            if res is None:
                continue
            X_all, y_all, ids_all = res
            # Restrict to nonPH (label==0)
            nonph_mask = np.array([lbl_lookup.get(c, 1) == 0 for c in ids_all])
            X = X_all[nonph_mask]
            ids = [c for c, m in zip(ids_all, nonph_mask) if m]
            y_proto = np.array([proto_lookup.get(c, -1) for c in ids], int)
            keep = y_proto >= 0
            X = X[keep]; y_proto = y_proto[keep]
            ids = [c for c, k in zip(ids, keep) if k]

            n_before = len(ids)
            # Apply seg-failure exclusion
            keep_mask = np.array([c not in seg_fail_ids for c in ids])
            X_corr = X[keep_mask]
            y_corr = y_proto[keep_mask]
            ids_corr = [c for c, k in zip(ids, keep_mask) if k]
            n_after = len(ids_corr)
            n_excluded = n_before - n_after

            # Run probes BOTH on full (uncorrected) and corrected sets
            full = oof_lr_mlp(X, y_proto, seed=seed)
            corr = oof_lr_mlp(X_corr, y_corr, seed=seed)
            if full is None or corr is None:
                continue
            f_lr, f_mlp, f_lr_oof, f_mlp_oof = full
            c_lr, c_mlp, c_lr_oof, c_mlp_oof = corr

            summary["per_lambda"][f"lambda_{lam}"] = {
                "n_full": n_before,
                "n_corrected": n_after,
                "n_excluded_seg_fail": n_excluded,
                "full_cohort": {
                    "lr_auc": f_lr,
                    "lr_ci95": boot_ci(y_proto, f_lr_oof),
                    "mlp_auc": f_mlp,
                    "mlp_ci95": boot_ci(y_proto, f_mlp_oof),
                },
                "corrected_cohort": {
                    "lr_auc": c_lr,
                    "lr_ci95": boot_ci(y_corr, c_lr_oof),
                    "mlp_auc": c_mlp,
                    "mlp_ci95": boot_ci(y_corr, c_mlp_oof),
                },
            }
            print(f"l{lam}_s{seed}: n_full={n_before} n_corr={n_after} excl={n_excluded}")
            print(f"  full     LR={f_lr:.3f} MLP={f_mlp:.3f}")
            print(f"  corrected LR={c_lr:.3f} MLP={c_mlp:.3f}")

    out_json = OUT / "coral_probe.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Comparison MD
    md = ["# R13.4 — CORAL within-nonPH protocol probe (vs corrected-GRL R11)",
          "",
          f"Excluded {len(seg_fail_ids)} cases on seg-failure list (R13.2b).",
          "",
          "## Full cohort (legacy R11/R12 denominator, uncorrected)",
          "",
          "| λ | n | LR AUC [95% CI] | MLP AUC [95% CI] |",
          "|---|---|---|---|"]
    for lam in LAMBDAS:
        rec = summary["per_lambda"].get(f"lambda_{lam}")
        if not rec: continue
        f = rec["full_cohort"]
        md.append(f"| {lam} | {rec['n_full']} | "
                  f"{f['lr_auc']:.3f} [{f['lr_ci95'][0]:.3f}, {f['lr_ci95'][1]:.3f}] | "
                  f"{f['mlp_auc']:.3f} [{f['mlp_ci95'][0]:.3f}, {f['mlp_ci95'][1]:.3f}] |")

    md += ["",
           "## Corrected cohort (excluding seg-failures)",
           "",
           "| λ | n_full → n_corrected (excluded) | LR AUC [95% CI] | MLP AUC [95% CI] |",
           "|---|---|---|---|"]
    for lam in LAMBDAS:
        rec = summary["per_lambda"].get(f"lambda_{lam}")
        if not rec: continue
        c = rec["corrected_cohort"]
        md.append(f"| {lam} | {rec['n_full']} → {rec['n_corrected']} ({rec['n_excluded_seg_fail']}) | "
                  f"{c['lr_auc']:.3f} [{c['lr_ci95'][0]:.3f}, {c['lr_ci95'][1]:.3f}] | "
                  f"{c['mlp_auc']:.3f} [{c['mlp_ci95'][0]:.3f}, {c['mlp_ci95'][1]:.3f}] |")

    md += ["",
           "## Comparison vs corrected-GRL R11 baseline",
           "",
           "R11 corrected-GRL @ seed=42 (within-nonPH protocol LR, n=80, full):",
           "  λ=0: 0.840 | λ=1: 0.894 | λ=5: 0.842 | λ=10: 0.790",
           "",
           "R12 cross-seed pooled (n=80, full):",
           "  λ=0: 0.867 | λ=1: 0.902 | λ=5: 0.886 | λ=10: 0.873",
           "",
           "If CORAL drives the corrected-cohort LR AUC ≤0.60 with upper-CI ≤0.65,",
           "we have broken the protocol floor and R14 expands to seeds {1042, 2042}.",
           "Otherwise, the negative-result is sharper than R12 (now confirmed across",
           "two distinct deconfounder families on the same cohort), and the path to",
           "≥9.5 requires 345-cohort ingestion + segmentation re-runs.",
           ""]

    (OUT / "coral_probe.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved {out_json}")


if __name__ == "__main__":
    main()
