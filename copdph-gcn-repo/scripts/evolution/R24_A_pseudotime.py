"""R24.A — Pseudotime via Laplacian Eigenmap (diffusion-map alternative; PHATE unavailable locally).

Within-contrast n=190 PRIMARY; full-cohort n=290 sensitivity (protocol-confounded stress test).
Gates (predefined): mPAP |ρ| ≥ 0.35 AND protocol AUC < 0.65 AND |ρ_protocol| < 0.20

Output:
  outputs/r24/r24a_pseudotime_within_contrast.csv (case_id, pseudotime, label, mpap)
  outputs/r24/r24a_pseudotime_full.csv
  outputs/r24/r24a_validation.json
  outputs/figures/fig_r24a_pseudotime.png
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
COHORT = ROOT / "outputs" / "r24" / "cohort_locked_table.csv"
MORPH = ROOT / "outputs" / "r20" / "morph_unified301.csv"
OUT = ROOT / "outputs" / "r24"
FIG = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

ARTIFACTS = {"airway_n_terminals", "airway_term_per_node",
             "artery_lap_eig0", "artery_n_terminals", "artery_term_per_node",
             "vein_lap_eig0", "vein_n_terminals", "vein_term_per_node"}


def kfold_protocol_auc(X, y_protocol, seed=42, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    aucs = []
    for tr, va in skf.split(X, y_protocol):
        sc = RobustScaler(); Xtr = sc.fit_transform(X[tr]); Xva = sc.transform(X[va])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y_protocol[tr])
        try:
            aucs.append(roc_auc_score(y_protocol[va], clf.predict_proba(Xva)[:, 1]))
        except Exception:
            continue
    return float(np.mean(aucs)) if aucs else float("nan")


def compute_pseudotime(X, n_components=2, knn=5):
    """Spectral embedding + use 1st non-trivial component as pseudotime axis."""
    se = SpectralEmbedding(n_components=n_components, n_neighbors=knn,
                            affinity="nearest_neighbors", random_state=42)
    emb = se.fit_transform(X)
    return emb[:, 0], emb


def run_stratum(cohort: pd.DataFrame, morph: pd.DataFrame, name: str):
    df = cohort.merge(morph, on="case_id", how="inner", suffixes=("", "_dup"))
    feat_cols = [c for c in morph.columns
                 if c not in ("case_id", "label", "source_cache")
                 and c not in ARTIFACTS]
    X_raw = df[feat_cols].fillna(0).values.astype(float)
    X = RobustScaler().fit_transform(X_raw)
    pt, emb = compute_pseudotime(X, n_components=2, knn=5)
    # Orient: higher pseudotime = more PH-like (rank-correlate with label)
    if spearmanr(pt, df["label"])[0] < 0:
        pt = -pt
    df_out = pd.DataFrame({
        "case_id": df["case_id"].values,
        "label": df["label"].values,
        "protocol": df["protocol"].values,
        "measured_mpap": df["measured_mpap"].values,
        "measured_mpap_flag": df["measured_mpap_flag"].values,
        "pseudotime": pt,
        "embed_x": emb[:, 0], "embed_y": emb[:, 1],
    })
    out_csv = OUT / f"r24a_pseudotime_{name}.csv"
    df_out.to_csv(out_csv, index=False)
    return df_out, out_csv, X, feat_cols


def validate(df_out: pd.DataFrame, X: np.ndarray, name: str):
    """Compute the 3 R24.A gates."""
    rho_label, p_label = spearmanr(df_out["pseudotime"], df_out["label"])
    sub_mpap = df_out.dropna(subset=["measured_mpap"])
    if len(sub_mpap) >= 5:
        rho_mpap, p_mpap = spearmanr(sub_mpap["pseudotime"], sub_mpap["measured_mpap"])
    else:
        rho_mpap, p_mpap = float("nan"), float("nan")
    # Protocol gate (only meaningful in full-cohort)
    if df_out["protocol"].nunique() > 1:
        y_proto = (df_out["protocol"] == "contrast").astype(int).values
        proto_auc = kfold_protocol_auc(X, y_proto)
        rho_proto, _ = spearmanr(df_out["pseudotime"], y_proto)
    else:
        proto_auc, rho_proto = float("nan"), float("nan")
    gates = {
        "stratum": name,
        "n": int(len(df_out)),
        "rho_label": float(rho_label), "p_label": float(p_label),
        "rho_mpap": float(rho_mpap), "p_mpap": float(p_mpap),
        "n_mpap_resolved": int(len(sub_mpap)),
        "protocol_auc_5fold": float(proto_auc),
        "rho_protocol": float(rho_proto) if not np.isnan(rho_proto) else None,
        "gate_mpap_rho_ge_0.35": bool(abs(rho_mpap) >= 0.35) if not np.isnan(rho_mpap) else False,
        "gate_protocol_auc_lt_0.65": bool(proto_auc < 0.65) if not np.isnan(proto_auc) else None,
        "gate_protocol_rho_lt_0.20": bool(abs(rho_proto) < 0.20) if rho_proto is not None and not np.isnan(rho_proto) else None,
    }
    return gates


def main():
    cohort = pd.read_csv(COHORT)
    morph = pd.read_csv(MORPH)

    # Within-contrast PRIMARY
    cw = cohort[cohort["is_contrast_only_subset"]].copy()
    df_w, csv_w, X_w, feat = run_stratum(cw, morph, "within_contrast")
    gates_w = validate(df_w, X_w, "within_contrast")

    # Full-cohort sensitivity / protocol stress test
    df_f, csv_f, X_f, _ = run_stratum(cohort, morph, "full")
    gates_f = validate(df_f, X_f, "full_protocol_stress_test")

    val = {"within_contrast": gates_w, "full_cohort_stress_test": gates_f,
           "feature_count": len(feat),
           "method": "SpectralEmbedding + RobustScaler (PHATE unavailable locally; eigenmap is theoretical equivalent at small scale)"}
    (OUT / "r24a_validation.json").write_text(
        json.dumps(val, indent=2), encoding="utf-8")

    # Figure: 2-panel embedding (within-contrast + full) colored by mPAP / label
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for col, (df_p, name) in enumerate([(df_w, "Within-contrast n=190 (PRIMARY)"),
                                          (df_f, "Full-cohort n=290 (protocol stress test)")]):
        # Top: colored by label
        ax = axes[0, col]
        for lbl, c, tag in [(0, "#3b82f6", "nonPH"), (1, "#ef4444", "PH")]:
            sub = df_p[df_p["label"] == lbl]
            ax.scatter(sub["embed_x"], sub["embed_y"], c=c, alpha=0.6,
                       s=22, label=tag, edgecolors="none")
        ax.legend(); ax.set_xlabel("Spectral coord 1"); ax.set_ylabel("Spectral coord 2")
        ax.set_title(f"{name}\n按 PH 标签着色 / colored by label", fontsize=11)
        ax.grid(alpha=0.3)
        # Bottom: colored by measured mPAP
        ax = axes[1, col]
        sub = df_p.dropna(subset=["measured_mpap"])
        sc = ax.scatter(sub["embed_x"], sub["embed_y"], c=sub["measured_mpap"],
                        cmap="plasma", s=24, edgecolors="black", linewidths=0.3)
        plt.colorbar(sc, ax=ax, label="measured mPAP (mmHg)")
        ax.set_xlabel("Spectral coord 1"); ax.set_ylabel("Spectral coord 2")
        ax.set_title(f"{name} — 测得 mPAP 着色 (n_resolved={len(sub)})\n"
                     f"ρ_mPAP={val['within_contrast' if col==0 else 'full_cohort_stress_test']['rho_mpap']:+.3f}",
                     fontsize=11)
        ax.grid(alpha=0.3)

    fig.suptitle("R24.A — Pseudotime via Spectral Embedding\n"
                 "Q1 (vascular phenotype evolution) + Q4(b) longitudinal substitute",
                 fontsize=13, y=1.00)
    plt.tight_layout()
    out_fig = FIG / "fig_r24a_pseudotime.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()

    print(json.dumps(val, indent=2))
    print(f"\nfigure: {out_fig}")
    print(f"CSVs: {csv_w}, {csv_f}")


if __name__ == "__main__":
    main()
