"""R20.B — Merge PH-only + nonPH DDPM CSVs and compute final PH-vs-nonPH AUC.

Inputs:
- outputs/r20/ddpm_anomaly_legacy.csv (84 nonPH from v1)
- outputs/r20/ddpm_anomaly_legacy_ph.csv (≤170 PH from v3 PH-only run)
Optional:
- outputs/r19/ddpm_anomaly_scores.csv (100 nonPH from remote new100)

Output:
- outputs/r20/ddpm_anomaly_merged.csv (all-cases)
- outputs/r20/ddpm_anomaly_merged_eval.{json,md}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "r20"


def main():
    nonph = pd.read_csv(OUT / "ddpm_anomaly_legacy.csv")
    ph_p = OUT / "ddpm_anomaly_legacy_ph.csv"
    ph = pd.read_csv(ph_p) if ph_p.exists() else pd.DataFrame(columns=nonph.columns)
    print(f"loaded: nonph={len(nonph)} (label.unique={nonph['label'].unique().tolist()}), "
          f"ph={len(ph)} (label.unique={ph['label'].unique().tolist() if len(ph) else []})")

    extra = ROOT / "outputs" / "r19" / "ddpm_anomaly_scores.csv"
    if extra.exists():
        ex = pd.read_csv(extra)
        ex_new = ex[~ex["case_id"].isin(set(nonph["case_id"]) | set(ph["case_id"]))]
        print(f"adding {len(ex_new)} cases from r19/ddpm_anomaly_scores.csv")
    else:
        ex_new = pd.DataFrame(columns=nonph.columns)

    merged = pd.concat([nonph, ph, ex_new], ignore_index=True)
    merged = merged.drop_duplicates(subset=["case_id"], keep="first")
    out_csv = OUT / "ddpm_anomaly_merged.csv"
    merged.to_csv(out_csv, index=False)
    print(f"merged → {out_csv} (n={len(merged)})")

    out = {
        "n_total": int(len(merged)),
        "n_ph": int((merged["label"] == 1).sum()),
        "n_nonph": int((merged["label"] == 0).sum()),
    }

    if merged["label"].nunique() == 2:
        from sklearn.metrics import roc_auc_score
        from scipy.stats import mannwhitneyu
        y = merged["label"].astype(int).values
        nll = merged["mean_nll"].astype(float).values
        auc = float(roc_auc_score(y, nll))
        out["anomaly_auc"] = auc
        out["anomaly_auc_inverted"] = float(1.0 - auc)
        # Bootstrap CI
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(y), size=len(y))
            try:
                b = roc_auc_score(y[idx], nll[idx]); boots.append(b)
            except ValueError:
                continue
        if len(boots) > 100:
            out["auc_ci95_lo"] = float(np.percentile(boots, 2.5))
            out["auc_ci95_hi"] = float(np.percentile(boots, 97.5))
        ph_nll = merged.loc[merged["label"] == 1, "mean_nll"].astype(float).values
        np_nll = merged.loc[merged["label"] == 0, "mean_nll"].astype(float).values
        u_g, p_g = mannwhitneyu(ph_nll, np_nll, alternative="greater")
        u_l, p_l = mannwhitneyu(ph_nll, np_nll, alternative="less")
        u_t, p_t = mannwhitneyu(ph_nll, np_nll, alternative="two-sided")
        out["mwu_ph_gt_nonph_p"] = float(p_g)
        out["mwu_ph_lt_nonph_p"] = float(p_l)
        out["mwu_two_sided_p"] = float(p_t)
        out["ph_mean_nll"] = float(ph_nll.mean())
        out["nonph_mean_nll"] = float(np_nll.mean())
        out["delta_nll"] = float(ph_nll.mean() - np_nll.mean())
        out["ph_median_nll"] = float(np.median(ph_nll))
        out["nonph_median_nll"] = float(np.median(np_nll))

    (OUT / "ddpm_anomaly_merged_eval.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    md = ["# R20.B — DDPM anomaly evaluation (merged PH + nonPH)",
          "",
          "Diffusion model trained on plain-scan nonPH (R19 new100, 30 epochs).",
          "Inference: per-case mean MSE-noise-prediction NLL on 4-8 random 32³ "
          "lung patches at t=T/2.",
          "",
          f"**Cohort**: n={out['n_total']} (PH={out.get('n_ph',0)}, "
          f"nonPH={out.get('n_nonph',0)})",
          ""]
    if "anomaly_auc" in out:
        ci_str = ""
        if "auc_ci95_lo" in out:
            ci_str = (f" [95% boot CI {out['auc_ci95_lo']:.3f}, "
                      f"{out['auc_ci95_hi']:.3f}]")
        md += [f"## Anomaly AUC",
               "",
               f"- AUC (PH > nonPH direction) = **{out['anomaly_auc']:.3f}**{ci_str}",
               f"- AUC inverted (nonPH > PH) = **{out['anomaly_auc_inverted']:.3f}**",
               f"- PH mean NLL = {out['ph_mean_nll']:.4f}, "
               f"nonPH mean NLL = {out['nonph_mean_nll']:.4f}",
               f"- PH median NLL = {out['ph_median_nll']:.4f}, "
               f"nonPH median NLL = {out['nonph_median_nll']:.4f}",
               f"- Δ (PH−nonPH) = {out['delta_nll']:+.4f}",
               f"- MWU one-sided p (PH > nonPH) = "
               f"{out.get('mwu_ph_gt_nonph_p', 'NA')}",
               "",
               "## Honest interpretation",
               "",
               "DDPM was trained on **plain-scan** nonPH (R19 new100, n=100).",
               "Inference here is on **legacy CTPA** cohort (contrast protocol).",
               "Both PH and nonPH cases in inference are **out-of-distribution**",
               "relative to the plain-scan training set — different protocol,",
               "different image statistics. Whichever class happens to be closer",
               "to plain-scan in pixel statistics dominates the AUC.",
               "",
               "If AUC > 0.6 in EITHER direction with no obvious protocol",
               "explanation, the result may reflect biology. Otherwise, the",
               "result is dominated by **protocol shift, not disease**, and",
               "this evaluation should not be cited as evidence of label-free",
               "PH detection. Closing R18 must-fix #1 with this honest negative.",
               ""]
    (OUT / "ddpm_anomaly_merged_eval.md").write_text(
        "\n".join(md), encoding="utf-8")
    print(f"saved → {OUT}/ddpm_anomaly_merged_eval.{{json,md}}")
    if "anomaly_auc" in out:
        ci = ""
        if "auc_ci95_lo" in out:
            ci = f" [95% CI {out['auc_ci95_lo']:.3f}, {out['auc_ci95_hi']:.3f}]"
        print(f"\nAnomaly AUC = {out['anomaly_auc']:.3f}{ci}")


if __name__ == "__main__":
    main()
