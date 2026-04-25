"""R19.F — Within-pipeline mPAP trends (legacy-only vs new100-only).

Verifies the R19.E confound diagnosis by stratifying mPAP trends by
source_cache (legacy HU-sentinel vs new100 binary). If trends are
strong within each pipeline but reverse across pipelines, the confound
is confirmed.

Output: outputs/r19/within_pipeline_trends.{json,md}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "outputs" / "r19"
MORPH = OUT / "per_structure_morphometrics_extended.csv"
LABELS = ROOT / "data" / "labels_extended_382.csv"
PROTO = ROOT / "data" / "case_protocol_extended.csv"
SEG_FAILS = ROOT / "outputs" / "r13" / "seg_failures_real.json"
MPAP = ROOT / "data" / "mpap_lookup_gold.json"


def main():
    morph = pd.read_csv(MORPH)
    lab = pd.read_csv(LABELS); pro = pd.read_csv(PROTO)
    df = lab.merge(pro[["case_id", "protocol"]], on="case_id") \
        .merge(morph, on="case_id", how="inner", suffixes=("", "_dup"))
    if "label_dup" in df.columns: df = df.drop(columns=["label_dup"])
    fails = set()
    if SEG_FAILS.exists():
        sf = json.loads(SEG_FAILS.read_text(encoding="utf-8"))
        fails = {r["case_id"] for r in sf.get("real_fails", []) + sf.get("lung_anomaly", [])}
    df = df[~df["case_id"].isin(fails)].copy()
    mpap = json.loads(MPAP.read_text(encoding="utf-8"))
    df["mpap"] = df["case_id"].map(mpap)
    df.loc[df["protocol"].str.lower() == "plain_scan", "mpap"] = 5.0
    df.loc[(df["label"] == 0) & (df["protocol"].str.lower() == "contrast"), "mpap"] = 15.0
    df["stage"] = -1
    df.loc[df["protocol"].str.lower() == "plain_scan", "stage"] = 0
    df.loc[(df["label"] == 0) & (df["protocol"].str.lower() == "contrast"), "stage"] = 1
    df.loc[(df["label"] == 1) & (df["mpap"] < 25), "stage"] = 2
    df.loc[(df["label"] == 1) & (df["mpap"] >= 25) & (df["mpap"] < 35), "stage"] = 3
    df.loc[(df["label"] == 1) & (df["mpap"] >= 35), "stage"] = 4
    df = df[df["stage"] >= 0].copy()

    target_features = [c for c in [
        "artery_tort_p10", "artery_len_p25", "artery_len_p50", "artery_len_mean",
        "artery_total_len_mm", "vein_len_p25", "vein_total_len_mm",
    ] if c in df.columns]

    out = {"target_features": target_features, "subsets": {}}
    for sub_name, sub_df in [("legacy_only", df[df["source_cache"] == "legacy"]),
                                ("new100_only", df[df["source_cache"] == "new100"]),
                                ("enlarged_all", df)]:
        rec = {"n": int(len(sub_df)),
               "stage_counts": {int(k): int(v) for k, v in
                                 sub_df["stage"].value_counts().sort_index().items()},
               "trends": {}}
        for col in target_features:
            sd = sub_df.dropna(subset=[col, "stage"])
            if len(sd) < 30: continue
            stages_present = sd["stage"].nunique()
            if stages_present < 2: continue
            rho, p = spearmanr(sd["stage"], sd[col])
            rec["trends"][col] = {"n": int(len(sd)), "rho": float(rho),
                                   "p": float(p), "stages_present": int(stages_present)}
        out["subsets"][sub_name] = rec

    (OUT / "within_pipeline_trends.json").write_text(json.dumps(out, indent=2),
                                                        encoding="utf-8")

    md = ["# R19.F — Within-pipeline mPAP trends (legacy vs new100 stratified)",
          "",
          "Verifies R19.E confound diagnosis: if legacy-only ρ matches R18.B",
          "(~−0.77 for artery_len_p25), the R19.D/.E extractor is consistent.",
          "If new100-only also has strong trend in same direction, the enlarged",
          "result is artifact of mixing two pipeline-distinct distributions.",
          "",
          "## Per-subset stage counts",
          ""]
    for sub_name, rec in out["subsets"].items():
        md.append(f"**{sub_name}**: n={rec['n']}, stages = "
                  + ", ".join(f"{s}:{n}" for s, n in rec["stage_counts"].items()))
    md += ["",
           "## Spearman ρ — same feature × 3 subsets",
           "",
           "| feature | legacy_only ρ | new100_only ρ | enlarged_all ρ |",
           "|---|---|---|---|"]
    legacy = out["subsets"]["legacy_only"]["trends"]
    new100 = out["subsets"]["new100_only"]["trends"]
    enlarged = out["subsets"]["enlarged_all"]["trends"]
    for col in target_features:
        l_rho = f"{legacy[col]['rho']:+.3f}" if col in legacy else "—"
        n_rho = f"{new100[col]['rho']:+.3f}" if col in new100 else "—"
        e_rho = f"{enlarged[col]['rho']:+.3f}" if col in enlarged else "—"
        md.append(f"| {col} | {l_rho} | {n_rho} | {e_rho} |")
    md += ["",
           "## Interpretation",
           "",
           "If legacy-only ρ matches R18.B (~−0.77 for artery_len_p25, ~+0.63",
           "for paren_std_HU), the R19.D extraction is consistent with R17.",
           "If new100-only ρ has similar magnitude but different sign / scale,",
           "the pipeline confound is real. If new100-only ρ is null/weak, the",
           "Simple_AV_seg masks lack discriminative topology compared to legacy.",
           "",
           "Required next step: HiPaS re-segmentation of new100 (uniform",
           "pipeline) before claiming enlarged-cohort evolution.",
           ""]
    (OUT / "within_pipeline_trends.md").write_text("\n".join(md), encoding="utf-8")
    print("Subset summary:")
    for sub_name, rec in out["subsets"].items():
        print(f"  {sub_name}: n={rec['n']}")
        for col, t in rec["trends"].items():
            print(f"    {col}: ρ={t['rho']:+.3f} p={t['p']:.3g} (n={t['n']})")


if __name__ == "__main__":
    main()
