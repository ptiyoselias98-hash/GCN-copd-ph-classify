"""R24.0 — Locked cohort/analysis table prerequisite for all R24 sub-rounds.

Output: outputs/r24/cohort_locked_table.csv with columns:
  case_id, label, protocol, is_contrast_only_subset, measured_mpap,
  measured_mpap_flag, stage_default_used, fold_id, feature_source,
  exclusion_reason

Stratified 5-fold split on (label × protocol) within-contrast subset.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).parent.parent.parent
MORPH = ROOT / "outputs" / "r20" / "morph_unified301.csv"
LABELS = ROOT / "data" / "labels_extended_382.csv"
PROTO = ROOT / "data" / "case_protocol_extended.csv"
MPAP = ROOT / "data" / "mpap_lookup_gold.json"
OUT = ROOT / "outputs" / "r24"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    morph = pd.read_csv(MORPH)
    lab = pd.read_csv(LABELS)
    pro = pd.read_csv(PROTO)
    mp = json.loads(MPAP.read_text(encoding="utf-8"))

    df = morph[["case_id"]].merge(lab, on="case_id", how="left")
    df = df.merge(pro[["case_id", "protocol"]], on="case_id", how="left")
    df["protocol"] = df["protocol"].str.lower().fillna("unknown")
    df["is_contrast_only_subset"] = (df["protocol"] == "contrast")
    df["measured_mpap"] = df["case_id"].map(mp).astype(float)
    df["measured_mpap_flag"] = df["measured_mpap"].notna()
    plain_mask = df["protocol"] == "plain_scan"
    contrast_nonph_mask = (df["protocol"] == "contrast") & (df["label"] == 0)
    df["stage_default_used"] = np.nan
    df.loc[plain_mask & ~df["measured_mpap_flag"], "stage_default_used"] = 5.0
    df.loc[contrast_nonph_mask & ~df["measured_mpap_flag"], "stage_default_used"] = 15.0
    df["feature_source"] = "Simple_AV_seg_unified"
    df["exclusion_reason"] = ""

    # Stratified 5-fold within contrast (n=190)
    df["fold_id"] = -1
    contrast_idx = df.index[df["is_contrast_only_subset"]].tolist()
    if contrast_idx:
        cdf = df.loc[contrast_idx].reset_index(drop=False)
        y_strata = cdf["label"].astype(str)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold_i, (_, va) in enumerate(skf.split(cdf, y_strata), 1):
            df.loc[cdf.loc[va, "index"].values, "fold_id"] = fold_i

    cols = ["case_id", "label", "protocol", "is_contrast_only_subset",
            "measured_mpap", "measured_mpap_flag", "stage_default_used",
            "fold_id", "feature_source", "exclusion_reason"]
    df = df[cols]
    out_path = OUT / "cohort_locked_table.csv"
    df.to_csv(out_path, index=False)

    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    summary = {
        "n_total": int(len(df)),
        "n_contrast": int(df["is_contrast_only_subset"].sum()),
        "n_plain_scan": int((df["protocol"] == "plain_scan").sum()),
        "n_PH": int((df["label"] == 1).sum()),
        "n_nonPH": int((df["label"] == 0).sum()),
        "n_within_contrast_PH": int(((df["label"] == 1) & df["is_contrast_only_subset"]).sum()),
        "n_within_contrast_nonPH": int(((df["label"] == 0) & df["is_contrast_only_subset"]).sum()),
        "n_measured_mpap": int(df["measured_mpap_flag"].sum()),
        "fold_distribution": {f"fold_{i}": int((df["fold_id"]==i).sum()) for i in range(1,6)},
        "sha256": sha,
    }
    (OUT / "cohort_locked_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nlocked: {out_path}")
    print(f"SHA256: {sha}")


if __name__ == "__main__":
    main()
