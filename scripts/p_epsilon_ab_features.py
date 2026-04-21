"""P-ε v1: test if adding P-β lobe A:B features boosts classification over radiomics alone.

Design:
  Features set A: 45 commercial radiomics (current baseline)
  Features set B: Set A + 6 lobe-level A:B features (5 per-lobe + upper-lower diff)

Cross-validated (5-fold × 3-rep, stratified) logistic regression & random forest.
Compare AUC/Accuracy/Sens/Spec/F1/Precision.
"""
from __future__ import annotations
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    from pypinyin import lazy_pinyin
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pypinyin"], check=True)
    from pypinyin import lazy_pinyin

ROOT = Path(__file__).resolve().parent.parent
RAD_CSV       = ROOT / "data" / "copd_ph_radiomics.csv"
CLINICAL_XLSX = ROOT / "copd-ph患者113例0331.xlsx"
CLUSTER_CSV   = ROOT / "outputs" / "cluster_topology" / "cluster_assignments.csv"
OUT_DIR       = ROOT / "outputs" / "p_epsilon_ab_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load & align
rad = pd.read_csv(RAD_CSV)
clin = pd.ExcelFile(CLINICAL_XLSX).parse("Sheet1")
ca = pd.read_csv(CLUSTER_CSV)

# Normalize pinyin keys
def chn2py(x):
    if not isinstance(x, str) or not x.strip(): return ""
    return "".join(lazy_pinyin(x.strip())).lower()

def case2py(s):
    parts = str(s).split("_")
    return parts[1].lower() if len(parts) >= 2 else ""

rad["key"]  = rad["patient_id"].apply(lambda s: str(s).lower())
clin["key"] = clin["name"].apply(chn2py)
ca["key"]   = ca["case_id"].apply(case2py)

print(f"radiomics: {len(rad)} cases, uniq keys={rad['key'].nunique()}")
print(f"clinical:  {len(clin)} cases")
print(f"cluster:   {len(ca)} cases (has true label)")

# ── Compute A:B features from clinical xlsx
LOBES = ["左上肺叶","左下肺叶","右上肺叶","右中肺叶","右下肺叶"]
UPPER = ["左上肺叶","右上肺叶"]
LOWER = ["左下肺叶","右下肺叶"]
def diam_proxy(vol_ml, length_cm):
    vol = pd.to_numeric(vol_ml, errors="coerce")
    length = pd.to_numeric(length_cm, errors="coerce")
    ok = (vol > 0) & (length > 0)
    r = np.sqrt(vol / length / np.pi)
    return r.where(ok)

ab_cols = []
for lobe in LOBES:
    art_v = pd.to_numeric(clin.get(f"动脉({lobe})容积(ml)"), errors="coerce")
    br_v  = pd.to_numeric(clin.get(f"{lobe}支气管体积(ml)"), errors="coerce")
    br_l  = pd.to_numeric(clin.get(f"{lobe}支气管长度(cm)"), errors="coerce")
    art_d = diam_proxy(art_v, br_l)   # assume L_artery ≈ L_bronchus
    br_d  = diam_proxy(br_v,  br_l)
    col = f"AB_{lobe}"
    clin[col] = (art_d / br_d).astype(float)
    ab_cols.append(col)
clin["AB_upper"] = clin[[f"AB_{l}" for l in UPPER]].mean(axis=1)
clin["AB_lower"] = clin[[f"AB_{l}" for l in LOWER]].mean(axis=1)
clin["AB_upper_minus_lower"] = clin["AB_upper"] - clin["AB_lower"]
AB_ALL = ab_cols + ["AB_upper_minus_lower"]

# ── Merge cluster (label source) ← radiomics ← clinical A:B
# label priority: cluster > radiomics > clinical
base = rad[["key","label"] + [c for c in rad.columns if c not in ("patient_id","label","key")]].copy()
# attach AB features from clinical
ab_df = clin[["key"] + AB_ALL].copy().drop_duplicates("key")
merged = base.merge(ab_df, on="key", how="inner")
print(f"\nMerged radiomics×clinical: {len(merged)} rows, PH={int((merged['label']==1).sum())}, nonPH={int((merged['label']==0).sum())}")

# Separate feature blocks
rad_cols = [c for c in base.columns if c not in ("key","label")]
# drop non-numeric from rad
num_rad = []
for c in rad_cols:
    if pd.to_numeric(merged[c], errors="coerce").notna().mean() > 0.9:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")
        num_rad.append(c)
print(f"Numeric radiomics feats: {len(num_rad)}")
print(f"A:B feats: {len(AB_ALL)}")

# Drop rows with NaN in used cols
X_rad    = merged[num_rad].fillna(0.0).values
X_ab     = merged[AB_ALL].fillna(0.0).values
X_all    = np.hstack([X_rad, X_ab])
y        = merged["label"].astype(int).values

# ── CV eval
def eval_cv(X, y, name, n_rep=3, n_splits=5, seed=42):
    rows = []
    for rep in range(n_rep):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed+rep)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            for clf_name, clf in [
                ("logreg", Pipeline([("s",StandardScaler()), ("m",LogisticRegression(max_iter=2000, C=1.0))])),
                ("rf", RandomForestClassifier(n_estimators=300, random_state=seed+rep, n_jobs=-1)),
            ]:
                clf.fit(X[tr], y[tr])
                p = clf.predict_proba(X[te])[:,1]
                yhat = (p >= 0.5).astype(int)
                tn, fp, fn, tp = confusion_matrix(y[te], yhat, labels=[0,1]).ravel()
                rows.append(dict(
                    feats=name, clf=clf_name, rep=rep, fold=fold,
                    AUC=roc_auc_score(y[te], p),
                    Accuracy=accuracy_score(y[te], yhat),
                    Sensitivity=tp/(tp+fn+1e-9),
                    Specificity=tn/(tn+fp+1e-9),
                    Precision=tp/(tp+fp+1e-9),
                    F1=f1_score(y[te], yhat, zero_division=0),
                ))
    return pd.DataFrame(rows)

res_A = eval_cv(X_rad,  y, "radiomics_only")
res_B = eval_cv(X_ab,   y, "AB_only")
res_C = eval_cv(X_all,  y, "radiomics+AB")
all_res = pd.concat([res_A, res_B, res_C], ignore_index=True)
all_res.to_csv(OUT_DIR/"ab_features_cv_raw.csv", index=False, encoding="utf-8-sig")

# Aggregate
agg = all_res.groupby(["feats","clf"]).agg(
    AUC=("AUC","mean"), AUC_std=("AUC","std"),
    Accuracy=("Accuracy","mean"), Sensitivity=("Sensitivity","mean"),
    Specificity=("Specificity","mean"), Precision=("Precision","mean"), F1=("F1","mean"),
).round(4).reset_index()
print("\n=== AUC/Accuracy/Sens/Spec/F1 by feature-set × classifier ===")
print(agg.to_string(index=False))
agg.to_csv(OUT_DIR/"ab_features_cv_agg.csv", index=False, encoding="utf-8-sig")

# Markdown summary
md = [f"# P-ε v1 : A:B feature boost test ({pd.Timestamp.today().date()})", ""]
md.append(f"**n = {len(merged)}** (radiomics×clinical intersection, PH={int((merged['label']==1).sum())}, nonPH={int((merged['label']==0).sum())})")
md.append(f"**Feature blocks**: radiomics={len(num_rad)}, A:B={len(AB_ALL)}  ·  CV: 5-fold × 3-rep stratified")
md.append("")
md.append("| feats | clf | AUC | AUC_std | Accuracy | Sens | Spec | Prec | F1 |")
md.append("|---|---|---|---|---|---|---|---|---|")
for _, r in agg.iterrows():
    md.append(f"| {r['feats']} | {r['clf']} | {r['AUC']:.3f} | {r['AUC_std']:.3f} | "
              f"{r['Accuracy']:.3f} | {r['Sensitivity']:.3f} | {r['Specificity']:.3f} | "
              f"{r['Precision']:.3f} | {r['F1']:.3f} |")
md.append("")

# ΔAUC analysis
pivot_auc = agg.pivot(index="clf", columns="feats", values="AUC")
if "radiomics_only" in pivot_auc.columns and "radiomics+AB" in pivot_auc.columns:
    delta = pivot_auc["radiomics+AB"] - pivot_auc["radiomics_only"]
    md.append("## ΔAUC from adding A:B features")
    md.append("")
    md.append("| clf | radiomics_only | +A:B | ΔAUC |")
    md.append("|---|---|---|---|")
    for clf_name in pivot_auc.index:
        md.append(f"| {clf_name} | {pivot_auc.loc[clf_name,'radiomics_only']:.3f} | "
                  f"{pivot_auc.loc[clf_name,'radiomics+AB']:.3f} | {delta[clf_name]:+.3f} |")
    md.append("")
    verdict = "boosts" if (delta > 0).all() else ("mixed" if (delta > 0).any() else "does not boost")
    md.append(f"**Verdict**: adding P-β A:B features {verdict} AUC over radiomics alone.")

(OUT_DIR/"p_epsilon_summary.md").write_text("\n".join(md), encoding="utf-8")
print(f"\nWrote {OUT_DIR/'p_epsilon_summary.md'}")
