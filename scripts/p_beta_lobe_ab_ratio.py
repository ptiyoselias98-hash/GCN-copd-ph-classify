"""P-β: Lobe-level Artery:Bronchus ratio hypothesis test.

Medical prior (radiologyassistant.nl): segmental artery diameter > adjacent
bronchus diameter (A:B > 1), especially in UPPER lobes, is a sign of PH.

Strategy: xlsx has per-lobe artery/bronchus volume + length. Use
  diameter_proxy = sqrt(volume / length / pi)   (radius-like)
  A:B = diam_artery / diam_bronchus
Test: Mann-Whitney U (PH vs non-PH), per lobe and upper-vs-lower contrast.
"""
from __future__ import annotations
import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

try:
    from pypinyin import lazy_pinyin
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pypinyin"], check=True)
    from pypinyin import lazy_pinyin

ROOT = Path(__file__).resolve().parent.parent
CLINICAL_XLSX = ROOT / "copd-ph患者113例0331.xlsx"
CLUSTER_CSV   = ROOT / "outputs" / "cluster_topology" / "cluster_assignments.csv"
OUT_DIR       = ROOT / "outputs" / "p_beta_lobe_ab"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df_clin = pd.ExcelFile(CLINICAL_XLSX).parse("Sheet1")
df_clin["name_pinyin"] = df_clin["name"].apply(
    lambda x: "".join(lazy_pinyin(str(x).strip())) if isinstance(x, str) and x.strip() else ""
)

# get label from cluster_assignments (has label column)
ca = pd.read_csv(CLUSTER_CSV)
ca["name_pinyin"] = ca["case_id"].apply(
    lambda s: s.split("_")[1] if len(s.split("_")) >= 2 else ""
)
# join label onto clinical
df = df_clin.merge(ca[["name_pinyin","label"]], on="name_pinyin", how="inner")
df = df.drop_duplicates("name_pinyin")
print(f"Joined clinical×label: {len(df)} cases")
print(f"  PH (label=1): {(df['label']==1).sum()}")
print(f"  non-PH (label=0): {(df['label']==0).sum()}")

LOBES = ["左上肺叶","左下肺叶","右上肺叶","右中肺叶","右下肺叶"]
UPPER = ["左上肺叶","右上肺叶"]
LOWER = ["左下肺叶","右下肺叶"]

def diam_proxy(vol_ml, length_cm):
    """radius-equivalent in cm ≈ sqrt(V / L / pi). Returns NaN where invalid."""
    vol = pd.to_numeric(vol_ml, errors="coerce")
    length = pd.to_numeric(length_cm, errors="coerce")
    ok = (vol > 0) & (length > 0)
    r = np.sqrt(vol / length / np.pi)
    r = r.where(ok)
    return r

# Assumption: pulmonary artery/vein run alongside bronchus (broncho-vascular bundle),
# so their physical length ≈ bronchus length at segmental level.
# Diameter proxy: r ≈ √(V / L_bronchus / π)
rows = []
for lobe in LOBES:
    art_vol_col = f"动脉({lobe})容积(ml)"
    vei_vol_col = f"静脉({lobe})容积(ml)"
    br_vol_col  = f"{lobe}支气管体积(ml)"
    br_len_col  = f"{lobe}支气管长度(cm)"
    missing = [c for c in [art_vol_col, vei_vol_col, br_vol_col, br_len_col] if c not in df.columns]
    if missing:
        print(f"[miss] {lobe}: {missing}")
        continue
    # bronchus diameter proxy
    br_d  = diam_proxy(df[br_vol_col], df[br_len_col])
    # artery / vein diameter proxy assuming co-located with bronchus (L_a ≈ L_b)
    art_d = diam_proxy(df[art_vol_col], df[br_len_col])
    vei_d = diam_proxy(df[vei_vol_col], df[br_len_col])
    df[f"_AB_{lobe}"] = art_d / br_d
    df[f"_VB_{lobe}"] = vei_d / br_d
    # also volume ratio (no length assumption needed)
    art_v = pd.to_numeric(df[art_vol_col], errors="coerce")
    br_v  = pd.to_numeric(df[br_vol_col],  errors="coerce")
    df[f"_AB_vol_{lobe}"] = art_v / br_v.where(br_v > 0)

# ── Per-lobe A:B test (PH vs non-PH)
print("\n=== A:B (artery-to-bronchus diameter proxy) per lobe ===")
print(f"{'lobe':<10s} {'n_PH':>5s} {'n_nonPH':>8s} {'med_PH':>8s} {'med_non':>8s} "
      f"{'mean_PH':>8s} {'mean_non':>8s} {'U':>10s} {'p':>10s} {'AB>1 PH%':>10s} {'AB>1 non%':>10s}")
md_rows = []
for lobe in LOBES:
    c = f"_AB_{lobe}"
    if c not in df.columns: continue
    s_ph = df.loc[df["label"]==1, c].dropna()
    s_no = df.loc[df["label"]==0, c].dropna()
    if len(s_ph) < 5 or len(s_no) < 5: continue
    U, p = stats.mannwhitneyu(s_ph, s_no, alternative="two-sided")
    ab1_ph = float((s_ph > 1.0).mean() * 100)
    ab1_no = float((s_no > 1.0).mean() * 100)
    print(f"{lobe:<10s} {len(s_ph):>5d} {len(s_no):>8d} "
          f"{s_ph.median():>8.3f} {s_no.median():>8.3f} "
          f"{s_ph.mean():>8.3f} {s_no.mean():>8.3f} "
          f"{U:>10.1f} {p:>10.3e} {ab1_ph:>9.1f}% {ab1_no:>9.1f}%")
    md_rows.append((lobe, len(s_ph), len(s_no),
                    s_ph.median(), s_no.median(),
                    s_ph.mean(), s_no.mean(),
                    U, p, ab1_ph, ab1_no))

# ── Upper vs Lower contrast (paired within patient)
print("\n=== Upper-vs-Lower A:B contrast per patient (expected: PH↑ in upper) ===")
upper_cols = [f"_AB_{l}" for l in UPPER if f"_AB_{l}" in df.columns]
lower_cols = [f"_AB_{l}" for l in LOWER if f"_AB_{l}" in df.columns]
df["_AB_upper"] = df[upper_cols].mean(axis=1)
df["_AB_lower"] = df[lower_cols].mean(axis=1)
df["_AB_up_minus_down"] = df["_AB_upper"] - df["_AB_lower"]

for label_val, label_name in [(1,"PH"),(0,"nonPH")]:
    s = df.loc[df["label"]==label_val, "_AB_up_minus_down"].dropna()
    w, p_wil = stats.wilcoxon(s) if len(s) >= 10 else (np.nan, np.nan)
    print(f"  {label_name}: n={len(s)} mean(up-low)={s.mean():.3f} "
          f"median={s.median():.3f} wilcoxon_p={p_wil:.3e}")

s_ph = df.loc[df["label"]==1, "_AB_up_minus_down"].dropna()
s_no = df.loc[df["label"]==0, "_AB_up_minus_down"].dropna()
U2, p2 = stats.mannwhitneyu(s_ph, s_no, alternative="two-sided")
print(f"  PH vs nonPH on (AB_upper − AB_lower): U={U2:.1f}, p={p2:.3e}")

# ── Write CSV + MD
pd.DataFrame(md_rows, columns=["lobe","n_PH","n_nonPH","median_PH","median_nonPH",
                                "mean_PH","mean_nonPH","U","p","pct_AB>1_PH","pct_AB>1_nonPH"]).to_csv(
    OUT_DIR / "ab_ratio_per_lobe.csv", index=False, encoding="utf-8-sig")

md = ["# P-β : Lobe-level Artery:Bronchus ratio (2026-04-21)", ""]
md.append("**Medical prior**: segmental A:B > 1, especially upper lobes, = PH signal.")
md.append(f"**Cases**: PH {(df['label']==1).sum()}, non-PH {(df['label']==0).sum()}")
md.append("**Diameter proxy**: √(V/L/π) (radius-equivalent in cm)")
md.append("")
md.append("## A:B per lobe (Mann-Whitney U, PH vs non-PH)")
md.append("")
md.append("| lobe | n_PH | n_nonPH | med_PH | med_non | mean_PH | mean_non | U | p | %AB>1 PH | %AB>1 non |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|")
for r in md_rows:
    md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.3f} | {r[4]:.3f} | "
              f"{r[5]:.3f} | {r[6]:.3f} | {r[7]:.1f} | {r[8]:.3e} | {r[9]:.1f}% | {r[10]:.1f}% |")
md.append("")
md.append("## Upper−Lower contrast per patient")
md.append("")
md.append("| group | n | mean(up-low) | median | Wilcoxon p |")
md.append("|---|---|---|---|---|")
for label_val, label_name in [(1,"PH"),(0,"nonPH")]:
    s = df.loc[df["label"]==label_val, "_AB_up_minus_down"].dropna()
    w, p_wil = stats.wilcoxon(s) if len(s) >= 10 else (np.nan, np.nan)
    md.append(f"| {label_name} | {len(s)} | {s.mean():.3f} | {s.median():.3f} | {p_wil:.3e} |")
md.append("")
md.append(f"**PH vs nonPH on (AB_upper − AB_lower)**: U={U2:.1f}, p={p2:.3e}  "
          f"({'supports' if p2<0.05 else 'does NOT support'} medical prior at α=0.05)")
md.append("")

(OUT_DIR / "p_beta_summary.md").write_text("\n".join(md), encoding="utf-8")
print(f"\nWrote {OUT_DIR/'p_beta_summary.md'}")
