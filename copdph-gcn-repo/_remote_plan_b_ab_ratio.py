"""Plan B — Artery:Bronchus diameter ratio by Z-tertile (upper / mid / lower
as a cheap anatomical proxy; no true lobe mask on server).

Runs server-side under pulmonary_bv5_py39 conda env. Reads all 269 tri
cache pkls, pairs each artery branch with its nearest airway branch by
spatial NN, computes per-pair A:B diameter ratio, stratifies by within-case
Z-tertile, and tests PH vs non-PH.

Uploads the resulting CSV + summary back to /tmp to fetch locally.
"""
import paramiko

HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"

SCRIPT = r'''
source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39

cat > /tmp/_plan_b_ab.py <<'PY'
from __future__ import annotations
import csv, glob, os, pickle, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, ttest_ind

CACHE = "/home/imss/cw/GCN copdnoph copdph/cache_tri_converted"
LABELS_CSV = "/home/imss/cw/GCN copdnoph copdph/data/labels_expanded_282.csv"
OUT = Path("/tmp/plan_b_out")
OUT.mkdir(exist_ok=True)

# --- labels
labels = {}
with open(LABELS_CSV, newline="") as f:
    r = csv.DictReader(f)
    key = "case_id" if "case_id" in r.fieldnames else "patient_id"
    for row in r:
        labels[row[key]] = int(row["label"])
print(f"labels: {len(labels)}")

# --- per-case compute
MAX_PAIR_MM = 25.0  # anatomical-companion distance cap (mm voxel-space)
rows = []
skipped = 0
for pkl in sorted(glob.glob(os.path.join(CACHE, "*.pkl"))):
    case_id = Path(pkl).stem.replace("_tri", "")
    if case_id not in labels:
        continue
    try:
        with open(pkl, "rb") as f:
            d = pickle.load(f)
    except Exception as e:
        print("pkl fail:", case_id, e); skipped += 1; continue
    art = d.get("artery"); air = d.get("airway")
    if art is None or air is None:
        skipped += 1; continue
    def _pos_of(g):
        # graph.x cols 7..9 are centroid (voxel coords); pos may be None
        if getattr(g, "pos", None) is not None:
            return g.pos.numpy().astype(float)
        if g.x is not None and g.x.size(1) >= 10:
            return g.x[:, 7:10].numpy().astype(float)
        return None
    ap = _pos_of(art); ip = _pos_of(air)
    if ap is None or ip is None or len(ap) == 0 or len(ip) == 0:
        skipped += 1; continue
    ad = art.x[:, 0].numpy().astype(float)  # artery diameters
    id_ = air.x[:, 0].numpy().astype(float)  # airway diameters

    # filter NaN/inf rows
    a_ok = np.isfinite(ap).all(axis=1) & np.isfinite(ad)
    i_ok = np.isfinite(ip).all(axis=1) & np.isfinite(id_)
    if a_ok.sum() == 0 or i_ok.sum() == 0:
        skipped += 1; continue
    ap = ap[a_ok]; ad = ad[a_ok]
    ip = ip[i_ok]; id_ = id_[i_ok]
    # spacing to convert voxel distance to mm (sanitize NaN/inf -> 1.0)
    sp = d.get("spacing", (1.0, 1.0, 1.0))
    sp = np.array(sp, dtype=float).reshape(-1)
    if sp.size != 3:
        sp = np.array([1.0, 1.0, 1.0], dtype=float)
    sp = np.where(np.isfinite(sp) & (sp > 0), sp, 1.0)
    ap_mm = ap * sp
    ip_mm = ip * sp

    # Re-filter after multiply in case of any residual non-finite
    fa = np.isfinite(ap_mm).all(axis=1)
    fi = np.isfinite(ip_mm).all(axis=1)
    if fa.sum() == 0 or fi.sum() == 0:
        skipped += 1; continue
    ap_mm = ap_mm[fa]; ad = ad[fa]
    ip_mm = ip_mm[fi]; id_ = id_[fi]
    # keep ap (voxel) aligned with ap_mm for Z extraction later
    ap = ap[fa]

    # NN pairing: for each artery branch, find nearest airway branch
    tree = cKDTree(ip_mm)
    dists, idxs = tree.query(ap_mm, k=1)
    mask = (dists <= MAX_PAIR_MM) & (id_[idxs] > 0) & (ad > 0)
    if mask.sum() == 0:
        skipped += 1; continue

    paired_art_d = ad[mask]
    paired_air_d = id_[idxs[mask]]
    paired_art_z = ap[mask, 2]
    ratio = paired_art_d / paired_air_d

    # Z-tertile within case (upper = largest Z in HFS orientation, but
    # cannot assume orientation, so just use "high-Z" vs "low-Z" ≡
    # top-third vs bottom-third as anatomy-agnostic proxy).
    z = paired_art_z
    q1, q2 = np.quantile(z, [1/3, 2/3])
    upper = z >= q2
    lower = z <= q1
    middle = ~(upper | lower)

    def _agg(sel):
        if sel.sum() == 0:
            return np.nan, np.nan, 0
        r = ratio[sel]
        return float(r.mean()), float((r > 1.0).mean()), int(sel.sum())

    u_mean, u_frac_gt1, u_n = _agg(upper)
    m_mean, m_frac_gt1, m_n = _agg(middle)
    l_mean, l_frac_gt1, l_n = _agg(lower)
    all_mean, all_frac_gt1, all_n = _agg(np.ones_like(z, dtype=bool))

    rows.append({
        "case_id": case_id,
        "label": labels[case_id],
        "n_pairs": int(mask.sum()),
        "n_artery_nodes": int(ap.shape[0]),
        "n_airway_nodes": int(ip.shape[0]),
        "pair_rate": float(mask.sum()) / ap.shape[0],
        "upper_AB_mean": u_mean, "upper_frac_gt1": u_frac_gt1, "upper_n": u_n,
        "middle_AB_mean": m_mean, "middle_frac_gt1": m_frac_gt1, "middle_n": m_n,
        "lower_AB_mean": l_mean, "lower_frac_gt1": l_frac_gt1, "lower_n": l_n,
        "all_AB_mean": all_mean, "all_frac_gt1": all_frac_gt1, "all_n": all_n,
    })

print(f"cases computed: {len(rows)}  skipped: {skipped}")

# --- write per-case CSV
csv_p = OUT / "plan_b_per_case.csv"
with open(csv_p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"wrote {csv_p}")

# --- group stats
import statistics as stat
def _col(key):
    return np.array([r[key] for r in rows], dtype=float)
lab = _col("label")
ph_m = lab == 1
nonph_m = lab == 0

summary = []
for col, desc in [
    ("upper_AB_mean",    "upper-tertile A:B mean"),
    ("middle_AB_mean",   "middle-tertile A:B mean"),
    ("lower_AB_mean",    "lower-tertile A:B mean"),
    ("upper_frac_gt1",   "upper-tertile fraction A:B>1"),
    ("middle_frac_gt1",  "middle-tertile fraction A:B>1"),
    ("lower_frac_gt1",   "lower-tertile fraction A:B>1"),
    ("all_AB_mean",      "all-pair A:B mean"),
    ("all_frac_gt1",     "all-pair fraction A:B>1"),
    ("pair_rate",        "artery-airway pair rate"),
]:
    v = _col(col)
    v_ph = v[ph_m]; v_ph = v_ph[~np.isnan(v_ph)]
    v_non = v[nonph_m]; v_non = v_non[~np.isnan(v_non)]
    if len(v_ph) < 3 or len(v_non) < 3:
        continue
    try:
        u, p_u = mannwhitneyu(v_ph, v_non, alternative="two-sided")
    except Exception:
        p_u = np.nan
    t, p_t = ttest_ind(v_ph, v_non, equal_var=False, nan_policy="omit")
    summary.append({
        "feature": col, "desc": desc,
        "ph_mean": float(np.mean(v_ph)), "ph_std": float(np.std(v_ph)), "ph_n": int(len(v_ph)),
        "nonph_mean": float(np.mean(v_non)), "nonph_std": float(np.std(v_non)), "nonph_n": int(len(v_non)),
        "diff": float(np.mean(v_ph) - np.mean(v_non)),
        "mann_whitney_p": float(p_u) if not np.isnan(p_u) else None,
        "welch_p": float(p_t),
    })

sum_p = OUT / "plan_b_summary.csv"
with open(sum_p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader(); w.writerows(summary)
print(f"wrote {sum_p}")

# Print quick table
print()
print(f"{'feature':30s} {'PH mean±std':22s} {'nonPH mean±std':22s} {'diff':>8s} {'MW-p':>10s}")
for s in summary:
    ph = f"{s['ph_mean']:.3f}±{s['ph_std']:.3f} (n={s['ph_n']})"
    no = f"{s['nonph_mean']:.3f}±{s['nonph_std']:.3f} (n={s['nonph_n']})"
    p = s['mann_whitney_p']
    p_str = f"{p:.2e}" if p is not None else "N/A"
    print(f"{s['feature']:30s} {ph:22s} {no:22s} {s['diff']:+.3f}   {p_str}")
PY

python /tmp/_plan_b_ab.py 2>&1 | tee /tmp/_plan_b_run.log
echo "---"
ls -la /tmp/plan_b_out/
'''

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
_, o, e = c.exec_command(SCRIPT, timeout=300)
print(o.read().decode(errors="replace"))
err = e.read().decode(errors="replace")
if err.strip():
    print("--- STDERR (first 2000) ---")
    print(err[:2000])

# Download result CSVs
sftp = c.open_sftp()
from pathlib import Path as P
local_out = P(r"E:\桌面文件\图卷积-肺小血管演化规律探索\outputs\p_zeta_cluster_269\plan_b")
local_out.mkdir(parents=True, exist_ok=True)
for fn in ["plan_b_per_case.csv", "plan_b_summary.csv"]:
    try:
        sftp.get(f"/tmp/plan_b_out/{fn}", str(local_out / fn))
        print(f"downloaded {fn}")
    except IOError as ex:
        print(f"skip {fn}: {ex}")
sftp.close()
c.close()
