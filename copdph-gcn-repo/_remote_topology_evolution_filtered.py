"""Topology-evolution — filtered re-run after discovering that the raw
n=269 clustering was dominated by segmentation completeness (cluster 1: 23
airway-only cases with 0% PH; cluster 2: 57 vessel-only cases with 3.5% PH).

This re-run keeps only cases where ALL THREE structures have a non-trivial
tree (each with n_nodes >= 20 for artery/vein and >= 5 for airway). On that
"clean-segmentation" sub-cohort we re-run A + B + C (2-GAE-seed ensemble)
to ask the *actual* question: **on cases with full segmentation, does
unsupervised topology alone still separate PH from non-PH?**

Uses all artifacts already produced by _remote_topology_evolution.py.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

ROOT = Path(r"E:\桌面文件\图卷积-肺小血管演化规律探索\outputs\p_zeta_cluster_269\topology_evolution")
SEED = 42

# ---- load raw graph stats so we can build a completeness mask
B      = np.load(ROOT / "expB_stats_features.npz")
X_stat = B["X_raw"]        # per-case, 57 features (19 per structure)
case_ids = B["case_ids"]
y        = B["y"]

# Column indices (see list in _remote_topology_evolution.py):
# per structure 19 columns starting at: artery 0, vein 19, airway 38
# first col is n_nodes for each structure
ART_N  = X_stat[:, 0]
VEIN_N = X_stat[:, 19]
AIR_N  = X_stat[:, 38]
mask = (ART_N >= 20) & (VEIN_N >= 20) & (AIR_N >= 5)
print(f"[filtered] kept {int(mask.sum())} / {len(mask)} cases")
print(f"           PH:    {int(((y == 1) & mask).sum())} / {int((y == 1).sum())}")
print(f"           nonPH: {int(((y == 0) & mask).sum())} / {int((y == 0).sum())}")

# ---- pull features, sub-select, rerun sweeps
A   = np.load(ROOT / "expA_wl_features.npz")
C   = np.load(ROOT / "expC_gae_embeddings.npz")
X_wl  = A["X"][mask]
X_sta = StandardScaler().fit_transform(X_stat[mask])
X_gae = C["X"][mask]
y_f   = y[mask]
cid_f = case_ids[mask]

def sweep(X, tag):
    rows = []
    for k in (2, 3, 4):
        for method, fn in [
            ("kmeans",   lambda kk: KMeans(n_clusters=kk, n_init=20, random_state=SEED).fit_predict(X)),
            ("spectral", lambda kk: SpectralClustering(n_clusters=kk, n_init=20, random_state=SEED,
                                                       affinity="nearest_neighbors", n_neighbors=10,
                                                       assign_labels="kmeans").fit_predict(X)),
        ]:
            try:
                lab = fn(k)
            except Exception as e:
                print(f"[{tag}] {method} k={k} fail: {e}"); continue
            sizes = np.bincount(lab).tolist()
            ari = float(adjusted_rand_score(y_f, lab))
            nmi = float(normalized_mutual_info_score(y_f, lab))
            try:
                sil = float(silhouette_score(X, lab)) if len(set(lab)) > 1 else float("nan")
            except Exception:
                sil = float("nan")
            # per-cluster PH rates
            ph_rates = [float(y_f[lab == ci].mean()) if (lab == ci).sum() else float("nan")
                        for ci in range(k)]
            rows.append({"experiment": tag, "method": method, "k": k,
                         "ARI_vs_PH": ari, "NMI_vs_PH": nmi, "silhouette": sil,
                         "sizes": sizes, "ph_rates": ph_rates})
    return rows

all_rows = []
all_rows += sweep(X_wl,  "A_WL")
all_rows += sweep(X_sta, "B_stats")
all_rows += sweep(X_gae, "C_GAE")

print()
print(f"{'exp':10s} {'method':10s} {'k':>2s} {'ARI':>7s} {'NMI':>7s} {'sil':>7s}  sizes   PH-rates")
for r in all_rows:
    rts = " ".join(f"{p:.2f}" for p in r["ph_rates"])
    print(f"{r['experiment']:10s} {r['method']:10s} {r['k']:>2d} "
          f"{r['ARI_vs_PH']:>7.3f} {r['NMI_vs_PH']:>7.3f} {r['silhouette']:>7.3f}  "
          f"{r['sizes']}   [{rts}]")

best = max(all_rows, key=lambda r: r["ARI_vs_PH"])
print(f"\n[filtered] best: {best}")

out = {
    "n_total": int(len(y)),
    "n_kept":  int(mask.sum()),
    "filter":  "artery_n>=20 AND vein_n>=20 AND airway_n>=5",
    "label_counts_kept": [int((y_f == 0).sum()), int((y_f == 1).sum())],
    "rows": all_rows,
    "best": best,
}
with open(ROOT / "topo_summary_filtered.json", "w") as f:
    json.dump(out, f, indent=2)

# Per-cluster topology profile for the filtered best
if best["method"] == "kmeans":
    lab = KMeans(n_clusters=best["k"], n_init=20, random_state=SEED).fit_predict(
        {"A_WL": X_wl, "B_stats": X_sta, "C_GAE": X_gae}[best["experiment"]])
else:
    lab = SpectralClustering(n_clusters=best["k"], n_init=20, random_state=SEED,
                              affinity="nearest_neighbors", n_neighbors=10,
                              assign_labels="kmeans").fit_predict(
        {"A_WL": X_wl, "B_stats": X_sta, "C_GAE": X_gae}[best["experiment"]])

feat_names = []
for s in ("artery", "vein", "airway"):
    for n in ["n_nodes","n_edges","density","mean_deg","max_deg","std_deg",
             "n_leaves","n_bifurc","mean_diam","std_diam","max_diam","p90_diam",
             "mean_len","sum_len","mean_tort","std_tort","max_strahler",
             "mean_strahler","n_strahler_ge3"]:
        feat_names.append(f"{s}_{n}")
X_raw_f = X_stat[mask]
prof = []
for ci in range(best["k"]):
    m = lab == ci
    row = {"cluster": int(ci), "n": int(m.sum()),
           "ph_rate": float(y_f[m].mean()) if m.sum() else float("nan")}
    for j, fn in enumerate(feat_names):
        row[fn] = float(np.nanmean(X_raw_f[m, j])) if m.sum() else float("nan")
    prof.append(row)

with open(ROOT / "topo_best_cluster_profile_filtered.json", "w") as f:
    json.dump({"best": best, "profile": prof, "feat_names": feat_names,
               "assignments": lab.tolist(),
               "case_ids": [str(x) for x in cid_f.tolist()],
               "y": [int(v) for v in y_f.tolist()]}, f, indent=2)

# Top-30 feature deltas
import csv
with open(ROOT / "topo_best_cluster_diffs_filtered.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["feature"] + [f"cluster_{i}" for i in range(best["k"])] + ["delta"])
    diffs = []
    for j, fn in enumerate(feat_names):
        vals = [prof[c][fn] for c in range(best["k"])]
        diffs.append((fn, vals, max(vals) - min(vals)))
    diffs.sort(key=lambda t: -abs(t[2]))
    for fn, vals, d in diffs[:30]:
        w.writerow([fn] + [f"{v:.3f}" for v in vals] + [f"{d:+.3f}"])

print("\nwrote:")
print("  ", ROOT / "topo_summary_filtered.json")
print("  ", ROOT / "topo_best_cluster_profile_filtered.json")
print("  ", ROOT / "topo_best_cluster_diffs_filtered.csv")
