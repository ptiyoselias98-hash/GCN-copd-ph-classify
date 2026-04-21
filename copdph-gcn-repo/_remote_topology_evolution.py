"""Topology-evolution experiments A + B + C on n=269 tri_structure cohort
(all label-free clustering; PH label used only for external validation).

Parallelism layout on the remote dual-GPU server:
  Phase 1 — 3 jobs run in parallel:
    • CPU  : WL subtree-hash kernel + graph statistics (joblib n_jobs=-1)
    • GPU0 : GAE self-supervised training, seed 42
    • GPU1 : GAE self-supervised training, seed 43     (ensemble for stability)
  Phase 2 — single job:
    • merge both GAE seeds (average per-case embeddings),
      run KMeans/Spectral at k=2..4 on A/B/C, report ARI/NMI vs PH.

A) WL (Weisfeiler-Lehman) graph kernel — label-free topology signature.
B) Graph statistics — radiomics-style baseline (for comparison only).
C) Graph Autoencoder (GAE) — self-supervised, no label, dual-seed ensemble.
"""
import paramiko

HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"

# NB: the scripts below are written into /tmp on the server, then launched
# in parallel. After both GPU jobs + the CPU job finish, the cluster step
# runs once and aggregates. Keeping each job in its own file simplifies
# process isolation and lets bash's `wait` manage completion.

SCRIPT = r'''
set -e
source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39
mkdir -p /tmp/topo_out

# ─────────────────────── Common loader ───────────────────────
cat > /tmp/_topo_loader.py <<'PY'
import csv, glob, os, pickle
from pathlib import Path
import torch

CACHE      = "/home/imss/cw/GCN copdnoph copdph/cache_tri_converted"
LABELS_CSV = "/home/imss/cw/GCN copdnoph copdph/data/labels_expanded_282.csv"

def load_cases():
    labels = {}
    with open(LABELS_CSV, newline="") as f:
        r = csv.DictReader(f)
        key = "case_id" if "case_id" in r.fieldnames else "patient_id"
        for row in r:
            labels[row[key]] = int(row["label"])
    out = []
    for pkl in sorted(glob.glob(os.path.join(CACHE, "*.pkl"))):
        cid = Path(pkl).stem.replace("_tri", "")
        if cid not in labels:
            continue
        try:
            with open(pkl, "rb") as f:
                d = pickle.load(f)
        except Exception:
            continue
        if not all(k in d and d[k] is not None for k in ("artery", "vein", "airway")):
            continue
        if any(getattr(d[k], "x", None) is None or d[k].x.shape[0] == 0
               for k in ("artery", "vein", "airway")):
            continue
        out.append((cid, d, labels[cid]))
    return out
PY

# ─────────────────────── Job 1: WL + stats on CPU (joblib) ───────────────────────
cat > /tmp/_topo_wlstat.py <<'PY'
from __future__ import annotations
import hashlib, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, "/tmp")
from _topo_loader import load_cases

OUT = Path("/tmp/topo_out"); OUT.mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)

cases = load_cases()
print(f"[wlstat] loaded {len(cases)} cases", flush=True)
case_ids = np.array([c[0] for c in cases])
y        = np.array([c[2] for c in cases], dtype=int)

# ----- WL refinement per case (CPU parallel)
def _wl_one(cid, d, T=3, n_diam_bins=6):
    graphs = {}
    for k in ("artery", "vein", "airway"):
        x  = d[k].x.float()
        ei = d[k].edge_index if d[k].edge_index is not None else torch.zeros((2, 0), dtype=torch.long)
        graphs[k] = (x, ei)
    all_init, adj = {}, {}
    for name, (x, ei) in graphs.items():
        N = x.shape[0]
        diam = x[:, 0].numpy().astype(float)
        diam = np.nan_to_num(diam, nan=np.nanmedian(diam) if np.isfinite(np.nanmedian(diam)) else 1.0)
        qs = np.quantile(diam, np.linspace(0, 1, n_diam_bins + 1)[1:-1])
        bins = np.digitize(diam, qs)
        ei_np = ei.numpy()
        deg = np.zeros(N, dtype=int)
        if ei_np.size > 0:
            for u in ei_np[0]:
                deg[u] += 1
        all_init[name] = [f"{name}|deg{min(deg[i], 8)}|d{bins[i]}" for i in range(N)]
        a = defaultdict(list)
        for u, v in zip(ei_np[0], ei_np[1]):
            a[int(u)].append(int(v))
        adj[name] = a
    bag = Counter()
    cur = {n: list(all_init[n]) for n in all_init}
    for n in cur:
        for lbl in cur[n]:
            bag[lbl] += 1
    for t in range(T):
        nxt = {}
        for name, c in cur.items():
            a = adj[name]
            ns = []
            for i, lbl in enumerate(c):
                nbr = sorted(c[j] for j in a[i])
                ns.append("t{}:{}".format(t+1, hashlib.md5(("|".join([lbl] + nbr)).encode()).hexdigest()[:16]))
            nxt[name] = ns
        cur = nxt
        for n in cur:
            for lbl in cur[n]:
                bag[lbl] += 1
    return bag

t0 = time.time()
case_bags = Parallel(n_jobs=-1, backend="loky", verbose=1)(
    delayed(_wl_one)(cid, d) for cid, d, _ in cases
)
print(f"[wlstat] WL bags in {time.time()-t0:.1f}s", flush=True)

# vocab top-K
global_bag = Counter()
for b in case_bags:
    global_bag.update(b)
VOCAB_K = 2048
vocab   = [w for w, _ in global_bag.most_common(VOCAB_K)]
vocab_i = {w: i for i, w in enumerate(vocab)}
X_wl    = np.zeros((len(case_bags), len(vocab)), dtype=np.float32)
for ci, b in enumerate(case_bags):
    for w, c in b.items():
        if w in vocab_i:
            X_wl[ci, vocab_i[w]] = c
norms = np.linalg.norm(X_wl, axis=1, keepdims=True); norms[norms == 0] = 1
X_wl  = X_wl / norms
svd   = TruncatedSVD(n_components=min(64, X_wl.shape[1]-1, X_wl.shape[0]-1), random_state=SEED)
X_wl64 = svd.fit_transform(X_wl)
print(f"[wlstat] WL feat {X_wl.shape} -> SVD {X_wl64.shape}", flush=True)
np.savez(OUT / "expA_wl_features.npz", X=X_wl64, case_ids=case_ids, y=y)

# ----- Graph statistics per case (CPU parallel)
def _stats_one(cid, d):
    def _safe(v, op):
        v = np.asarray(v, dtype=float); v = v[np.isfinite(v)]
        if len(v) == 0: return 0.0
        return float(op(v))
    feats = []
    for name in ("artery", "vein", "airway"):
        x  = d[name].x.float()
        ei = d[name].edge_index if d[name].edge_index is not None else torch.zeros((2, 0), dtype=torch.long)
        N  = x.shape[0]
        diam     = x[:, 0].numpy().astype(float)
        length   = x[:, 1].numpy().astype(float) if x.shape[1] >= 2 else np.zeros(N)
        tort     = x[:, 2].numpy().astype(float) if x.shape[1] >= 3 else np.zeros(N)
        strahler = x[:, 10].numpy().astype(float) if x.shape[1] >= 11 else np.zeros(N)
        ei_np = ei.numpy()
        E = ei_np.shape[1]
        deg = np.zeros(N, dtype=int)
        if E > 0:
            for u in ei_np[0]:
                deg[u] += 1
        feats += [
            float(N), float(E), float(E) / max(N, 1),
            _safe(deg, np.mean), _safe(deg, np.max), _safe(deg, np.std),
            float((deg == 1).sum()), float((deg >= 3).sum()),
            _safe(diam, np.mean), _safe(diam, np.std), _safe(diam, np.max),
            _safe(diam, lambda v: np.quantile(v, 0.9)),
            _safe(length, np.mean), _safe(length, np.sum),
            _safe(tort, np.mean), _safe(tort, np.std),
            _safe(strahler, np.max), _safe(strahler, np.mean),
            float((strahler >= 3).sum()),
        ]
    return feats

t0 = time.time()
rows = Parallel(n_jobs=-1, backend="loky", verbose=1)(
    delayed(_stats_one)(cid, d) for cid, d, _ in cases
)
print(f"[wlstat] graph stats in {time.time()-t0:.1f}s", flush=True)
X_stat      = np.nan_to_num(np.asarray(rows, dtype=np.float32))
X_stat_std  = StandardScaler().fit_transform(X_stat)
np.savez(OUT / "expB_stats_features.npz",
         X=X_stat_std, X_raw=X_stat, case_ids=case_ids, y=y)
print(f"[wlstat] stats feat {X_stat_std.shape}  ->  OK", flush=True)
PY

# ─────────────────────── Job 2: GAE worker (one seed on one GPU) ───────────────────────
cat > /tmp/_topo_gae.py <<'PY'
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import sys
sys.path.insert(0, "/tmp")
from _topo_loader import load_cases

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out",  type=str, required=True)
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[gae seed={args.seed}] device={DEVICE} "
      f"visible={torch.cuda.device_count()}", flush=True)

cases = load_cases()
print(f"[gae seed={args.seed}] loaded {len(cases)} cases", flush=True)
case_ids = np.array([c[0] for c in cases])
y        = np.array([c[2] for c in cases], dtype=int)

# Build per-structure Data objects (sanitize NaN features + empty edges)
per_struct = {"artery": [], "vein": [], "airway": []}
for cid, d, _ in cases:
    for k in per_struct:
        x = torch.nan_to_num(d[k].x.float(), 0.0, 0.0, 0.0)
        ei = d[k].edge_index
        if ei is None or ei.numel() == 0:
            N = x.size(0)
            ei = torch.stack([torch.arange(N), torch.arange(N)]).long()
        per_struct[k].append(Data(x=x, edge_index=ei))

class Enc(nn.Module):
    def __init__(self, in_dim=12, hidden=64, out=32):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden)
        self.c2 = GCNConv(hidden, out)
    def forward(self, x, ei):
        return self.c2(F.relu(self.c1(x, ei)), ei)

encoders = {k: Enc().to(DEVICE) for k in per_struct}
params = []
for e in encoders.values():
    params += list(e.parameters())
opt = torch.optim.Adam(params, lr=5e-3, weight_decay=1e-5)

BATCH  = 32
EPOCHS = 60
loaders = {k: DataLoader(per_struct[k], batch_size=BATCH, shuffle=True,
                         num_workers=2, pin_memory=True)
           for k in per_struct}

def recon_loss(z, pos_ei, neg_ei):
    def _score(ei):
        return (z[ei[0]] * z[ei[1]]).sum(dim=-1)
    pos = torch.sigmoid(_score(pos_ei)).clamp(min=1e-6)
    neg = 1 - torch.sigmoid(_score(neg_ei)).clamp(max=1 - 1e-6)
    return -(torch.log(pos).mean() + torch.log(neg).mean())

print(f"[gae seed={args.seed}] training …", flush=True)
t0 = time.time()
for ep in range(EPOCHS):
    tot = 0.0; nb = 0
    # iterate three structures independently — they are disjoint params
    for k, loader in loaders.items():
        enc = encoders[k]
        enc.train()
        for batch in loader:
            batch = batch.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            z = enc(batch.x, batch.edge_index)
            neg_ei = negative_sampling(batch.edge_index, num_nodes=batch.num_nodes,
                                       num_neg_samples=batch.edge_index.size(1))
            if neg_ei.numel() == 0:
                continue
            loss = recon_loss(z, batch.edge_index, neg_ei)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
    if ep % 10 == 0 or ep == EPOCHS - 1:
        print(f"  [gae seed={args.seed}] ep {ep:02d}  mean loss {tot/max(nb,1):.4f}", flush=True)
print(f"[gae seed={args.seed}] trained in {time.time()-t0:.1f}s", flush=True)

# per-case embedding: mean-pool encoder output, concat 3 structures
for e in encoders.values():
    e.eval()
embs = []
with torch.no_grad():
    for ci in range(len(cases)):
        parts = []
        for k in per_struct:
            data = per_struct[k][ci].to(DEVICE)
            z = encoders[k](data.x, data.edge_index)
            parts.append(z.mean(dim=0))
        embs.append(torch.cat(parts).cpu().numpy())
X = np.stack(embs).astype(np.float32)
print(f"[gae seed={args.seed}] embedding shape {X.shape}  ->  {args.out}", flush=True)
np.savez(args.out, X=X, case_ids=case_ids, y=y, seed=args.seed)
PY

# ─────────────────────── Job 3: merge + cluster (after parallel phase) ───────────────────────
cat > /tmp/_topo_cluster.py <<'PY'
from __future__ import annotations
import json, csv
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

OUT  = Path("/tmp/topo_out")
SEED = 42

A = np.load(OUT / "expA_wl_features.npz")
B = np.load(OUT / "expB_stats_features.npz")
G42 = np.load(OUT / "expC_gae_seed42.npz")
G43 = np.load(OUT / "expC_gae_seed43.npz")
# Ensemble: average the two GAE seed embeddings (ordering is identical — both
# enumerated cases in the same load order).
assert list(G42["case_ids"]) == list(G43["case_ids"])
X_gae = 0.5 * (G42["X"].astype(np.float32) + G43["X"].astype(np.float32))
np.savez(OUT / "expC_gae_embeddings.npz",
         X=X_gae, case_ids=G42["case_ids"], y=G42["y"],
         seeds=np.array([42, 43]))

y         = A["y"]
case_ids  = A["case_ids"]

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
                print(f"[{tag}] {method} k={k} fail: {e}", flush=True); continue
            sizes = np.bincount(lab).tolist()
            ari = float(adjusted_rand_score(y, lab))
            nmi = float(normalized_mutual_info_score(y, lab))
            try:
                sil = float(silhouette_score(X, lab)) if len(set(lab)) > 1 else float("nan")
            except Exception:
                sil = float("nan")
            rows.append({"experiment": tag, "method": method, "k": k,
                         "ARI_vs_PH": ari, "NMI_vs_PH": nmi, "silhouette": sil,
                         "sizes": sizes})
    return rows

all_rows = []
all_rows += sweep(A["X"],     "A_WL")
all_rows += sweep(B["X"],     "B_stats")
all_rows += sweep(X_gae,      "C_GAE")

with open(OUT / "topo_summary.json", "w") as f:
    json.dump({"n_cases": int(len(y)),
               "label_counts": [int((y == 0).sum()), int((y == 1).sum())],
               "rows": all_rows}, f, indent=2)
with open(OUT / "topo_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["experiment","method","k","ARI_vs_PH",
                                       "NMI_vs_PH","silhouette","sizes"])
    w.writeheader()
    for r in all_rows:
        r2 = dict(r); r2["sizes"] = "|".join(map(str, r["sizes"]))
        w.writerow(r2)

print()
print(f"{'exp':10s} {'method':10s} {'k':>2s} {'ARI':>7s} {'NMI':>7s} {'sil':>7s}  sizes", flush=True)
for r in all_rows:
    print(f"{r['experiment']:10s} {r['method']:10s} {r['k']:>2d} "
          f"{r['ARI_vs_PH']:>7.3f} {r['NMI_vs_PH']:>7.3f} "
          f"{r['silhouette']:>7.3f}  {r['sizes']}", flush=True)

# Best experiment -> dump per-cluster topology profile using raw graph stats
best = max(all_rows, key=lambda r: r["ARI_vs_PH"])
print(f"\n[topo] best across A/B/C: {best}", flush=True)
X_best = {"A_WL": A["X"], "B_stats": B["X"], "C_GAE": X_gae}[best["experiment"]]
if best["method"] == "kmeans":
    lab = KMeans(n_clusters=best["k"], n_init=20, random_state=SEED).fit_predict(X_best)
else:
    lab = SpectralClustering(n_clusters=best["k"], n_init=20, random_state=SEED,
                              affinity="nearest_neighbors", n_neighbors=10,
                              assign_labels="kmeans").fit_predict(X_best)

feat_names = []
for s in ("artery", "vein", "airway"):
    for n in ["n_nodes","n_edges","density","mean_deg","max_deg","std_deg",
             "n_leaves","n_bifurc","mean_diam","std_diam","max_diam","p90_diam",
             "mean_len","sum_len","mean_tort","std_tort","max_strahler",
             "mean_strahler","n_strahler_ge3"]:
        feat_names.append(f"{s}_{n}")

X_raw = B["X_raw"]
prof = []
for ci in range(best["k"]):
    m = lab == ci
    row = {"cluster": int(ci), "n": int(m.sum()),
           "ph_rate": float(y[m].mean()) if m.sum() else float("nan")}
    for j, fn in enumerate(feat_names):
        row[fn] = float(np.nanmean(X_raw[m, j])) if m.sum() else float("nan")
    prof.append(row)

with open(OUT / "topo_best_cluster_profile.json", "w") as f:
    json.dump({"best": best, "profile": prof, "feat_names": feat_names,
               "assignments": lab.tolist(),
               "case_ids": [str(x) for x in case_ids.tolist()],
               "y": [int(v) for v in y.tolist()]}, f, indent=2)

# Top-40 features by |Δ| between clusters
with open(OUT / "topo_best_cluster_diffs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["feature"] + [f"cluster_{i}" for i in range(best["k"])] + ["delta"])
    diffs = []
    for j, fn in enumerate(feat_names):
        vals = [prof[c][fn] for c in range(best["k"])]
        diffs.append((fn, vals, max(vals) - min(vals)))
    diffs.sort(key=lambda t: -abs(t[2]))
    for fn, vals, d in diffs[:40]:
        w.writerow([fn] + [f"{v:.3f}" for v in vals] + [f"{d:+.3f}"])
print("[topo] wrote best-cluster topology profile + top-40 diffs", flush=True)
PY

# ─────────────────────── Phase 1: launch 3 parallel jobs ───────────────────────
echo "=== phase 1: parallel CPU + 2-GPU training ==="
python /tmp/_topo_wlstat.py  > /tmp/topo_out/_log_wlstat.log 2>&1 &
PID_WLSTAT=$!
CUDA_VISIBLE_DEVICES=0 python /tmp/_topo_gae.py --seed 42 --out /tmp/topo_out/expC_gae_seed42.npz \
    > /tmp/topo_out/_log_gae_seed42.log 2>&1 &
PID_G42=$!
CUDA_VISIBLE_DEVICES=1 python /tmp/_topo_gae.py --seed 43 --out /tmp/topo_out/expC_gae_seed43.npz \
    > /tmp/topo_out/_log_gae_seed43.log 2>&1 &
PID_G43=$!
echo "  wlstat pid=$PID_WLSTAT   gae0 pid=$PID_G42   gae1 pid=$PID_G43"

RC_WLSTAT=0; RC_G42=0; RC_G43=0
wait $PID_WLSTAT || RC_WLSTAT=$?
wait $PID_G42    || RC_G42=$?
wait $PID_G43    || RC_G43=$?
echo "  rc wlstat=$RC_WLSTAT  gae0=$RC_G42  gae1=$RC_G43"

echo
echo "--- wlstat log ---"
tail -40 /tmp/topo_out/_log_wlstat.log
echo
echo "--- gae seed42 log ---"
tail -40 /tmp/topo_out/_log_gae_seed42.log
echo
echo "--- gae seed43 log ---"
tail -40 /tmp/topo_out/_log_gae_seed43.log

if [ $RC_WLSTAT -ne 0 ] || [ $RC_G42 -ne 0 ] || [ $RC_G43 -ne 0 ]; then
    echo "PHASE 1 FAILED"; exit 1
fi

# ─────────────────────── Phase 2: merge + cluster ───────────────────────
echo
echo "=== phase 2: cluster and summarize ==="
python /tmp/_topo_cluster.py 2>&1 | tee /tmp/topo_out/_log_cluster.log
echo
ls -la /tmp/topo_out/
'''

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=20, allow_agent=False, look_for_keys=False)
_, o, e = c.exec_command(SCRIPT, timeout=3600)
print(o.read().decode(errors="replace"))
err = e.read().decode(errors="replace")
if err.strip():
    print("--- STDERR (first 3000) ---")
    print(err[:3000])

# download artifacts
sftp = c.open_sftp()
from pathlib import Path as P
local_out = P(r"E:\桌面文件\图卷积-肺小血管演化规律探索\outputs\p_zeta_cluster_269\topology_evolution")
local_out.mkdir(parents=True, exist_ok=True)
want = [
    "topo_summary.json", "topo_summary.csv",
    "topo_best_cluster_profile.json", "topo_best_cluster_diffs.csv",
    "expA_wl_features.npz", "expB_stats_features.npz", "expC_gae_embeddings.npz",
    "expC_gae_seed42.npz", "expC_gae_seed43.npz",
    "_log_wlstat.log", "_log_gae_seed42.log", "_log_gae_seed43.log", "_log_cluster.log",
]
for fn in want:
    try:
        sftp.get(f"/tmp/topo_out/{fn}", str(local_out / fn))
        print(f"downloaded {fn}")
    except IOError as ex:
        print(f"skip {fn}: {ex}")
sftp.close()
c.close()
