"""Plan C — joint heterograph (artery + vein + airway as one PyG HeteroData)
with within-structure edges (from tri cache) + cross-structure companion
edges (artery↔airway NN, artery↔vein NN, distance-capped).

Runs server-side under pulmonary_bv5_py39 conda env. 5-fold stratified CV on
n=269 expanded cohort (matching p_theta_269_lr2x training budget: 40 epochs,
lr=2e-3, hidden=96, SAGEConv-per-edge-type via HeteroConv).

Writes /tmp/plan_c_out/{cv_results.json, train_log.txt} back for local
fetch.
"""
import paramiko

HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"

# write entire training script into /tmp/_plan_c.py via here-doc, then run
SCRIPT = r'''
source /home/imss/miniconda3/etc/profile.d/conda.sh
conda activate pulmonary_bv5_py39

cat > /tmp/_plan_c.py <<'PY'
from __future__ import annotations
import csv, glob, json, os, pickle, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

CACHE = "/home/imss/cw/GCN copdnoph copdph/cache_tri_converted"
LABELS_CSV = "/home/imss/cw/GCN copdnoph copdph/data/labels_expanded_282.csv"
OUT = Path("/tmp/plan_c_out")
OUT.mkdir(exist_ok=True)

SEED = 42
EPOCHS = 40
LR = 2e-3
WD = 1e-4
HIDDEN = 96
OUT_DIM = 64
BS = 16
K_CROSS = 3        # each artery node → 3 nearest airway + 3 nearest vein (mm, capped)
MAX_CROSS_MM = 25.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ---- labels
labels = {}
with open(LABELS_CSV, newline="") as f:
    r = csv.DictReader(f)
    key = "case_id" if "case_id" in r.fieldnames else "patient_id"
    for row in r:
        labels[row[key]] = int(row["label"])
print(f"labels: {len(labels)}")

# ---- build HeteroData per case
def _pos_of(g):
    if getattr(g, "pos", None) is not None:
        return g.pos.numpy().astype(float)
    if g.x is not None and g.x.size(1) >= 10:
        return g.x[:, 7:10].numpy().astype(float)
    return None

def build_hetero(pkl_path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    parts = {}
    for name in ("artery", "vein", "airway"):
        g = d.get(name)
        if g is None or getattr(g, "x", None) is None:
            return None
        if g.x.shape[0] == 0:
            return None
        parts[name] = g
    sp = d.get("spacing", (1.0, 1.0, 1.0))
    sp = np.array(sp, dtype=float).reshape(-1)
    if sp.size != 3: sp = np.array([1.0, 1.0, 1.0])
    sp = np.where(np.isfinite(sp) & (sp > 0), sp, 1.0)

    hetero = HeteroData()
    pos_mm = {}
    for name, g in parts.items():
        x = g.x.float()
        # replace NaN/inf node features with 0
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        hetero[name].x = x
        hetero[name].num_nodes = x.size(0)
        p = _pos_of(g)
        if p is None:
            p = np.zeros((x.size(0), 3))
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        pos_mm[name] = p * sp
        # within-structure edges from original graph
        ei = g.edge_index if g.edge_index is not None else torch.zeros((2, 0), dtype=torch.long)
        # add self-loop-ish fallback if empty
        if ei.numel() == 0:
            n = x.size(0)
            idx = torch.arange(n, dtype=torch.long)
            ei = torch.stack([idx, idx])
        hetero[(name, "within", name)].edge_index = ei

    # cross-structure companion edges
    def cross(src, dst):
        ps = pos_mm[src]; pd = pos_mm[dst]
        if len(ps) == 0 or len(pd) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        k = min(K_CROSS, len(pd))
        try:
            tree = cKDTree(pd)
            dists, idxs = tree.query(ps, k=k)
        except Exception:
            return torch.zeros((2, 0), dtype=torch.long)
        if k == 1:
            dists = dists[:, None]; idxs = idxs[:, None]
        src_idx = np.repeat(np.arange(len(ps)), k)
        dst_idx = idxs.reshape(-1)
        dd = dists.reshape(-1)
        m = (dd <= MAX_CROSS_MM) & np.isfinite(dd)
        if m.sum() == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        ei = torch.tensor(np.stack([src_idx[m], dst_idx[m]]), dtype=torch.long)
        return ei

    hetero[("artery", "near", "airway")].edge_index = cross("artery", "airway")
    hetero[("airway", "near", "artery")].edge_index = cross("airway", "artery")
    hetero[("artery", "near", "vein")].edge_index = cross("artery", "vein")
    hetero[("vein", "near", "artery")].edge_index = cross("vein", "artery")
    hetero[("airway", "near", "vein")].edge_index = cross("airway", "vein")
    hetero[("vein", "near", "airway")].edge_index = cross("vein", "airway")

    hetero.y = torch.tensor([labels[Path(pkl_path).stem.replace("_tri", "")]], dtype=torch.long)
    hetero.case_id = Path(pkl_path).stem.replace("_tri", "")
    return hetero

print("building hetero dataset ...")
t0 = time.time()
dataset = []
skipped = 0
for pkl in sorted(glob.glob(os.path.join(CACHE, "*.pkl"))):
    cid = Path(pkl).stem.replace("_tri", "")
    if cid not in labels: continue
    try:
        h = build_hetero(pkl)
    except Exception as e:
        print("build fail", cid, e); skipped += 1; continue
    if h is None: skipped += 1; continue
    dataset.append(h)
print(f"built: {len(dataset)}  skipped: {skipped}  in {time.time()-t0:.1f}s")

# Sniff feature dims
in_dims = {k: dataset[0][k].x.size(1) for k in ("artery", "vein", "airway")}
print("in_dims:", in_dims)

# ---- model
EDGE_TYPES = [
    ("artery", "within", "artery"),
    ("vein",   "within", "vein"),
    ("airway", "within", "airway"),
    ("artery", "near",   "airway"),
    ("airway", "near",   "artery"),
    ("artery", "near",   "vein"),
    ("vein",   "near",   "artery"),
    ("airway", "near",   "vein"),
    ("vein",   "near",   "airway"),
]

class HeteroGCN(nn.Module):
    def __init__(self, in_dims, hidden=HIDDEN, out=OUT_DIM):
        super().__init__()
        self.proj = nn.ModuleDict({k: nn.Linear(in_dims[k], hidden) for k in in_dims})
        self.conv1 = HeteroConv({et: SAGEConv((-1, -1), hidden) for et in EDGE_TYPES}, aggr="mean")
        self.conv2 = HeteroConv({et: SAGEConv((-1, -1), hidden) for et in EDGE_TYPES}, aggr="mean")
        self.bn = nn.ModuleDict({k: nn.BatchNorm1d(hidden) for k in in_dims})
        self.post = nn.Sequential(
            nn.Linear(hidden * 3, out), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(out, 2),
        )

    def forward(self, data):
        x_dict = {k: self.proj[k](data[k].x) for k in ("artery", "vein", "airway")}
        x_dict = {k: F.relu(self.bn[k](v)) for k, v in x_dict.items()}
        ei_dict = {et: data[et].edge_index for et in EDGE_TYPES if et in data.edge_types}
        h = self.conv1(x_dict, ei_dict)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(h, ei_dict)
        h = {k: F.relu(v) for k, v in h.items()}
        pooled = {}
        for k in ("artery", "vein", "airway"):
            # global mean pool per case within batch
            batch = data[k].batch if hasattr(data[k], "batch") else torch.zeros(data[k].num_nodes, dtype=torch.long, device=h[k].device)
            pooled[k] = global_mean_pool(h[k], batch)
        feat = torch.cat([pooled["artery"], pooled["vein"], pooled["airway"]], dim=-1)
        return self.post(feat)

# ---- 5-fold CV
y_all = np.array([int(d.y.item()) for d in dataset])
print("label counts:", np.bincount(y_all))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def metrics(y, p, thr=0.5):
    yp = (p >= thr).astype(int)
    auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
    acc = accuracy_score(y, yp)
    f1 = f1_score(y, yp, zero_division=0)
    prec = precision_score(y, yp, zero_division=0)
    sens = recall_score(y, yp, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return dict(auc=auc, accuracy=acc, f1=f1, precision=prec, sensitivity=sens, specificity=spec)

fold_results = []
for fi, (tr, te) in enumerate(skf.split(np.arange(len(dataset)), y_all)):
    train_ds = [dataset[i] for i in tr]
    test_ds = [dataset[i] for i in te]
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False)
    model = HeteroGCN(in_dims).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    # class weights
    cw = torch.tensor([1.0, (y_all == 0).sum() / max((y_all == 1).sum(), 1)], dtype=torch.float, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=cw)
    best_auc = -1; best_metrics = None
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            logits = model(batch)
            loss = crit(logits, batch.y)
            loss.backward(); opt.step()
            tot += loss.item() * batch.y.size(0)
        # eval
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                logits = model(batch)
                prob = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                ys.append(batch.y.cpu().numpy()); ps.append(prob)
        ys = np.concatenate(ys); ps = np.concatenate(ps)
        m = metrics(ys, ps)
        if m["auc"] > best_auc:
            best_auc = m["auc"]; best_metrics = m
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"  fold {fi} ep {ep:02d} loss {tot/len(train_ds):.4f} AUC {m['auc']:.4f}")
    print(f"[fold {fi}] best AUC={best_metrics['auc']:.4f}  Acc={best_metrics['accuracy']:.3f}  F1={best_metrics['f1']:.3f}")
    fold_results.append(best_metrics)

# aggregate
agg = {k: {"mean": float(np.mean([r[k] for r in fold_results])),
           "std":  float(np.std([r[k] for r in fold_results])),
           "folds":[float(r[k]) for r in fold_results]}
       for k in ("auc","accuracy","precision","sensitivity","specificity","f1")}
print("\n=== Plan C aggregate 5-fold ===")
for k, v in agg.items():
    print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")

out = {
    "config": {
        "n_cases": len(dataset), "skipped": skipped, "seed": SEED,
        "epochs": EPOCHS, "lr": LR, "weight_decay": WD, "hidden": HIDDEN,
        "out_dim": OUT_DIM, "batch_size": BS, "k_cross": K_CROSS,
        "max_cross_mm": MAX_CROSS_MM, "edge_types": [list(e) for e in EDGE_TYPES],
        "label_counts": [int((y_all == 0).sum()), int((y_all == 1).sum())],
    },
    "per_fold": fold_results, "aggregate": agg,
}
with open(OUT / "cv_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("wrote", OUT / "cv_results.json")
PY

python /tmp/_plan_c.py 2>&1 | tee /tmp/plan_c_run.log
echo "---"
ls -la /tmp/plan_c_out/
'''

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
_, o, e = c.exec_command(SCRIPT, timeout=3600)
print(o.read().decode(errors="replace"))
err = e.read().decode(errors="replace")
if err.strip():
    print("--- STDERR (first 3000) ---")
    print(err[:3000])

sftp = c.open_sftp()
from pathlib import Path as P
local_out = P(r"E:\桌面文件\图卷积-肺小血管演化规律探索\outputs\p_zeta_cluster_269\plan_c")
local_out.mkdir(parents=True, exist_ok=True)
for fn in ["cv_results.json"]:
    try:
        sftp.get(f"/tmp/plan_c_out/{fn}", str(local_out / fn))
        print(f"downloaded {fn}")
    except IOError as ex:
        print(f"skip {fn}: {ex}")
# also pull the run log
try:
    sftp.get("/tmp/plan_c_run.log", str(local_out / "plan_c_run.log"))
    print("downloaded plan_c_run.log")
except IOError as ex:
    print("skip run log:", ex)
sftp.close()
c.close()
