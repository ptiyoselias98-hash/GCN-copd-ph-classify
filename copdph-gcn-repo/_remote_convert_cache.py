"""Convert root cache_tri (new-schema) → tri_structure-style _tri.pkl (PyG Data)."""
import paramiko
HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"

SCRIPT = '''
import pickle, glob, os, torch
from torch_geometric.data import Data

SRC = "/home/imss/cw/GCN copdnoph copdph/cache_tri"
DST = "/home/imss/cw/GCN copdnoph copdph/cache_tri_converted"
os.makedirs(DST, exist_ok=True)

def _trivial_data(feat_dim):
    x = torch.zeros((1, feat_dim), dtype=torch.float32)
    edge_index = torch.zeros((2,0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

ok, fail = 0, 0
feat_dim = None
for src in sorted(glob.glob(os.path.join(SRC, "*.pkl"))):
    base = os.path.basename(src).replace(".pkl","")
    dst = os.path.join(DST, f"{base}_tri.pkl")
    if os.path.exists(dst):
        ok += 1; continue
    try:
        with open(src,"rb") as f: d = pickle.load(f)
        out = {"patient_id": base, "label": int(d["label"]), "spacing": d.get("spacing")}
        # extract graph from each structure
        for k in ("artery","vein","airway"):
            v = d.get(k)
            if v is None:
                # skip for now, fill later if dim known
                out[k] = None
            elif isinstance(v, dict) and "graph" in v:
                g = v["graph"]
                if feat_dim is None and hasattr(g,"x"): feat_dim = g.x.shape[1]
                out[k] = g
            elif hasattr(v, "x"):
                if feat_dim is None: feat_dim = v.x.shape[1]
                out[k] = v
            else:
                out[k] = None
        # now replace None structures with trivial placeholder (use feat_dim from any present)
        if feat_dim is None: feat_dim = 12
        for k in ("artery","vein","airway"):
            if out[k] is None:
                out[k] = _trivial_data(feat_dim)
        with open(dst,"wb") as f: pickle.dump(out, f)
        ok += 1
    except Exception as e:
        print(f"FAIL {base}: {e}")
        fail += 1

print(f"ok={ok} fail={fail} total_dst={len(glob.glob(os.path.join(DST,'*')))}")
'''
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
sftp = c.open_sftp()
with sftp.file("/tmp/_convert.py","w") as f: f.write(SCRIPT)
sftp.close()
_, o, e = c.exec_command("source /home/imss/miniconda3/etc/profile.d/conda.sh && conda activate pulmonary_bv5_py39 && python /tmp/_convert.py", timeout=300)
print(o.read().decode()); err = e.read().decode()
if err.strip(): print("ERR:", err)
c.close()
