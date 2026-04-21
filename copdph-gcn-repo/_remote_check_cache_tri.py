"""Check what format the root cache_tri contains (268 files)."""
import paramiko
HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"

REMOTE = "/tmp/_chk.py"
SCRIPT = """
import pickle, os, glob
files = sorted(glob.glob("/home/imss/cw/GCN copdnoph copdph/cache_tri/*.pkl"))
print("total:", len(files))
for p in files[:3]:
    with open(p,"rb") as f: d = pickle.load(f)
    print(os.path.basename(p), "keys:", list(d.keys())[:10] if isinstance(d,dict) else type(d).__name__)
# count by suffix
import re
bases = set(os.path.basename(f).replace("_tri.pkl",".pkl").replace(".pkl","") for f in files)
tri_suffix = sum(1 for f in files if f.endswith("_tri.pkl"))
print("with _tri suffix:", tri_suffix)
print("unique bases:", len(bases))
"""
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
sftp = c.open_sftp()
with sftp.file(REMOTE,"w") as f: f.write(SCRIPT)
sftp.close()
_, o, e = c.exec_command(f"source /home/imss/miniconda3/etc/profile.d/conda.sh && conda activate pulmonary_bv5_py39 && python {REMOTE}", timeout=60)
print(o.read().decode("utf-8","replace"))
err = e.read().decode("utf-8","replace")
if err.strip(): print("ERR:", err)
c.close()
