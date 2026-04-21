"""Inspect exact schema of root cache_tri file vs tri_structure/cache_tri."""
import paramiko
HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"

SCRIPT = """
import pickle
paths = [
    "/home/imss/cw/GCN copdnoph copdph/cache_tri_linked/nonph_baozhiding_k00748487_tuesday_june_18_2019_000_tri.pkl",
    "/home/imss/cw/GCN copdnoph copdph/tri_structure/cache_tri/nonph_caochenglin_g02017953_thursday_july_9_2020_000_tri.pkl",
]
for p in paths:
    print("====", p)
    with open(p,"rb") as f: d = pickle.load(f)
    print("  top keys:", list(d.keys()))
    for k in ("artery","vein","airway"):
        if k in d:
            v = d[k]
            print(f"  {k}: type={type(v).__name__}", "keys:" if isinstance(v,dict) else "", list(v.keys())[:10] if isinstance(v,dict) else f"x.shape={v.x.shape if hasattr(v,'x') else '?'}")
"""
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
sftp = c.open_sftp()
with sftp.file("/tmp/_inspect.py","w") as f: f.write(SCRIPT)
sftp.close()
_, o, e = c.exec_command("source /home/imss/miniconda3/etc/profile.d/conda.sh && conda activate pulmonary_bv5_py39 && python /tmp/_inspect.py", timeout=30)
print(o.read().decode()); err = e.read().decode()
if err.strip(): print("ERR:", err)
c.close()
