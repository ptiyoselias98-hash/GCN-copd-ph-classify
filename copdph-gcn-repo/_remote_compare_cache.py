"""Compare airway node counts between tri_structure/cache_tri and sprint7/cache_tri via SFTP+exec."""
import paramiko
HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"

REMOTE_SCRIPT = "/tmp/_compare_cache.py"
SCRIPT = """
import pickle, os, glob, statistics
for cache_dir in ["tri_structure/cache_tri","sprint7/cache_tri"]:
    files = sorted(glob.glob(os.path.join(cache_dir,"*_tri.pkl")))
    print("===", cache_dir, len(files), "files ===")
    airway_nodes, artery_nodes, vein_nodes = [], [], []
    for p in files:
        with open(p,"rb") as f: d = pickle.load(f)
        for k,v in d.items():
            if hasattr(v,"x"):
                n = v.x.shape[0]
                kl = k.lower()
                if "air" in kl or "bron" in kl: airway_nodes.append(n)
                elif "art" in kl: artery_nodes.append(n)
                elif "vein" in kl: vein_nodes.append(n)
    def stats(L,name):
        if not L: print("  ", name, ": EMPTY"); return
        print("  ", name, ": n=", len(L), "median=", statistics.median(L),
              "min=", min(L), "max=", max(L), "<=8:", sum(1 for x in L if x<=8))
    stats(airway_nodes,"airway")
    stats(artery_nodes,"artery")
    stats(vein_nodes,"vein")
    if files:
        with open(files[0],"rb") as f: d = pickle.load(f)
        print("  keys:", list(d.keys())[:10])
"""

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
sftp = c.open_sftp()
with sftp.file(REMOTE_SCRIPT, "w") as f:
    f.write(SCRIPT)
sftp.close()

cmd = f'cd "/home/imss/cw/GCN copdnoph copdph" && source /home/imss/miniconda3/etc/profile.d/conda.sh && conda activate pulmonary_bv5_py39 && python {REMOTE_SCRIPT}'
_, o, e = c.exec_command(cmd, timeout=300)
print(o.read().decode("utf-8","replace"))
err = e.read().decode("utf-8","replace")
if err.strip(): print("ERR:", err)
c.close()
