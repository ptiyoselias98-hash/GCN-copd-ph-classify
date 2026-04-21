"""Fetch n=269 shared_embeddings.npz from the best tri_structure job + labels
so Plan A (unsupervised clustering at n=269) can run locally on CPU.

Best job: p_theta_269_lr2x (AUC 0.928, mean pool, lr=2e-3, mpap_aux).
"""
import paramiko
from pathlib import Path

HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"
REMOTE_ROOT = "/home/imss/cw/GCN copdnoph copdph"
LOCAL_OUT = Path(r"E:\桌面文件\图卷积-肺小血管演化规律探索\outputs\p_zeta_cluster_269")
LOCAL_OUT.mkdir(parents=True, exist_ok=True)

FILES = [
    (f"{REMOTE_ROOT}/outputs/p_theta_269_lr2x/shared_embeddings.npz",
     LOCAL_OUT / "p_theta_269_lr2x_embeddings.npz"),
    (f"{REMOTE_ROOT}/outputs/p_theta_269_lr2x/cv_results.json",
     LOCAL_OUT / "p_theta_269_lr2x_cv_results.json"),
    (f"{REMOTE_ROOT}/outputs/p_theta_269_lr2x/cluster_analysis.json",
     LOCAL_OUT / "p_theta_269_lr2x_cluster_analysis.json"),
    (f"{REMOTE_ROOT}/data/labels_expanded_282.csv",
     LOCAL_OUT / "labels_expanded_282.csv"),
]

c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False, look_for_keys=False)
sftp = c.open_sftp()
for remote, local in FILES:
    try:
        sftp.get(remote, str(local))
        print(f"OK  {local.name}  ({local.stat().st_size} bytes)")
    except IOError as e:
        print(f"MISS {remote}: {e}")
sftp.close()
c.close()
