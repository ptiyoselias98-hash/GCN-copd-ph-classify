"""W1 protocol-confound ablation.

Re-trains arm_b (tri-flat) and arm_c (+ lung scalars) on the 197-case
CONTRAST-ENHANCED ONLY subset (170 PH + 27 nonPH). If the AUC drops
substantially vs. the 243-case full cohort, the v1/v2 vessel signal is
dominated by acquisition-protocol cues rather than PH biology; if it
holds, disease signal is credible.
"""
from __future__ import annotations

import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import paramiko

HOST, PORT, USER, PASS = "10.60.147.117", 22, "imss", "imsslab"
REMOTE = "/home/imss/cw/GCN copdnoph copdph"
LOCAL = Path(r"E:\桌面文件\图卷积-肺小血管演化规律探索\copdph-gcn-repo\_local_data_contrast_only")
ENV = "source /home/imss/miniconda3/etc/profile.d/conda.sh && conda activate pulmonary_bv5_py39"


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=30, allow_agent=False, look_for_keys=False)
    sftp = c.open_sftp()

    # 1) Push labels + splits
    sftp.put(str(LOCAL / "labels_contrast_only.csv"), f"{REMOTE}/data/labels_contrast_only.csv")
    print(f"[push] labels_contrast_only.csv")

    for i in range(1, 6):
        try:
            sftp.stat(f"{REMOTE}/data/splits_contrast_only")
        except FileNotFoundError:
            sftp.mkdir(f"{REMOTE}/data/splits_contrast_only")
            break
    try:
        sftp.stat(f"{REMOTE}/data/splits_contrast_only")
    except FileNotFoundError:
        sftp.mkdir(f"{REMOTE}/data/splits_contrast_only")

    for i in range(1, 6):
        try:
            sftp.stat(f"{REMOTE}/data/splits_contrast_only/fold_{i}")
        except FileNotFoundError:
            sftp.mkdir(f"{REMOTE}/data/splits_contrast_only/fold_{i}")
        for t in ("train.txt", "val.txt"):
            sftp.put(str(LOCAL / f"fold_{i}" / t),
                     f"{REMOTE}/data/splits_contrast_only/fold_{i}/{t}")
    print(f"[push] splits_contrast_only/fold_{{1..5}}/train+val")
    sftp.close()

    def _run(cmd, t=60):
        _, o, e = c.exec_command(cmd, timeout=t)
        return o.read().decode("utf-8", "replace"), e.read().decode("utf-8", "replace")

    # 2) Confirm uploads
    o, _ = _run(f"wc -l '{REMOTE}/data/labels_contrast_only.csv' "
                f"'{REMOTE}/data/splits_contrast_only'/fold_*/val.txt 2>&1 | head -20")
    print("[check]\n" + o)

    # 3) Launch arm_b_contrast (GPU0)
    out_b = "outputs/sprint6_arm_b_contrast_only_v2"
    log_b = f"{out_b}/run.log"
    launch_b = (
        f"cd '{REMOTE}' && mkdir -p '{out_b}' && "
        f"CUDA_VISIBLE_DEVICES=0 nohup python -u run_sprint6_v2.py "
        f"  --arm arm_a "
        f"  --cache_dir ./cache_v2_tri_flat "
        f"  --radiomics ./data/copd_ph_radiomics.csv "
        f"  --labels ./data/labels_contrast_only.csv "
        f"  --splits ./data/splits_contrast_only "
        f"  --output_dir './{out_b}' "
        f"  --epochs 120 --batch_size 16 --lr 1e-3 "
        f"  --keep_full_node_dim --skip_enhanced --num_workers 4 "
        f"  --augment edge_drop,feature_mask "
        f"  --repeats 3 "
        f"  < /dev/null > '{log_b}' 2>&1 & disown; echo arm_b_contrast-launched"
    )
    try:
        o, e = _run(f"bash -lc \"{ENV} && {launch_b}\"", t=10)
        print("[launch arm_b_contrast]", (o or "").strip())
    except Exception as ex:
        print("[launch arm_b_contrast] (paramiko ok — nohup backgrounded)", ex)

    # 4) Launch arm_c_contrast (GPU1)
    out_c = "outputs/sprint6_arm_c_contrast_only_v2"
    log_c = f"{out_c}/run.log"
    launch_c = (
        f"cd '{REMOTE}' && mkdir -p '{out_c}' && "
        f"CUDA_VISIBLE_DEVICES=1 nohup python -u run_sprint6_v2.py "
        f"  --arm arm_c "
        f"  --cache_dir ./cache_v2_tri_flat "
        f"  --radiomics ./data/copd_ph_radiomics.csv "
        f"  --labels ./data/labels_contrast_only.csv "
        f"  --splits ./data/splits_contrast_only "
        f"  --output_dir './{out_c}' "
        f"  --lung_features_csv ./data/lung_features_only.csv "
        f"  --epochs 120 --batch_size 16 --lr 1e-3 "
        f"  --keep_full_node_dim --skip_enhanced --num_workers 4 "
        f"  --augment edge_drop,feature_mask "
        f"  --repeats 3 "
        f"  < /dev/null > '{log_c}' 2>&1 & disown; echo arm_c_contrast-launched"
    )
    try:
        o, e = _run(f"bash -lc \"{ENV} && {launch_c}\"", t=10)
        print("[launch arm_c_contrast]", (o or "").strip())
    except Exception as ex:
        print("[launch arm_c_contrast] (paramiko ok — nohup backgrounded)", ex)

    # 5) sleep then confirm pids + tails
    time.sleep(8)
    o, _ = _run("pgrep -fa 'contrast_only' | head -6")
    print("[pgrep-after]\n" + o)
    for label, log in (("arm_b_contrast", log_b), ("arm_c_contrast", log_c)):
        o, _ = _run(f"tail -n 12 '{REMOTE}/{log}' 2>&1")
        print(f"[tail {label}]\n{o}")

    c.close()


if __name__ == "__main__":
    main()
