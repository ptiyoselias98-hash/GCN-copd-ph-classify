"""Remote: compile-check + import smoke-test in the conda env."""
from __future__ import annotations
import sys
import os
import paramiko

HOST, PORT, USER = "10.60.147.117", 22, "imss"
PASS = os.environ.get("IMSS_SSH_PASSWORD")
if not PASS:
    raise RuntimeError("Set IMSS_SSH_PASSWORD before running this script.")
REMOTE_REPO = "/home/imss/cw/GCN copdnoph copdph"
ENV_ACT = "source /home/imss/miniconda3/etc/profile.d/conda.sh && conda activate pulmonary_bv5_py39"


def main() -> int:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, PORT, USER, PASS, timeout=15, allow_agent=False,
                look_for_keys=False)

    def run(cmd: str, timeout: int = 120) -> tuple[int, str, str]:
        stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        rc = stdout.channel.recv_exit_status()
        return rc, out, err

    cmd = (
        f"bash -lc \"{ENV_ACT} && cd '{REMOTE_REPO}' && "
        "python -c 'import sys; print(sys.version); "
        "import py_compile; "
        "[py_compile.compile(f, doraise=True) for f in "
        "[\\\"enhance_features.py\\\",\\\"hybrid_gcn.py\\\",\\\"run_sprint2.py\\\"]]; "
        "print(\\\"COMPILE_OK\\\")'\""
    )
    rc, out, err = run(cmd)
    print("[compile]", rc)
    print(out)
    if err.strip():
        print("STDERR:", err)

    # quick import smoke (no GPU needed, just import + augment one synthetic)
    smoke = (
        "from enhance_features import augment_graph, EXPECTED_OUT_DIM, GLOBAL_FEATURE_DIM;"
        "from hybrid_gcn import HybridGCN;"
        "import torch;"
        "from torch_geometric.data import Data, Batch;"
        "N=15; E=30;"
        "d=Data(x=torch.randn(N,12), edge_index=torch.randint(0,N,(2,E)),"
        " pos=torch.randn(N,3), y=torch.tensor([1]));"
        "aug=augment_graph(d, commercial_total_vol_ml=10.0,"
        " commercial_fractal_dim=1.5, commercial_artery_density=-700.0,"
        " commercial_vein_density=-650.0, pipeline_total_vol_ml=8.0,"
        " commercial_vein_bv5=0.3, commercial_vein_branch_count=50,"
        " commercial_bv5_ratio=0.4, commercial_artery_vein_vol_ratio=1.2,"
        " commercial_total_bv5=0.9, commercial_lung_density_std=40.0,"
        " commercial_vein_bv10=1.1, commercial_total_branch_count=120,"
        " commercial_vessel_tortuosity=1.05);"
        "aug.radiomics=torch.randn(1,45); aug2=aug.clone(); aug2.y=torch.tensor([0]);"
        "b=Batch.from_data_list([aug,aug2]);"
        "print('x',b.x.shape,'glob',b.global_features.shape);"
        "m=HybridGCN(gcn_in=13, radiomics_dim=45, mode='hybrid', global_dim=12); m.eval();"
        "logits,_,_=m(b.x,b.edge_index,b.batch,radiomics=b.radiomics,"
        " global_features=b.global_features);"
        "print('hybrid logits', logits.shape);"
        "m2=HybridGCN(gcn_in=13, radiomics_dim=45, mode='gcn_only', global_dim=12); m2.eval();"
        "logits2,_,_=m2(b.x,b.edge_index,b.batch,radiomics=b.radiomics,"
        " global_features=b.global_features);"
        "print('gcn_only logits', logits2.shape);"
        "print('SMOKE_OK')"
    )
    cmd2 = f"bash -lc \"{ENV_ACT} && cd '{REMOTE_REPO}' && python -c \\\"{smoke}\\\"\""
    rc, out, err = run(cmd2, timeout=180)
    print("[smoke]", rc)
    print(out)
    if err.strip():
        print("STDERR:", err[-2000:])

    cli.close()
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
