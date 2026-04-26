"""R26.B — Populate RUN_MANIFEST.json with actual SHA256 + producing commit SHA.

Reads outputs/r24/RUN_MANIFEST.json skeleton, fills SHA256 for each existing artifact.
Producing-commit SHA is read from `git log -1 --format=%H -- <path>` per file.
"""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
MANIFEST = ROOT / "outputs" / "r24" / "RUN_MANIFEST.json"


def file_sha256(p: Path) -> str | None:
    if not p.exists(): return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def producing_commit(repo_root: Path, rel_path: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", rel_path],
            cwd=repo_root, capture_output=True, text=True, timeout=15)
        sha = out.stdout.strip()
        return sha if sha else None
    except Exception:
        return None


def populate(node: dict, repo_root: Path):
    """Walk schema dicts, fill sha256 + commit_sha for each {sha256, commit_sha, path} leaf."""
    if isinstance(node, dict):
        if "path" in node and "sha256" in node and "commit_sha" in node:
            p = Path(node["path"])
            abs_p = (ROOT.parent / p) if not p.is_absolute() else p
            # try a couple of resolution strategies
            for cand in [abs_p, ROOT / p, ROOT / Path(*p.parts[1:]) if len(p.parts) > 1 else ROOT]:
                if cand.exists():
                    abs_p = cand; break
            if abs_p.exists():
                node["sha256"] = file_sha256(abs_p)
                rel = str(p).replace("\\", "/")
                node["commit_sha"] = producing_commit(repo_root, rel)
            else:
                node["sha256"] = "MISSING"
                node["commit_sha"] = "MISSING"
        else:
            for v in node.values():
                populate(v, repo_root)
    elif isinstance(node, list):
        for item in node:
            populate(item, repo_root)


def main():
    repo_root = ROOT.parent
    d = json.loads(MANIFEST.read_text(encoding="utf-8"))
    d["manifest_version"] = "v8-populated"
    d["populated_at"] = "2026-04-26"
    schema = d.get("schema", {})
    populate(schema, repo_root)

    # Add R25/R26 artifacts that weren't in v7 skeleton
    add_artifacts = {
        "r25b_ssl_extended_oof": {"sha256": None, "commit_sha": None,
                                   "path": "outputs/r24/r25b_ssl_extended_oof.csv"},
        "r25b_pca_baseline_oof": {"sha256": None, "commit_sha": None,
                                   "path": "outputs/r24/r25b_pca_extended_oof.csv"},
        "r25c_ensemble_oof": {"sha256": None, "commit_sha": None,
                               "path": "outputs/r24/r25c_ensemble_oof.csv"},
        "r26a_modality_ablation": {"sha256": None, "commit_sha": None,
                                    "path": "outputs/r24/r26a_modality_ablation.csv"},
        "extended_features_212": {"sha256": None, "commit_sha": None,
                                   "path": "outputs/r24/extended_features_212.csv"},
    }
    populate(add_artifacts, repo_root)
    schema.setdefault("r25_r26_artifacts", {}).update(add_artifacts)

    # Update gate verdicts from validation.json files
    val_files = {
        "r24a_mpap_rho_ge_0.35": ("outputs/r24/r24a_validation.json",
                                    "within_contrast.rho_mpap"),
        "r24g_oof_rho_ge_0.50": ("outputs/r24/r24g_validation.json", "ssl_d32.oof_rho_vs_mpap"),
        "r25c_ensemble_rho_ge_0.50": ("outputs/r24/r25c_validation.json", "ensemble_rho"),
    }
    for gate_name, (rel_p, key_path) in val_files.items():
        p = ROOT / rel_p
        if p.exists():
            try:
                v = json.loads(p.read_text(encoding="utf-8"))
                for k in key_path.split("."):
                    v = v[k]
                schema.setdefault("gates", {}).setdefault(gate_name, {})
                schema["gates"][gate_name]["statistic"] = float(v)
                if "ge_0.50" in gate_name:
                    schema["gates"][gate_name]["threshold"] = 0.50
                    schema["gates"][gate_name]["verdict"] = "PASS" if float(v) >= 0.50 else "FAIL"
                elif "ge_0.35" in gate_name:
                    schema["gates"][gate_name]["threshold"] = 0.35
                    schema["gates"][gate_name]["verdict"] = "PASS" if abs(float(v)) >= 0.35 else "FAIL"
            except Exception as e:
                print(f"could not populate {gate_name}: {e}")

    MANIFEST.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print("RUN_MANIFEST populated:")
    print(f"  total schema keys: {len(schema)}")
    # Count populated vs MISSING
    def count_pop(node, c):
        if isinstance(node, dict):
            if "sha256" in node and node.get("sha256") and node["sha256"] != "MISSING":
                c["populated"] += 1
            elif "sha256" in node:
                c["missing"] += 1
            else:
                for v in node.values(): count_pop(v, c)
        elif isinstance(node, list):
            for v in node: count_pop(v, c)
    counts = {"populated": 0, "missing": 0}
    count_pop(schema, counts)
    print(f"  artifacts: {counts['populated']} populated, {counts['missing']} missing")


if __name__ == "__main__":
    main()
