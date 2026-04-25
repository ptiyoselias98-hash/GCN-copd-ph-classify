"""Autonomous README updater — appends a Round section + refreshes the score table.

Usage from the cron loop:
    python scripts/figures/update_readme_round.py <N> <score> \
      --artifacts outputs/r<N>/result.md outputs/r<N>/figure.png \
      --summary "short 1-sentence summary"

The script:
  - regenerates figures via R7_make_figures.py (so fig1 score table is current),
  - ensures each listed artefact (md) is linked in the new section,
  - embeds each listed png via markdown `![](path)`,
  - inserts a `## ARIS Round {N}` section at the end of README.md if not present,
  - updates the Round history table with the new score.

Idempotent: running twice for the same round replaces the section in place.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent  # .../图卷积-肺小血管演化规律探索
README = ROOT / "README.md"
REPO = ROOT / "copdph-gcn-repo"
STATE = REPO / "review-stage" / "REVIEW_STATE.json"


def regenerate_figures() -> None:
    gen = REPO / "scripts" / "figures" / "R7_make_figures.py"
    if gen.exists():
        try:
            subprocess.run(
                ["python", str(gen)],
                cwd=str(REPO), check=True, capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"WARN: fig regenerator failed: {exc}")


def build_section(n: int, score: float, artefacts: list[str], summary: str) -> str:
    lines = [
        f"## ARIS Round {n} — score {score:.1f}/10",
        "",
        summary or f"Round {n} delivered per AUTONOMOUS_LOOP_PLAN.md.",
        "",
    ]
    # Embed PNG artefacts; link others
    for a in artefacts:
        p = Path(a.replace("\\", "/"))
        rel = p.as_posix()
        if not rel.startswith("copdph-gcn-repo/"):
            rel = f"copdph-gcn-repo/{rel}"
        if rel.lower().endswith(".png"):
            lines.append(f"![{p.stem}]({rel})")
            lines.append("")
        else:
            lines.append(f"- [{p.name}]({rel})")
    lines.append("")
    return "\n".join(lines)


def upsert_section(readme_text: str, section: str, n: int) -> str:
    header_re = re.compile(
        rf"^## ARIS Round {n} — .*?(?=^## |\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    if header_re.search(readme_text):
        return header_re.sub(section.rstrip() + "\n\n", readme_text)
    # Append at end
    sep = "\n\n" if not readme_text.endswith("\n\n") else ""
    return readme_text + sep + section


def upsert_score_table(readme_text: str) -> str:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    hist = sorted(
        (h for h in state["history"] if "score" in h),
        key=lambda h: h["round"],
    )
    table = ["## ARIS round history (auto-generated)", "",
             "| Round | Score | Verdict |", "|---|---|---|"]
    for h in hist:
        table.append(f"| R{h['round']} | {h['score']}/10 | {h.get('verdict', '?')} |")
    block = "\n".join(table) + "\n"

    pat = re.compile(
        r"^## ARIS round history \(auto-generated\).*?(?=^## |\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    if pat.search(readme_text):
        return pat.sub(block + "\n", readme_text)
    # Insert near the top (after the first `## ` that isn't the title)
    first_hdr = re.search(r"^## ", readme_text, flags=re.MULTILINE)
    if first_hdr:
        pos = first_hdr.start()
        return readme_text[:pos] + block + "\n" + readme_text[pos:]
    return readme_text + "\n\n" + block


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("round_n", type=int)
    ap.add_argument("score", type=float)
    ap.add_argument("--artifacts", nargs="*", default=[])
    ap.add_argument("--summary", default="")
    args = ap.parse_args()

    regenerate_figures()

    text = README.read_text(encoding="utf-8")
    section = build_section(args.round_n, args.score, args.artifacts, args.summary)
    text = upsert_section(text, section, args.round_n)
    text = upsert_score_table(text)
    README.write_text(text, encoding="utf-8")
    print(f"Updated {README} with Round {args.round_n} section (score {args.score}).")


if __name__ == "__main__":
    main()
