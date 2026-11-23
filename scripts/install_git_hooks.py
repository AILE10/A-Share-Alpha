"""Install local Git pre-commit (privacy guard) and post-commit (auto-push) hooks."""

from __future__ import annotations

import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS_DIR = ROOT / ".git" / "hooks"

PRE_COMMIT = """#!/bin/sh
# Privacy, secret & large-file guard before every commit
if [ -x "./backend/.venv/Scripts/python.exe" ]; then
  PY="./backend/.venv/Scripts/python.exe"
elif [ -x "./backend/.venv/bin/python" ]; then
  PY="./backend/.venv/bin/python"
else
  PY="python"
fi
"$PY" scripts/sync_github.py --check-only
"""

POST_COMMIT = """#!/bin/sh
# Auto-push verified local commits to GitHub origin (set SKIP_AUTO_PUSH=1 to skip)
if [ "$SKIP_AUTO_PUSH" = "1" ]; then
  exit 0
fi
if git remote | grep -qx "origin"; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  git push -u origin "$BRANCH" || echo "[post-commit] Auto-push failed; run 'git push' manually."
fi
"""


def main() -> None:
    if not (ROOT / ".git").exists():
        raise SystemExit("ERROR: .git not found. Run 'git init' first.")
    HOOKS_DIR.mkdir(parents=True, exist_ok=True)
    for name, body in [("pre-commit", PRE_COMMIT), ("post-commit", POST_COMMIT)]:
        path = HOOKS_DIR / name
        path.write_text(body, encoding="utf-8", newline="\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"Installed hook: {path}")


if __name__ == "__main__":
    main()
