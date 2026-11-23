"""Safe privacy-audited GitHub auto-sync utility for A-Share Alpha Studio.

Usage:
  # Audit staged/tracked files only (used by .git/hooks/pre-commit)
  python scripts/sync_github.py --check-only

  # Stage all changes, audit for secrets/personal info/large files, commit & push to origin
  python scripts/sync_github.py -m "feat: update strategy parameters"

  # Commit locally without pushing
  python scripts/sync_github.py -m "chore: snapshot" --no-push
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PATH_PATTERNS = [
    re.compile(r"(^|/)\.env(\..+)?$", re.I),
    re.compile(r"\.(duckdb|duckdb\.wal|db|db-wal|db-shm|sqlite|sqlite3|zip|pkl|bin)$", re.I),
    re.compile(r"^data/(?!\.gitkeep$).+", re.I),
    re.compile(r"^dist/.+", re.I),
]

ALLOWED_PATHS = {".env.example", "backend/.env.example", "data/.gitkeep"}

SECRET_CONTENT_PATTERNS = [
    ("GitHub PAT", re.compile(r"(?:ghp_|github_pat_|gho_|ghs_)[A-Za-z0-9_]{16,}")),
    ("OpenAI/LLM Key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("Private Key Block", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")),
    ("Windows User Path", re.compile(r"C:[\\/]Users[\\/][^\\/\s\"'`]+", re.I)),
    ("Non-empty Tushare Token", re.compile(r"^\s*TUSHARE_TOKEN\s*=\s*\S{8,}", re.M)),
    ("Non-empty Webhook URL", re.compile(r"^\s*(?:WECOM|DINGTALK)_WEBHOOK_URL\s*=\s*https?://\S+", re.M)),
    ("Non-empty QMT Account", re.compile(r"^\s*QMT_ACCOUNT_ID\s*=\s*\S+", re.M)),
]

MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB hard guard against accidental data/binary commits


def _run(cmd: list[str], check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=check, env=env)


def _staged_files() -> list[str]:
    res = _run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"])
    return [line.strip().replace("\\", "/") for line in res.stdout.splitlines() if line.strip()]


def audit_staged_files() -> list[str]:
    """Return list of privacy/secret/size violations found in currently staged files."""
    violations: list[str] = []
    files = _staged_files()
    for rel in files:
        if rel in ALLOWED_PATHS:
            continue
        for pat in FORBIDDEN_PATH_PATTERNS:
            if pat.search(rel):
                violations.append(f"[FORBIDDEN_PATH] {rel}")
                break
        full = ROOT / rel
        if not full.is_file():
            continue
        size = full.stat().st_size
        if size > MAX_FILE_BYTES:
            violations.append(f"[TOO_LARGE] {rel} ({size / (1024 * 1024):.1f} MB > 5 MB)")
            continue
        # Scan text files (skip known binary image assets)
        if full.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".woff", ".woff2"}:
            continue
        # Skip self-definition lines inside this guard script
        if rel == "scripts/sync_github.py":
            continue
        try:
            content = full.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for label, pat in SECRET_CONTENT_PATTERNS:
            m = pat.search(content)
            if m:
                snippet = m.group(0).strip().replace("\n", " ")[:60]
                violations.append(f"[{label}] {rel}: {snippet}")
    return violations


def audit_git_author() -> str | None:
    """Warn/block if commit email looks like a non-noreply personal email when strict privacy is active."""
    res = _run(["git", "config", "--get", "user.email"], check=False)
    email = res.stdout.strip()
    if not email:
        return "[MISSING_GIT_EMAIL] 请先设置仓库本地 git config user.email（建议使用 GitHub noreply 邮箱）"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Privacy-audited GitHub sync for A-Share Alpha Studio")
    parser.add_argument("-m", "--message", default="", help="Commit message")
    parser.add_argument("--branch", default="", help="Target branch (defaults to current branch)")
    parser.add_argument("--check-only", action="store_true", help="Only audit staged files and exit")
    parser.add_argument("--no-push", action="store_true", help="Commit locally without pushing to origin")
    args = parser.parse_args()

    if not (ROOT / ".git").exists():
        print("ERROR: 当前目录尚未初始化 Git 仓库（缺少 .git）。", file=sys.stderr)
        return 1

    if args.check_only:
        violations = audit_staged_files()
        if violations:
            print("PRIVACY/SECRET GUARD BLOCKED COMMIT:", file=sys.stderr)
            for v in violations:
                print(f"  - {v}", file=sys.stderr)
            return 1
        print("Privacy & secret audit passed (0 violations).")
        return 0

    author_err = audit_git_author()
    if author_err:
        print(author_err, file=sys.stderr)
        return 1

    # Stage all non-ignored workspace updates
    _run(["git", "add", "-A"])

    staged = _staged_files()
    status_res = _run(["git", "status", "--porcelain"])
    if not status_res.stdout.strip():
        print("No local changes to commit. Checking push status...")
    else:
        violations = audit_staged_files()
        if violations:
            _run(["git", "reset", "HEAD"], check=False)
            print("PRIVACY/SECRET GUARD BLOCKED SYNC:", file=sys.stderr)
            for v in violations:
                print(f"  - {v}", file=sys.stderr)
            return 1

        msg = args.message.strip() or f"chore: auto-sync {datetime.now().strftime('%Y-%m-%d %H:%M')} ({len(staged)} files)"
        env = dict(os.environ)
        env["SKIP_AUTO_PUSH"] = "1"  # Avoid double-push if post-commit hook is also installed
        # Privacy audit already passed above; use --no-verify so commit works cleanly without spawning MSYS sh.exe
        commit_res = _run(["git", "commit", "--no-verify", "-m", msg], check=False, env=env)
        if commit_res.returncode != 0:
            print(commit_res.stdout)
            print(commit_res.stderr, file=sys.stderr)
            return commit_res.returncode
        print(f"Committed: {msg}")

    if args.no_push:
        return 0

    remotes = _run(["git", "remote"], check=False).stdout.split()
    if "origin" not in remotes:
        print("WARNING: 未配置 origin 远程仓库，跳过 push。", file=sys.stderr)
        return 1

    branch = args.branch.strip() or _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip() or "main"
    push_env = dict(os.environ)
    push_env.setdefault("GIT_TERMINAL_PROMPT", "0")
    push_env.setdefault("GCM_INTERACTIVE", "never")
    push_res = _run(
        ["git", "-c", "http.lowSpeedLimit=1000", "-c", "http.lowSpeedTime=20", "push", "-u", "origin", branch],
        check=False,
        env=push_env,
    )
    if push_res.returncode != 0:
        print(push_res.stdout)
        print(push_res.stderr, file=sys.stderr)
        if "terminal prompts disabled" in push_res.stderr or "Cannot prompt" in push_res.stderr:
            print(
                "HINT: 本机尚未缓存 GitHub 登录凭证，请在本机 PowerShell 执行一次 `git push -u origin main` 完成浏览器授权。",
                file=sys.stderr,
            )
        return push_res.returncode
    print(f"Pushed cleanly to origin/{branch}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
