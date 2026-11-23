"""Package A-Share Alpha Studio into two portable Antigravity-ready bundles under dist/:

1. dist/a-share-alpha-studio-antigravity-plugin.zip
   Official Antigravity Plugin bundle (plugin.json + rules/AGENTS.md + skills/*/SKILL.md + logo.svg)
   plus Windows (install-plugin.ps1) and macOS/Linux (install-plugin.sh) one-click installers.

2. dist/a-share-alpha-studio-portable-full.zip
   Clean, portable full-stack project (backend + frontend + scripts + docker + docs + embedded
   .agents/plugins/a-share-alpha-studio + BOOTSTRAP_ANTIGRAVITY.md), excluding .venv, node_modules,
   caches, secrets (.env) and runtime databases (pass --include-data to snapshot data/ too).

Usage:
    backend\\.venv\\Scripts\\python.exe scripts\\package_for_antigravity.py
    backend\\.venv\\Scripts\\python.exe scripts\\package_for_antigravity.py --include-data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_NAME = "a-share-alpha-studio"
PLUGIN_SRC = ROOT / ".agents" / "plugins" / PLUGIN_NAME
DIST_DIR = ROOT / "dist"

ALLOWED_PLUGIN_FIELDS = {
    "name", "displayName", "version", "description", "logo",
    "suggestedPrompts", "author", "homepage", "repository", "disabled",
}

EXCLUDE_DIRS = {
    ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    ".ruff_cache", ".git", "dist", "build", ".vite",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pyd", ".log"}
EXCLUDE_FILES = {".env"}

INSTALL_PS1 = r"""param(
    [string]$Workspace = ""
)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Src = Join-Path $Here "a-share-alpha-studio"

if (-not (Test-Path (Join-Path $Src "plugin.json"))) {
    throw "未找到 a-share-alpha-studio\plugin.json，请先完整解压本压缩包。"
}

if ($Workspace -ne "") {
    $Target = Join-Path $Workspace ".agents\plugins\a-share-alpha-studio"
    $Scope = "工作区级 ($Workspace)"
} else {
    $Target = Join-Path $env:USERPROFILE ".gemini\config\plugins\a-share-alpha-studio"
    $Scope = "全局 (~/.gemini/config/plugins)"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Target) | Out-Null
if (Test-Path $Target) { Remove-Item -Recurse -Force $Target }
Copy-Item -Recurse -Force $Src $Target

Write-Host "[OK] 已将 A-Share Alpha Studio 插件安装至 $Scope" -ForegroundColor Green
Write-Host "     路径: $Target" -ForegroundColor Cyan
Write-Host "     下一轮对话或打开 Antigravity Customizations 页面即可自动生效（无需重启）。" -ForegroundColor Gray
"""

INSTALL_SH = r"""#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/a-share-alpha-studio"

if [ ! -f "$SRC/plugin.json" ]; then
  echo "ERROR: $SRC/plugin.json not found. Unzip the archive completely first." >&2
  exit 1
fi

if [ "${1:-}" != "" ]; then
  TARGET="$1/.agents/plugins/a-share-alpha-studio"
  SCOPE="workspace ($1)"
else
  TARGET="$HOME/.gemini/config/plugins/a-share-alpha-studio"
  SCOPE="global (~/.gemini/config/plugins)"
fi

mkdir -p "$(dirname "$TARGET")"
rm -rf "$TARGET"
cp -R "$SRC" "$TARGET"

echo "[OK] Installed A-Share Alpha Studio plugin to $SCOPE"
echo "     Path: $TARGET"
echo "     Skills and rules are automatically discovered on the next Antigravity turn."
"""


def validate_plugin() -> dict:
    manifest_path = PLUGIN_SRC / "plugin.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing plugin manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unknown = set(manifest.keys()) - ALLOWED_PLUGIN_FIELDS
    if unknown:
        raise ValueError(f"Unsupported top-level fields in plugin.json: {unknown}")
    if manifest.get("name") != PLUGIN_NAME:
        raise ValueError(f"plugin.json name must match {PLUGIN_NAME}")

    logo_rel = manifest.get("logo")
    if logo_rel and not (PLUGIN_SRC / logo_rel).exists():
        raise FileNotFoundError(f"Declared logo missing: {PLUGIN_SRC / logo_rel}")

    rules_file = PLUGIN_SRC / "rules" / "AGENTS.md"
    if rules_file.exists():
        text = rules_file.read_text(encoding="utf-8").lstrip()
        if text.startswith("---"):
            raise ValueError("Plugin rules/AGENTS.md must be plain markdown with NO YAML frontmatter")

    skills = list((PLUGIN_SRC / "skills").glob("*/SKILL.md"))
    if not skills:
        raise ValueError("Plugin must bundle at least one skills/<name>/SKILL.md")
    return {
        "name": manifest["name"],
        "version": manifest.get("version", "0.1.0"),
        "skills": [p.parent.name for p in skills],
        "has_rules": rules_file.exists(),
        "has_logo": bool(logo_rel),
    }


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def should_skip(rel_path: Path, include_data: bool) -> bool:
    parts = rel_path.parts
    if not parts:
        return True
    if not include_data and parts[0] == "data":
        return True
    for part in parts:
        if part in EXCLUDE_DIRS or part.endswith(".egg-info"):
            return True
    if rel_path.name in EXCLUDE_FILES or rel_path.suffix in EXCLUDE_SUFFIXES:
        return True
    # Avoid packaging DuckDB/SQLite WAL lock files even with --include-data
    if rel_path.name.endswith((".wal", "-wal", "-shm")):
        return True
    return False


def build_plugin_zip() -> tuple[Path, int]:
    out = DIST_DIR / f"{PLUGIN_NAME}-antigravity-plugin.zip"
    count = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(PLUGIN_SRC.rglob("*")):
            if p.is_file():
                arc = Path(PLUGIN_NAME) / p.relative_to(PLUGIN_SRC)
                zf.write(p, arc.as_posix())
                count += 1
        zf.writestr("install-plugin.ps1", INSTALL_PS1)
        zf.writestr("install-plugin.sh", INSTALL_SH)
        count += 2
    return out, count


def build_portable_full_zip(include_data: bool) -> tuple[Path, int]:
    suffix = "portable-with-data" if include_data else "portable-full"
    out = DIST_DIR / f"{PLUGIN_NAME}-{suffix}.zip"
    count = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(ROOT.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT)
            if should_skip(rel, include_data=include_data):
                continue
            arc = Path(PLUGIN_NAME) / rel
            zf.write(p, arc.as_posix())
            count += 1
    return out, count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-data", action="store_true", help="Also snapshot local DuckDB/SQLite/models into portable zip")
    args = ap.parse_args()

    plugin_info = validate_plugin()
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    plugin_zip, plugin_cnt = build_plugin_zip()
    full_zip, full_cnt = build_portable_full_zip(include_data=args.include_data)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "plugin_validation": plugin_info,
        "bundles": [
            {
                "type": "antigravity_plugin",
                "file": plugin_zip.name,
                "size_bytes": plugin_zip.stat().st_size,
                "files_count": plugin_cnt,
                "sha256": sha256_of(plugin_zip),
            },
            {
                "type": "portable_full_workspace",
                "include_data": args.include_data,
                "file": full_zip.name,
                "size_bytes": full_zip.stat().st_size,
                "files_count": full_cnt,
                "sha256": sha256_of(full_zip),
            },
        ],
    }
    manifest_path = DIST_DIR / "bundle-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
