"""Runtime helpers for loading project dependencies inside QGIS."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def plugin_root() -> Path:
    """Return the QGIS plugin root directory."""
    return Path(__file__).resolve().parent


def runtime_config_path() -> Path:
    """Return the optional runtime config file path."""
    return plugin_root() / "runtime_config.json"


def _prepend_sys_path(path: Path) -> None:
    if not path.exists():
        return
    text = str(path.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)


def _discover_project_venv_paths() -> list[Path]:
    roots = {
        plugin_root(),
        plugin_root().parent,
        plugin_root().parent.parent,
    }
    candidates: list[Path] = []
    for root in roots:
        venv_root = root / ".venv"
        if not venv_root.exists():
            continue
        candidates.extend(sorted((venv_root / "lib").glob("python*/site-packages")))
        windows_site = venv_root / "Lib" / "site-packages"
        if windows_site.exists():
            candidates.append(windows_site)
    return candidates


def _load_runtime_config_paths() -> list[Path]:
    config_file = runtime_config_path()
    if not config_file.exists():
        return []
    try:
        data = json.loads(config_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    raw_paths = data.get("python_paths", [])
    if not isinstance(raw_paths, list):
        return []
    return [Path(str(item)).expanduser() for item in raw_paths if str(item).strip()]


def _load_env_override_paths() -> list[Path]:
    raw = os.environ.get("URBANOMY_QGIS_EXTRA_PYTHONPATH", "")
    if not raw.strip():
        return []
    return [Path(part).expanduser() for part in raw.split(os.pathsep) if part.strip()]


def ensure_runtime_on_path() -> None:
    """Expose external dependency locations to the QGIS Python runtime."""
    for path in _load_env_override_paths():
        _prepend_sys_path(path)
    for path in _load_runtime_config_paths():
        _prepend_sys_path(path)
    for path in _discover_project_venv_paths():
        _prepend_sys_path(path)
