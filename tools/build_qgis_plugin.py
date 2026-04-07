"""Build a zip archive for the minimal Urbanomy QGIS plugin."""

from __future__ import annotations

import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = REPO_ROOT / "qgis_plugin" / "urbanomy_qgis"
DIST_ROOT = REPO_ROOT / "dist"
ZIP_PATH = DIST_ROOT / "urbanomy_qgis_plugin.zip"


def main() -> None:
    DIST_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(PLUGIN_ROOT.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            relative = path.relative_to(PLUGIN_ROOT.parent)
            archive.write(path, arcname=str(relative))
    print(f"Built QGIS plugin archive: {ZIP_PATH}")


if __name__ == "__main__":
    main()
