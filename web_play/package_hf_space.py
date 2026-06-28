from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "hf-space"

PATHS_TO_COPY = [
    "Dockerfile",
    ".dockerignore",
    "pyproject.toml",
    "requirements.txt",
    "assets",
    "game",
    "rl",
    "web_play",
    "saved-runs/run-30.2__env-V2__full-training__attention-model__minibatch-size-256_0528-085514/checkpoints",
    "saved-runs/run-31__transformer-model_0531-020523/checkpoints",
]


def copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
        shutil.copytree(src, dst, ignore=ignore)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    for rel in PATHS_TO_COPY:
        copy_path(ROOT / rel, OUT / rel)

    shutil.copy2(ROOT / "web_play" / "HF_SPACE_README.md", OUT / "README.md")
    print(f"Wrote Hugging Face Space package to {OUT}")


if __name__ == "__main__":
    main()
