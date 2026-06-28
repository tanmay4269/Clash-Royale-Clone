from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


RUN_ARCHITECTURES = {
    "run-30": ("legacy_deep_sets_attention", "Deep Sets, entity attention"),
    "run-30.2": ("legacy_deep_sets_attention", "Deep Sets, entity attention"),
    "run-31": ("transformer", "Transformer"),
}


@dataclass(frozen=True)
class CheckpointInfo:
    id: str
    run_id: str
    run_name: str
    architecture: str
    architecture_label: str
    checkpoint_name: str
    checkpoint_index: int | None
    elo: int | None
    path: str


def _run_id_for(path: Path) -> str:
    match = re.match(r"(run-\d+(?:\.\d+)?)", path.name)
    return match.group(1) if match else path.name


def _architecture_for(run_dir: Path) -> tuple[str, str]:
    run_id = _run_id_for(run_dir)
    if run_id in RUN_ARCHITECTURES:
        return RUN_ARCHITECTURES[run_id]

    lowered = run_dir.name.lower()
    if "transformer" in lowered:
        return "transformer", "Transformer"
    if "attention" in lowered:
        return "legacy_deep_sets_attention", "Deep Sets, entity attention"
    return "deep_sets", "Deep Sets"


def _checkpoint_parts(path: Path) -> tuple[int | None, int | None]:
    match = re.match(r"checkpoint_(\d+)(?:_(\d+))?\.pt$", path.name)
    if not match:
        return None, None
    checkpoint_index = int(match.group(1))
    elo = int(match.group(2)) if match.group(2) else None
    return checkpoint_index, elo


def discover_checkpoints(root: Path) -> list[CheckpointInfo]:
    saved_runs = root / "saved-runs"
    if not saved_runs.exists():
        return []

    checkpoints: list[CheckpointInfo] = []
    for run_dir in sorted(p for p in saved_runs.iterdir() if p.is_dir()):
        run_id = _run_id_for(run_dir)
        if run_id not in {"run-30", "run-30.2", "run-31"}:
            continue

        checkpoint_dir = run_dir / "checkpoints"
        if not checkpoint_dir.exists():
            continue

        architecture, architecture_label = _architecture_for(run_dir)
        for ckpt in sorted(checkpoint_dir.glob("checkpoint_*.pt")):
            checkpoint_index, elo = _checkpoint_parts(ckpt)
            rel_path = ckpt.relative_to(root).as_posix()
            checkpoints.append(
                CheckpointInfo(
                    id=rel_path,
                    run_id=run_id,
                    run_name=run_dir.name,
                    architecture=architecture,
                    architecture_label=architecture_label,
                    checkpoint_name=ckpt.name,
                    checkpoint_index=checkpoint_index,
                    elo=elo,
                    path=str(ckpt),
                )
            )

    return sorted(
        checkpoints,
        key=lambda item: (
            item.run_id,
            -(item.elo if item.elo is not None else -1),
            -(item.checkpoint_index if item.checkpoint_index is not None else -1),
        ),
    )


def best_by_run(checkpoints: list[CheckpointInfo]) -> dict[str, CheckpointInfo]:
    best: dict[str, CheckpointInfo] = {}
    for checkpoint in checkpoints:
        current = best.get(checkpoint.run_id)
        if current is None:
            best[checkpoint.run_id] = checkpoint
            continue
        candidate_key = (
            checkpoint.elo if checkpoint.elo is not None else -1,
            checkpoint.checkpoint_index if checkpoint.checkpoint_index is not None else -1,
        )
        current_key = (
            current.elo if current.elo is not None else -1,
            current.checkpoint_index if current.checkpoint_index is not None else -1,
        )
        if candidate_key > current_key:
            best[checkpoint.run_id] = checkpoint
    return best
