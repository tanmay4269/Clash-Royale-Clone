#!/usr/bin/env python3
"""Analyze checkpoint Elo history from training terminal logs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"^\s*step\s+([0-9]+)\b")
RUN_RE = re.compile(r"wandb:\s+setting up run\s+(.+)$")
STORED_RE = re.compile(r"Stored checkpoint with ELO\s+(-?[0-9]+);\s+checkpoint idx\s+([0-9]+)")
RENAMED_RE = re.compile(r"Renamed checkpoint\s+([0-9]+)\s+from ELO\s+(-?[0-9]+)\s+to\s+(-?[0-9]+)")


@dataclass(frozen=True)
class EloEvent:
    event_index: int
    step: int | None
    checkpoint_index: int
    elo: int
    event_type: str
    previous_elo: int | None = None


def strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


def parse_log(log_path: Path) -> tuple[str, list[EloEvent], Counter[int]]:
    run_name = log_path.stem
    events: list[EloEvent] = []
    games_against: Counter[int] = Counter()
    latest_step: int | None = None

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = strip_ansi(raw_line).strip()

            run_match = RUN_RE.search(line)
            if run_match:
                run_name = run_match.group(1).strip()

            step_match = STEP_RE.search(line)
            if step_match:
                latest_step = int(step_match.group(1))

            stored_match = STORED_RE.search(line)
            if stored_match:
                elo = int(stored_match.group(1))
                checkpoint_index = int(stored_match.group(2))
                events.append(
                    EloEvent(
                        event_index=len(events),
                        step=latest_step,
                        checkpoint_index=checkpoint_index,
                        elo=elo,
                        event_type="stored",
                    )
                )
                continue

            renamed_match = RENAMED_RE.search(line)
            if renamed_match:
                checkpoint_index = int(renamed_match.group(1))
                previous_elo = int(renamed_match.group(2))
                elo = int(renamed_match.group(3))
                events.append(
                    EloEvent(
                        event_index=len(events),
                        step=latest_step,
                        checkpoint_index=checkpoint_index,
                        elo=elo,
                        event_type="renamed",
                        previous_elo=previous_elo,
                    )
                )
                games_against[checkpoint_index] += 1

    return run_name, events, games_against


def write_events_csv(events: list[EloEvent], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event_index", "step", "checkpoint_index", "elo", "event_type", "previous_elo"])
        for event in events:
            writer.writerow(
                [
                    event.event_index,
                    "" if event.step is None else event.step,
                    event.checkpoint_index,
                    event.elo,
                    event.event_type,
                    "" if event.previous_elo is None else event.previous_elo,
                ]
            )


def write_games_csv(games_against: Counter[int], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint_index", "observed_games_against"])
        for checkpoint_index, count in sorted(games_against.items()):
            writer.writerow([checkpoint_index, count])


def plot_elo_history(run_name: str, events: list[EloEvent], output_path: Path) -> None:
    by_checkpoint: dict[int, list[EloEvent]] = defaultdict(list)
    for event in events:
        if event.step is not None:
            by_checkpoint[event.checkpoint_index].append(event)

    fig = go.Figure()
    for checkpoint_index in sorted(by_checkpoint):
        checkpoint_events = by_checkpoint[checkpoint_index]
        fig.add_trace(
            go.Scatter(
                x=[event.step for event in checkpoint_events],
                y=[event.elo for event in checkpoint_events],
                mode="lines+markers",
                name=f"checkpoint {checkpoint_index}",
                marker={"size": 4},
                line={"width": 1.5},
                hovertemplate=(
                    "checkpoint=%{customdata[0]}<br>"
                    "event=%{customdata[1]}<br>"
                    "step=%{x}<br>"
                    "elo=%{y}<extra></extra>"
                ),
                customdata=[
                    [event.checkpoint_index, event.event_type]
                    for event in checkpoint_events
                ],
            )
        )

    fig.update_layout(
        title=f"Checkpoint Elo Over Training: {run_name}",
        xaxis_title="Training step",
        yaxis_title="Checkpoint Elo",
        template="plotly_white",
        hovermode="closest",
        legend_title_text="Checkpoint",
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


def plot_games_histogram(run_name: str, games_against: Counter[int], output_path: Path) -> None:
    checkpoint_indices = sorted(games_against)
    counts = [games_against[index] for index in checkpoint_indices]

    fig = go.Figure(
        data=[
            go.Bar(
                x=checkpoint_indices,
                y=counts,
                hovertemplate="checkpoint=%{x}<br>observed games=%{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=f"Observed Games Against Checkpoints: {run_name}",
        xaxis_title="Checkpoint index",
        yaxis_title="Observed games against",
        template="plotly_white",
        bargap=0.05,
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


def analyze_one_log(log_path: Path, output_root: Path) -> Path:
    run_name, events, games_against = parse_log(log_path)
    run_output_dir = output_root / log_path.stem
    run_output_dir.mkdir(parents=True, exist_ok=True)

    write_events_csv(events, run_output_dir / "checkpoint_elo_events.csv")
    write_games_csv(games_against, run_output_dir / "games_against_checkpoint_index.csv")

    if events:
        plot_elo_history(run_name, events, run_output_dir / "checkpoint_elo_over_training.html")
    if games_against:
        plot_games_histogram(run_name, games_against, run_output_dir / "games_against_checkpoint_index.html")

    summary_path = run_output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"run_name: {run_name}\n")
        f.write(f"log_path: {log_path}\n")
        f.write(f"checkpoint_elo_events: {len(events)}\n")
        f.write(f"checkpoints_seen: {len({event.checkpoint_index for event in events})}\n")
        f.write(f"observed_games_against: {sum(games_against.values())}\n")
        f.write("\n")
        f.write("Note: an observed game against a checkpoint is counted from a logged\n")
        f.write("'Renamed checkpoint ... from ELO ... to ...' event. If a game produced\n")
        f.write("no rounded Elo change and therefore no rename line, it is not visible in\n")
        f.write("the terminal log and cannot be counted by this parser.\n")

    return run_output_dir


def iter_log_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.log"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create checkpoint Elo plots and games-against histograms from training .log files."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="A single .log file or a directory containing .log files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("log_analysis") / "outputs",
        help="Directory where per-run analysis folders will be written.",
    )
    args = parser.parse_args()

    log_paths = iter_log_paths(args.input)
    if not log_paths:
        raise SystemExit(f"No .log files found at {args.input}")

    for log_path in log_paths:
        output_dir = analyze_one_log(log_path, args.output_dir)
        print(f"Wrote {log_path} -> {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
