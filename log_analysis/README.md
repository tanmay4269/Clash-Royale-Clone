# Log Analysis

Generate per-run checkpoint Elo plots from terminal `.log` files.

## Usage

Analyze all logs in `experiment_logs/`:

```bash
python log_analysis/analyze_run_log.py experiment_logs
```

Analyze one run log:

```bash
python log_analysis/analyze_run_log.py experiment_logs/rerun31_transformer.log
```

By default, outputs are written under `log_analysis/outputs/<log-stem>/`.
Each run folder contains:

- `checkpoint_elo_over_training.html`: checkpoint Elo over training step.
- `games_against_checkpoint_index.html`: bar chart of observed games against each checkpoint index.
- `checkpoint_elo_events.csv`: parsed checkpoint store/rename events.
- `games_against_checkpoint_index.csv`: source data for the histogram.
- `summary.txt`: parse counts and notes.

The histogram counts logged `Renamed checkpoint ... from ELO ... to ...`
events. In this trainer, those lines are emitted when a checkpoint's rounded
Elo changes after a game. If a game produces no rounded Elo change, the terminal
log has no rename line for it, so that game cannot be recovered from the log.
