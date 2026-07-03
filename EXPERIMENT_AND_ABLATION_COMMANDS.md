# Experiment and Ablation Commands

TLDR: run the single Bash block below from the repo root. It launches clean, non-resume runs for the run-30/run-31 replacement plus a bounded ablation set.

This deliberately does not use `--resume_run`. The old problem was restarted/log-spliced runs, so every command below creates a fresh `runs/<name>_<timestamp>` folder and a matching full terminal log in `experiment_logs/`.

## One Copy-Paste Block

```bash
bash <<'EOF'
set -euo pipefail

cd /Users/tvg/Desktop/01-Projects/04-Game-Dev/03-Clash-Royale

mkdir -p experiment_logs

COMMON_ARGS=(
  --max_steps 10000000
  --num_envs 8
  --num_games_in_buffer 8
  --gae_gamma 0.997
  --tower_damage_reward_scale 2e-4
  --tower_distruction_reward 0.5
  --winning_reward 5.0
  --step_penalty 0.0
  --minibatch_size 256
  --num_ppo_epochs 8
  --save_state_every 100000
)

run_exp() {
  local name="$1"
  shift

  echo
  echo "===== ${name} ====="
  PYTHONUNBUFFERED=1 python -m rl.ppo_trainer \
    --run_name "${name}" \
    "${COMMON_ARGS[@]}" \
    "$@" 2>&1 | tee "experiment_logs/${name}.log"
}

# Clean replacements for the article's incomplete run 30 and run 31 logs.
run_exp rerun30_deepsets_pointer_attention \
  --network_type deep_sets \
  --use_pointer_decoder \
  --use_attention_over_entities

run_exp rerun31_transformer \
  --network_type transformer \
  --use_pointer_decoder

# Network ablations. These are the highest-value architecture comparisons from the article.
run_exp ablate_no_pointer \
  --network_type deep_sets \
  --no-use_pointer_decoder \
  --use_attention_over_entities

run_exp ablate_no_entity_attention \
  --network_type deep_sets \
  --use_pointer_decoder \
  --no-use_attention_over_entities

run_exp ablate_no_pointer_no_attention_old_deepsets_baseline \
  --network_type deep_sets \
  --no-use_pointer_decoder \
  --no-use_attention_over_entities

# Reward ablations. The full-reward baseline is rerun30_deepsets_pointer_attention.
run_exp ablate_reward_outcome_only \
  --network_type deep_sets \
  --use_pointer_decoder \
  --use_attention_over_entities \
  --tower_damage_reward_scale 0.0 \
  --tower_distruction_reward 0.0

run_exp ablate_reward_outcome_plus_hp_damage \
  --network_type deep_sets \
  --use_pointer_decoder \
  --use_attention_over_entities \
  --tower_damage_reward_scale 2e-4 \
  --tower_distruction_reward 0.0

run_exp ablate_reward_outcome_plus_tower_destruction \
  --network_type deep_sets \
  --use_pointer_decoder \
  --use_attention_over_entities \
  --tower_damage_reward_scale 0.0 \
  --tower_distruction_reward 0.5

# Training ablations. These target the biggest historical confounds without running the whole TODO table.
run_exp ablate_gamma_099 \
  --network_type deep_sets \
  --use_pointer_decoder \
  --use_attention_over_entities \
  --gae_gamma 0.99

run_exp ablate_minibatch_2048 \
  --network_type deep_sets \
  --use_pointer_decoder \
  --use_attention_over_entities \
  --minibatch_size 2048
EOF
```

## What Each Run Answers

| Run | Purpose |
|---|---|
| `rerun30_deepsets_pointer_attention` | Clean Deep Sets result for the article table. Also acts as the full-reward, `gamma=0.997`, minibatch-256 baseline. |
| `rerun31_transformer` | Clean Transformer result under the same reward/training settings. |
| `ablate_no_pointer` | Tests the pointer decoder while keeping entity attention. |
| `ablate_no_entity_attention` | Tests entity attention while keeping the pointer decoder. |
| `ablate_no_pointer_no_attention_old_deepsets_baseline` | Reconstructs the older Deep Sets baseline shape for the bundled architecture comparison. |
| `ablate_reward_outcome_only` | Sparse reward baseline: terminal outcome only. |
| `ablate_reward_outcome_plus_hp_damage` | Tests dense HP-damage shaping without tower-destruction bonus. |
| `ablate_reward_outcome_plus_tower_destruction` | Tests discrete tower-destruction shaping without HP-damage shaping. |
| `ablate_gamma_099` | Tests the shorter discount horizon against the `gamma=0.997` baseline. |
| `ablate_minibatch_2048` | Tests whether the old larger minibatch is worse than minibatch 256. |

## Why This Is Not Every TODO

The article lists more ablations than you should actually run immediately. This file picks the ones with direct CLI support and high interpretive value.

Not included in the default block:

- `--kl_early_stopping`: historical evidence already looked bad, and it is less central than the architecture/reward comparisons.
- `--drop_forced_skips`: historically confounded with rollout length, and still less central than the main network/reward questions.
- `--step_penalty`: add later only if the reward ablations show dense reward is worth refining.
- `--obs_normalization` / `--value_normalization`: previous running-normalization behavior looked unstable, so this is not a first-pass ablation.
- Actor-critic sharing, activation function, linear position decoder, card-conditioned position: useful, but lower priority than pointer, attention, Transformer, reward, gamma, and minibatch size.

## Simulator-Fidelity Ablations Are Not Runnable Yet

The HTML also asks for:

- tower symmetry versus asymmetric levels
- custom sudden death versus standard regulation/overtime
- fixed deployment mask versus tower-dependent territory unlock
- shared spell/troop deployment mask versus spell-specific targeting

Those are real ablations, but the current `rl/ppo_trainer.py` CLI does not expose flags for them. They need simulator/config code first. Do not try to fake these with trainer flags; the commands would not be testing the intended simulator changes.

## Practical Notes

- These runs write checkpoints/videos under `runs/`, W&B metadata under `wandb/`, and terminal logs under `experiment_logs/`.
- W&B logging is enabled by default. If you want purely local runs, add `--no-wandb_logging` inside `COMMON_ARGS`.
- The current trainer does not expose `--seed`; all runs use the hard-coded trainer seed `42`. For proper multi-seed reporting, add a seed CLI flag before repeating this matrix.
- The block is wrapped in `bash <<'EOF'` so it is safe to paste from your default zsh while still running with Bash arrays and fail-fast pipeline handling.
