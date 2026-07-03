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
