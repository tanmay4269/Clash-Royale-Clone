python rl/ppo_trainer.py \
    --run_name run_31__ablation_baseline \
    --network_type deep_sets \
    --obs_normalization

python rl/ppo_trainer.py \
    --run_name run_32__ablate__obs_norm \
    --network_type deep_sets \
    --obs_normalization
