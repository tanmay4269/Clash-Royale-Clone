python rl/ppo_trainer.py --profile --no-wandb_logging --num_games_in_buffer 8 --num_envs 8
OPTIMIZE_COLLISION_LISTS=1 python rl/ppo_trainer.py --profile --no-wandb_logging --num_games_in_buffer 8 --num_envs 8
OPTIMIZE_COLLISION_SPATIAL=1 python rl/ppo_trainer.py --profile --no-wandb_logging --num_games_in_buffer 8 --num_envs 8
OPTIMIZE_COLLISION_LISTS=1 OPTIMIZE_COLLISION_SPATIAL=1 python rl/ppo_trainer.py --profile --no-wandb_logging --num_games_in_buffer 8 --num_envs 8