import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")  # TODO: fix code later


import os
import glob
import argparse
from copy import deepcopy
import time; os.environ["TZ"] = "Asia/Kolkata"

import random
import numpy as np

from addict import Dict
from rich import print
from tqdm import tqdm

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl.networks import make_network
from rl.networks.botnet import BotNet

import gymnasium as gym
import rl.env.cr_gym_env
from rl.env.cr_flatten_norm_wrapper import CRFlattenNormWrapper
from rl.env.parallel_env import ParallelEnvManager

from rl.rollout_buffer import RolloutBuffer
from rl.checkpoint_management import *
from rl.utils import AverageMeter
from rl.heatmap_visualizer import HeatmapVisualizerWrapper
from rl.diagnostics_logger import DiagnosticsLogger

import wandb


class Trainer:
    def __init__(
        self,

        # Network
        network_type='deep_sets',
        activation_fn='tanh',
        disjoint_actor_critic=True,
        use_cnn_position_decoder=True,
        use_layer_init=True,
        append_deck_info_to_position_head_input=True,
        use_attention_over_entities=True,
        use_pointer_decoder=True,

        # Run / Logging
        gym_env_name="ClashRoyaleEnv-v0",
        run_name=None,
        resume_run=None,
        save_state_every=100_000,

        # Flags / Mode
        use_lr_tuner=True,
        overfit_mode=None,
        wandb_logging=True,
        debug=False,
        profile=False,

        # Reward Shaping
        gae_gamma=0.99,
        step_penalty=0.0,
        tower_damage_reward_scale=1/5000.0,
        tower_distruction_reward=0.5,
        winning_reward=5.0,

        # Environment
        num_envs=1,

        # PPO
        kl_threshold=0.01,
        kl_early_stopping=False,
        obs_normalization=False,
        value_normalization=False,
        num_ppo_epochs=40,
        drop_forced_skips=False,
        num_games_in_buffer=10,

        # Optimisation
        default_lr=1.5e-4,
        minibatch_size=2048,
    ):
        # Seed first: before ANY CUDA init, gym.make, or network construction
        self.cfg = Dict()
        self.cfg.seed = 42
        self.set_seed(self.cfg.seed)

        t.set_num_threads(1)
        t.set_num_interop_threads(1)
        t.set_default_dtype(t.float32)

        if t.cuda.is_available():
            device = t.device("cuda")
        else:
            device = t.device("cpu")

        print(f"Using device: {device}")
        self.device = device

        self.debug = debug or bool(os.environ.get("DEBUG_MODE"))
        if self.debug:
            os.environ["DEBUG_MODE"] = "1"

        self.profile = profile

        self.gym_env_name = gym_env_name
        self.num_envs = num_envs

        self.env_kwargs = {
            "step_penalty": step_penalty,
            "tower_damage_reward_scale": tower_damage_reward_scale,
            "tower_distruction_reward": tower_distruction_reward,
            "winning_reward": winning_reward,
        }

        # Reference env: used for metadata (obs space, arena config) and video recording.
        # Parallel workers run their own independent env instances.
        self.env = gym.make(self.gym_env_name, **self.env_kwargs)
        self.env = CRFlattenNormWrapper(self.env)

        self.arena = self.env.unwrapped.arena
        self.max_num_objects = self.arena.max_num_objects

        #
        occupancy_grid = self.arena.cell_occupancy
        scale = self.arena.tile_size

        tiled_occupancy_grid = np.where(occupancy_grid == 1, 1, 0)[scale//2::scale, scale//2::scale]
        invalid_position_mask = tiled_occupancy_grid.astype(bool).T
        invalid_position_mask[self.arena.height//2 :, :] = True  # bottom/opponent half is always invalid in player-1 view
        
        datetime_str = time.strftime('%m%d-%H%M%S')

        # Run directory layout
        #   Resuming: the run folder already exists; we reuse its name.
        #   New run : create a new timestamped folder under runs/.
        if resume_run:
            self.cfg.run_name = resume_run
        elif run_name:
            self.cfg.run_name = f"{run_name}_{datetime_str}"
        else:
            self.cfg.run_name = datetime_str

        if self.debug and not resume_run:
            self.cfg.run_name = f"DEBUG_{self.cfg.run_name}"

        self.run_dir = os.path.join("runs", self.cfg.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.save_state_every = save_state_every

        # Network Config
        self.cfg.network_type = network_type

        self.cfg.network.entity_encoder_in_ch = self.env.flat_card_space.shape[0]
        self.cfg.network.entity_encoder_mid_ch = 64
        self.cfg.network.entity_encoder_out_ch = 32

        self.cfg.network.trunk_extra_in_ch = 2
        self.cfg.network.trunk_out_ch = 128
        
        self.cfg.network.num_cards_in_hand = self.env.unwrapped.NUM_CARDS_IN_HAND
        self.cfg.network.max_num_cards = self.max_num_objects
        self.cfg.network.position_space_width = self.arena.width
        self.cfg.network.position_space_height = self.arena.height

        self.cfg.network.invalid_position_mask = t.tensor(invalid_position_mask).flatten()
        self.cfg.network.deploy_cost_idx = self.env.flattened_card_space_indices["deploy_cost"][0]

        # CLI Flags
        self.cfg.network.activation_fn = activation_fn
        self.cfg.network.disjoint_actor_critic = disjoint_actor_critic
        self.cfg.network.use_cnn_position_decoder = use_cnn_position_decoder
        self.cfg.network.use_layer_init = use_layer_init
        self.cfg.network.append_deck_info_to_position_head_input = append_deck_info_to_position_head_input
        self.cfg.network.use_attention_over_entities = use_attention_over_entities
        self.cfg.network.use_pointer_decoder = use_pointer_decoder

        # Buffer Related
        self.cfg.buffer.gae_gamma = gae_gamma

        self.cfg.buffer.gae_lambda = 0.95

        self.cfg.buffer.n_steps = int(self.arena.game_duration * 1/self.env.unwrapped.FIXED_DT * num_games_in_buffer)  # Usually 10 to 100 episodes
        # if self.debug:
        #     self.cfg.buffer.n_steps = 2048

        # Elo Rating
        self.cfg.elo.initial_rating = 1200
        self.cfg.elo.scale = 400
        self.cfg.elo.k_factor = 32

        self.current_elo = self.cfg.elo.initial_rating

        # Player Pool
        self.cfg.checkpoint_manager_type = 'elo_based'

        checkpoint_dir = os.path.join(self.run_dir, "checkpoints")

        if self.cfg.checkpoint_manager_type == 'advanced_temporal':
            self.checkpoint_manager = AdvancedTemporal_CheckpointManagement(
                checkpoint_dir=checkpoint_dir,
                loading_latest_ratio=0.5,
                loading_delta_window=0.2,
                min_games_before_checkpointing=100,
                score_queue_size=100,
                avg_score_threshold=0.55,
            )
        elif self.cfg.checkpoint_manager_type == 'elo_based':
            self.checkpoint_manager = AdvancedEloBased_CheckpointManagement(
                checkpoint_dir=checkpoint_dir,
                elo_cfg=self.cfg.elo,
                loading_latest_ratio=0.5,
                min_games_before_checkpointing=100,
                score_queue_size=100,
                avg_score_threshold=0.55
            )

        # PPO Update
        self.cfg.ppo_clip = 0.2
        self.cfg.grad_clip = 0.5

        # Loss
        self.cfg.critic_loss_coef = 0.5
        
        # Entropy
        self.cfg.entropy_loss_coef_initial = 0.01
        self.cfg.entropy_loss_coef_final   = 0.01
        # self.cfg.entropy_loss_coef_final   = 0.001

        self.entropy_loss_coef = self.cfg.entropy_loss_coef_initial

        # LR Finder
        self.cfg.lr_tuner.enabled = use_lr_tuner
        if self.debug:
            self.cfg.lr_tuner.enabled = False

        self.cfg.lr_tuner.min_lr = 1e-7
        self.cfg.lr_tuner.max_lr = 1e-1
        self.cfg.lr_tuner.num_steps = 200
        self.cfg.lr_tuner.pick_lr_factor = 0.3
        
        self.lr_tuned = False

        self.learning_rate = default_lr  # default lr

        # Misc
        self.cfg.minibatch_size = minibatch_size
        if os.environ.get("DEBUG_MODE"):
            self.cfg.minibatch_size = 128

        self.cfg.num_ppo_epochs = num_ppo_epochs  # gradient steps per rollout
        self.cfg.kl_threshold = kl_threshold  # KL early-stop threshold
        self.cfg.kl_early_stopping = kl_early_stopping
        self.cfg.drop_forced_skips = drop_forced_skips
        self.cfg.obs_normalization = obs_normalization
        self.cfg.value_normalization = value_normalization

        self.cfg.max_steps = 10_000_000  # total env steps

        # Running EMA stats for obs normalisation (updated once per rollout, applied at inference)
        # Initialised lazily after the first rollout when the flat-state dimension is known.
        self.obs_norm_mean: t.Tensor | None = None
        self.obs_norm_std:  t.Tensor | None = None
        self._state_shapes = None  # cached buffer state_shapes for obs dict flattening/unflattening


        # None, 'single-buffer', 'fixed-opponent', 'vs-random', 'vs-skip', or 'vs-scripted'
        self.overfit_mode = overfit_mode

        # Replay storing
        self.video_dir = os.path.join(self.run_dir, "videos")
        if self.debug:
            self.video_dir = os.path.join(self.run_dir, "videos", "DEBUG")
        os.makedirs(self.video_dir, exist_ok=True)

        self.video_every_k_global_steps = 20_000
        # if self.debug:
        #     self.video_every_k_global_steps = 5_000

        # WANDB logging
        self.wandb_logging = wandb_logging
        if self.debug:
            self.wandb_logging = False

        if self.wandb_logging:
            wandb.init(
                project="clash_royale-ppo_self_play",
                name=self.cfg.run_name,
                config=self.cfg.to_dict(),
                resume="allow",
                id=self.cfg.run_name,  # stable ID so resuming reattaches to the same wandb run
            )

        # Weights-to-resume
        #   Stored as an attr so train() can restore the full state after
        #   network/optimiser construction.
        self._resume_run = resume_run


    def train(self):
        self.set_seed(self.cfg.seed)

        N = self.num_envs
        ep_returns = []

        net_1, optimiser_1 = self.get_network_and_optimiser()

        # Restore full training state if this is a resumed run
        if self._resume_run:
            global_step = self._load_training_state(net_1, optimiser_1)
            loaded_checkpoint_indices = self.loaded_checkpoint_indices
            opponent_elos = self.opponent_elos
        else:
            global_step = 0
            loaded_checkpoint_indices = [None] * N
            opponent_elos = [self.cfg.elo.initial_rating] * N

        # next_video = 0
        next_video = (global_step // self.video_every_k_global_steps + 1) * self.video_every_k_global_steps
        next_save_state = (global_step // self.save_state_every + 1) * self.save_state_every

        net_1_cpu = make_network(self.cfg.network_type, **self.cfg.network.to_dict()).to("cpu")
        net_1_cpu.load_state_dict(net_1.state_dict())

        initial_net = deepcopy(net_1)
        
        if self.overfit_mode in ['vs-random', 'vs-skip', 'vs-scripted']:
            bot_type = self.overfit_mode.split('-')[1]
            net_2_list = [
                BotNet(
                    bot_type,
                    self.cfg.network.invalid_position_mask, 
                    self.cfg.network.num_cards_in_deck, 
                    self.cfg.network.position_space_width,
                    self.cfg.network.position_space_height
                ) for _ in range(N)
            ]
        else:
            net_2_list = [deepcopy(net_1_cpu) for _ in range(N)]
            if not self.overfit_mode:
                # If resuming or starting, initialize net_2_list by loading appropriate checkpoints
                for i in range(N):
                    ckpt_idx = loaded_checkpoint_indices[i]
                    if ckpt_idx is not None and ckpt_idx in self.checkpoint_manager.stored_by_idx:
                        elo, filename = self.checkpoint_manager.stored_by_idx[ckpt_idx]
                        checkpoint_path = os.path.join(self.checkpoint_manager.checkpoint_dir, filename)
                        net_2_list[i].load_state_dict(t.load(checkpoint_path, map_location='cpu', weights_only=True))
                        opponent_elos[i] = elo
                    else:
                        other_active = {loaded_checkpoint_indices[j] for j in range(N) if j != i and loaded_checkpoint_indices[j] is not None}
                        opponent_elos[i], loaded_checkpoint_indices[i] = self.checkpoint_manager.load(
                            net_2_list[i], self.current_elo, active_checkpoints=other_active
                        )

        # --- Parallel env setup ---
        if N > 1:
            penv = ParallelEnvManager(
                num_envs=N,
                env_name=self.gym_env_name,
                env_kwargs=self.env_kwargs,
                wrapper_cls=CRFlattenNormWrapper,
            )
            seeds = [self.cfg.seed + i for i in range(N)]
            states = penv.reset(seeds)
        else:
            penv = None
            states = [None]  # placeholder
            state_raw, _ = self.env.reset(seed=self.cfg.seed)
            states[0] = state_raw

        # Per-env state tracking
        states_1 = [None] * N  # player 1 obs per env
        states_2 = [None] * N  # player 2 obs per env
        ep_return = [np.zeros(2) for _ in range(N)]
        last_done = [False] * N

        for i in range(N):
            states_1[i], states_2[i] = self.split_observations(states[i])

        self.logger = DiagnosticsLogger(self.cfg, self.wandb_logging)

        profile_update_count = 0
        max_updates = 2 if self.profile else self.cfg.max_steps  # sentinel reused below

        while global_step < self.cfg.max_steps:
            net_1_cpu.load_state_dict(net_1.state_dict())

            # One buffer per env: keeps each env's trajectory contiguous for GAE
            buffers = [RolloutBuffer(**self.cfg.buffer.to_dict()) for _ in range(N)]

            # --- Profile: buffer collection timing ---
            _is_profile_update = self.profile and (profile_update_count == 1)
            _buf_collect_start = time.perf_counter() if _is_profile_update else None
            _frame_times: list[float] = []

            steps_collected = 0
            pbar = tqdm(total=self.cfg.buffer.n_steps, desc="Rollout", disable=not self.profile)

            while steps_collected < self.cfg.buffer.n_steps:
                steps_before = steps_collected

                if global_step >= next_video:
                    self.record_episode(global_step, net_1_cpu, net_2_list[0])
                    next_video += self.video_every_k_global_steps

                # --- Get actions for all envs ---
                with t.no_grad():
                    actions_2 = [None] * N
                    log_probs_2 = [None] * N
                    values_2 = [None] * N
                    actions_1 = [None] * N

                    # Batch active agent (player 2) observations and run once on CPU
                    obs_2_list = [self._normalize_obs(states_2[i]) for i in range(N)]
                    batched_obs_2 = self._batch_observations(obs_2_list)
                    batched_actions_2, batched_log_probs_2, _, batched_values_2 = net_1_cpu.get_action_and_value(batched_obs_2)

                    # Unbatch
                    actions_2 = self._unbatch_actions(batched_actions_2, N)
                    log_probs_2 = [batched_log_probs_2[i] for i in range(N)]
                    values_2 = [batched_values_2[i] for i in range(N)]

                    # Run opponent bots / policies sequentially on CPU
                    for i in range(N):
                        obs_1 = self._normalize_obs(states_1[i])
                        actions_1[i], _, _, _ = net_2_list[i].get_action_and_value(obs_1)

                # --- Build joined actions ---
                joined_actions = [self.join_actions(actions_1[i], actions_2[i]) for i in range(N)]

                # --- Step all envs ---
                _frame_t0 = time.perf_counter() if _is_profile_update else None
                if penv is not None:
                    results = penv.step(joined_actions)
                else:
                    # Single env fallback
                    try:
                        next_state, reward, terminated, truncated, info = self.env.step(joined_actions[0])
                    except Exception as e:
                        print(f"Env step failed: {e}")
                        next_state = states[0]
                        reward = [0.0, 0.0]
                        terminated = True
                        truncated = False
                        info = {}
                    results = [(next_state, reward, terminated, truncated, info)]

                if _is_profile_update:
                    _frame_times.append(time.perf_counter() - _frame_t0)

                # --- Process results from each env ---
                for i in range(N):
                    next_state, reward, terminated, truncated, info = results[i]
                    done = terminated or truncated
                    last_done[i] = terminated

                    if steps_collected < self.cfg.buffer.n_steps:
                        buffers[i].push(
                            states_2[i], actions_2[i], log_probs_2[i], 
                            reward[1], values_2[i], terminated
                        )
                        """
                            If a game ends simply because time ran out (truncated), GAE will set (1 - next_done) to 0. 
                            This incorrectly forces the value of the final state to 0, making the critic think running out 
                            of time violently acts as a massive negative return drop.
                            Solution: PPO should only mask out values if the agent actually "dies" or the game ends artificially.
                        """
                        steps_collected += 1

                    states[i] = next_state
                    states_1[i], states_2[i] = self.split_observations(next_state)

                    ep_return[i] += np.array(reward)

                    if done:
                        # TODO: get this from the env instead
                        score = 0
                        if ep_return[i][1] > ep_return[i][0]:
                            score = 1
                        elif ep_return[i][1] == ep_return[i][0]:
                            score = 0.5

                        # ELO update
                        E_A = 1 / (1 + 10 ** ((opponent_elos[i] - self.current_elo) / self.cfg.elo.scale))
                        self.current_elo = self.current_elo + self.cfg.elo.k_factor * (score - E_A)

                        # Update checkpoint ELO rating on disk using rename
                        ckpt_idx = loaded_checkpoint_indices[i]
                        if ckpt_idx is not None and self.cfg.checkpoint_manager_type == 'elo_based':
                            self.checkpoint_manager.update_checkpoint_elo(ckpt_idx, 1.0 - score, self.current_elo)

                        # Pass episode_info (tower kills, elixir, deck/pos histograms) from the env
                        episode_info = info.get("episode", None)
                        self.logger.on_episode_end_simple(
                            terminated, truncated,
                            self.current_elo, ep_return[i][1], score,
                            episode_info=episode_info,
                        )

                        ep_returns.append(ep_return[i].copy())
                        ep_return[i] = np.zeros(2)

                        ep_seed = self.cfg.seed
                        if self.overfit_mode not in ['fixed-opponent', 'vs-random', 'vs-skip', 'vs-scripted']:
                            other_active = {loaded_checkpoint_indices[j] for j in range(N) if j != i and loaded_checkpoint_indices[j] is not None}
                            opponent_elos[i], loaded_checkpoint_indices[i] = self.checkpoint_manager.load(
                                net_2_list[i], self.current_elo, active_checkpoints=other_active
                            )
                            # Potentially store a new checkpoint of net_1
                            self.checkpoint_manager.update(net_1, score, self.current_elo)
                            ep_seed = np.random.randint(0, 2**31)

                        if penv is not None:
                            states[i] = penv.reset_single(i, seed=ep_seed)
                        else:
                            states[i], _ = self.env.reset(seed=ep_seed)
                        states_1[i], states_2[i] = self.split_observations(states[i])
                        last_done[i] = False  # fresh episode start: bootstrap is valid

                pbar.update(steps_collected - steps_before)
                global_step += N

            pbar.close()

            # Compute GAE independently per env: each buffer has a contiguous trajectory
            _gae_t0 = time.perf_counter() if _is_profile_update else None
            with t.no_grad():
                for i in range(N):
                    if buffers[i].ptr > 0:
                        _, _, _, last_val = net_1_cpu.get_action_and_value(self._normalize_obs(states_1[i]))
                        buffers[i].compute_gae(last_val, last_done[i])
            _gae_time = (time.perf_counter() - _gae_t0) if _is_profile_update else None

            # Merge per-env buffers into one for PPO update
            buffer = RolloutBuffer.merge(buffers)

            # Update EMA obs normalisation stats from this rollout's flat states
            if self.cfg.obs_normalization:
                states_flat = buffer.states[:buffer.ptr]
                rollout_mean = states_flat.mean(dim=0)
                rollout_std  = states_flat.std(dim=0).clamp(min=1e-8)
                if self.obs_norm_mean is None:
                    self.obs_norm_mean = rollout_mean
                    self.obs_norm_std  = rollout_std
                else:
                    momentum = 0.99
                    self.obs_norm_mean = momentum * self.obs_norm_mean + (1 - momentum) * rollout_mean
                    self.obs_norm_std  = momentum * self.obs_norm_std  + (1 - momentum) * rollout_std
                self._state_shapes = buffer.state_shapes

            # --- Drop / log forced-skip steps ---
            if self.cfg.drop_forced_skips:
                n_before, n_dropped = buffer.drop_forced_skips(
                    net_1.max_elixirs, net_1.deploy_cost_idx
                )
                if self.debug:
                    n_after = n_before - n_dropped
                    print(
                        f"(skip-drop) step {global_step:7d} | "
                        f"dropped {n_dropped}/{n_before} "
                        f"({n_dropped / max(n_before, 1):.1%}) → {n_after} kept"
                    )
            elif self.debug:
                mask = buffer.forced_skip_mask(net_1.max_elixirs, net_1.deploy_cost_idx)
                n = buffer.ptr
                n_forced = int(mask.sum().item())
                print(
                    f"(debug) step {global_step:7d} | forced-skip fraction: "
                    f"{n_forced}/{n} = {n_forced / max(n, 1):.3%}"
                )

            # --- Profile: print buffer collection stats ---
            if _is_profile_update:
                _buf_total = time.perf_counter() - _buf_collect_start
                _n_frames = len(_frame_times)
                print("\n" + "="*60)
                print("(PROFILE) Buffer collection stats (update 2/2)")
                print(f"  Total collection time : {_buf_total:.3f}s")
                print(f"  Steps collected       : {steps_collected}")
                print(f"  Time per step         : {_buf_total / max(steps_collected, 1) * 1000:.3f}ms")
                print(f"  Env frames timed      : {_n_frames}")
                if _frame_times:
                    print(f"  Avg frame step time   : {np.mean(_frame_times)*1000:.3f}ms")
                    print(f"  Min frame step time   : {np.min(_frame_times)*1000:.3f}ms")
                    print(f"  Max frame step time   : {np.max(_frame_times)*1000:.3f}ms")
                    print(f"  P95 frame step time   : {np.percentile(_frame_times, 95)*1000:.3f}ms")
                print(f"  GAE compute time      : {_gae_time*1000:.3f}ms")
                print("="*60)

            pre_update_net = deepcopy(net_1)

            # PPO update
            actor_loss, critic_loss, entropy, ratio_mean, advantage_mean, explained_variance, \
                pre_clip_grad_norm, critic_weight_norm, approx_kl, clip_fraction, epochs_completed = \
                self.ppo_update(buffer, net_1, optimiser_1, global_step,
                                profile=_is_profile_update)
            
            self.logger.on_ppo_update(global_step, buffer, net_1, initial_net, pre_update_net)

            profile_update_count += 1
            if self.profile and profile_update_count >= 2:
                print("\n(PROFILE) 2 PPO updates complete: exiting.")
                break

            recent = ep_returns[-100:]
            avg_ep_return = np.mean([r[0] for r in recent]) if recent else float("nan")
            
            print(
                f"step {global_step:7d} | avg(last 100 eps): {avg_ep_return:10.3f} | "
                f"actor_loss: {actor_loss:7.3f} | critic_loss: {critic_loss:7.3f} | entropy: {entropy:.3f}"
            )

            if self.wandb_logging:
                wandb.log({
                    "actor_loss":           actor_loss,
                    "critic_loss":          critic_loss,
                    "entropy":              entropy,
                    "ratio_mean":           ratio_mean,
                    "advantage_mean":       advantage_mean,
                    "explained_variance":   explained_variance,

                    # Critic instability diagnostics
                    "critic_instability_diagnostics/pre_clip_grad_norm":  pre_clip_grad_norm,
                    "critic_instability_diagnostics/critic_weight_norm":  critic_weight_norm,
                    "critic_instability_diagnostics/value_mean":          buffer.values.mean().item(),
                    
                    # PPO update quality diagnostics
                    "ppo_update_diagnostics/clip_fraction":    clip_fraction,
                    "ppo_update_diagnostics/approx_kl":        approx_kl,
                    "ppo_update_diagnostics/epochs_completed": epochs_completed,
                }, step=global_step)

            buffer = None  # free merged buffer

            # Periodic full training-state checkpoint
            if global_step >= next_save_state:
                self._save_training_state(global_step, net_1, optimiser_1, loaded_checkpoint_indices, opponent_elos)
                next_save_state = global_step + self.save_state_every


    def ppo_update(self, buffer, net, optimiser, global_step=0, profile=False):
        if self.cfg.lr_tuner.enabled and not self.lr_tuned:
            self.lr_finder(buffer, net, optimiser)
            self.lr_tuned = True

        # Linear Learning Rate Decay
        frac = max(0.0, 1.0 - (global_step / self.cfg.max_steps))
        current_lr = self.learning_rate * frac
        for param_group in optimiser.param_groups:
            param_group["lr"] = current_lr

        # Linear Entropy Coefficient Decay
        self.entropy_loss_coef = self.cfg.entropy_loss_coef_final + \
            (self.cfg.entropy_loss_coef_initial - self.cfg.entropy_loss_coef_final) * frac

        num_epochs = self.cfg.num_ppo_epochs
        if self.overfit_mode == 'single-buffer':
            num_epochs = 1_000_000

        actor_loss_meter = AverageMeter()
        critic_loss_meter = AverageMeter()
        entropy_meter = AverageMeter()
        ratio_mean_meter = AverageMeter()
        advantage_mean_meter = AverageMeter()
        explained_variance_meter = AverageMeter()
        pre_clip_grad_norm_meter = AverageMeter()
        critic_weight_norm_meter = AverageMeter()
        approx_kl_meter = AverageMeter()
        clip_fraction_meter = AverageMeter()

        ppo_t0 = time.perf_counter() if profile else None
        epoch_times: list[float] = []
        epochs_completed = 0

        for epoch in range(num_epochs):
            epoch_t0 = time.perf_counter() if profile else None
            epoch_approx_kl_meter = AverageMeter()

            for batch in buffer.get_minibatches(
                self.cfg.minibatch_size,
                obs_norm_mean=self.obs_norm_mean if self.cfg.obs_normalization else None,
                obs_norm_std=self.obs_norm_std  if self.cfg.obs_normalization else None,
                value_normalization=self.cfg.value_normalization,
            ):
                actor_loss, critic_loss, entropy, ratio_mean, advantage_mean, explained_variance, \
                    pre_clip_grad_norm, critic_weight_norm, approx_kl, clip_fraction = \
                    self.on_batch_update(
                        net,
                        optimiser,
                        batch,
                        optimize=True,
                    )

                actor_loss_meter.add_sample(actor_loss)
                critic_loss_meter.add_sample(critic_loss)
                entropy_meter.add_sample(entropy)
                ratio_mean_meter.add_sample(ratio_mean)
                advantage_mean_meter.add_sample(advantage_mean)
                explained_variance_meter.add_sample(explained_variance)
                pre_clip_grad_norm_meter.add_sample(pre_clip_grad_norm)
                critic_weight_norm_meter.add_sample(critic_weight_norm)
                approx_kl_meter.add_sample(approx_kl)
                epoch_approx_kl_meter.add_sample(approx_kl)
                clip_fraction_meter.add_sample(clip_fraction)

            if profile:
                epoch_times.append(time.perf_counter() - epoch_t0)

            epochs_completed += 1

            # KL early stopping: exit epoch loop if policy has drifted too far
            epoch_kl = epoch_approx_kl_meter.average()
            if self.cfg.kl_early_stopping and epoch_kl > self.cfg.kl_threshold:
                print(f"(ppo) KL early stop at epoch {epoch} (approx_kl={epoch_kl:.4f} > threshold={self.cfg.kl_threshold})")
                break

            if epoch % 10 == 0 and self.overfit_mode == 'single-buffer':
                window = 10 * len(buffer) // self.cfg.minibatch_size
                _log = {
                    "actor_loss":         actor_loss_meter.window_average(window),
                    "critic_loss":        critic_loss_meter.window_average(window),
                    "entropy":            entropy_meter.window_average(window),
                    "ratio_mean":         ratio_mean_meter.window_average(window),
                    "advantage_mean":     advantage_mean_meter.window_average(window),
                    "explained_variance": explained_variance_meter.window_average(window)
                }

                if self.wandb_logging:
                    wandb.log(_log, step=epoch)
                print(f"Epoch {epoch:7d} | " + " | ".join([f"{k}: {v:.3f}" for k, v in _log.items()]))

        if profile:
            ppo_total = time.perf_counter() - ppo_t0
            n_epochs = len(epoch_times)
            print("\n" + "-"*60)
            print("(PROFILE) PPO update stats (update 2/2)")
            print(f"  num_ppo_epochs        : {n_epochs} / {num_epochs} (KL stop: {epochs_completed < num_epochs})")
            print(f"  Total PPO update time : {ppo_total:.3f}s")
            print(f"  Avg time per epoch    : {np.mean(epoch_times)*1000:.3f}ms")
            print(f"  Min epoch time        : {np.min(epoch_times)*1000:.3f}ms")
            print(f"  Max epoch time        : {np.max(epoch_times)*1000:.3f}ms")
            print(f"  Minibatch size        : {self.cfg.minibatch_size}")
            print(f"  Buffer size           : {len(buffer)}")
            n_batches = max(len(buffer) // self.cfg.minibatch_size, 1)
            print(f"  Batches per epoch     : {n_batches}")
            if epoch_times:
                print(f"  Avg time per batch    : {np.mean(epoch_times)/n_batches*1000:.3f}ms")
            print("-"*60 + "\n")

        return (
            actor_loss_meter.average(),
            critic_loss_meter.average(),
            entropy_meter.average(),
            ratio_mean_meter.average(),
            advantage_mean_meter.average(),
            explained_variance_meter.average(),
            pre_clip_grad_norm_meter.average(),
            critic_weight_norm_meter.average(),
            approx_kl_meter.average(),
            clip_fraction_meter.average(),
            epochs_completed,
        )


    def on_batch_update(self, net, optimiser, batch, optimize=False, scheduler=None):
        device = next(net.parameters()).device
        states, actions, old_log_probs, advantages, returns = batch

        states = {k: v.to(device) for k, v in states.items()}
        actions = {k: v.to(device) for k, v in actions.items()}
        old_log_probs = old_log_probs.to(device)
        advantages = advantages.to(device)
        returns = returns.to(device)

        _, new_log_probs, entropies, values = net.get_action_and_value(states, actions)
        ratio = (new_log_probs - old_log_probs).exp()

        actor_loss = -t.min(
            advantages * ratio,
            advantages * t.clip(ratio, 1 - self.cfg.ppo_clip, 1 + self.cfg.ppo_clip)
        ).mean()

        # returns may already be normalised (rollout-level); critic predicts in the same space
        critic_loss = ((returns - values) ** 2).mean()
        loss = actor_loss + self.cfg.critic_loss_coef * critic_loss - self.entropy_loss_coef * entropies.mean()

        if optimize:
            optimiser.zero_grad()
            loss.backward()
            pre_clip_grad_norm = nn.utils.clip_grad_norm_(net.parameters(), self.cfg.grad_clip).item()
            optimiser.step()
            if scheduler is not None:
                scheduler.step()
        else:
            pre_clip_grad_norm = 0.0

        # Critic weight norm: tracks if weights are drifting unbounded
        critic_weight_norm = sum(
            p.norm() ** 2
            for name, p in net.named_parameters()
            if 'critic' in name
        ).sqrt().item()

        explained_variance = 1 - (returns - values).var() / returns.var()

        with t.no_grad():
            # Approximate KL divergence (numerically stable, second-order estimate)
            # KL(π_old || π_new) ≈ mean((ratio - 1) - log(ratio))
            log_ratio = new_log_probs - old_log_probs
            approx_kl = ((ratio - 1) - log_ratio).mean().item()

            # Best metric to check if the rollout is efficiently being used and not wasted nor overfit.
            # Healthy range: 0.05 - 0.15
            # Below 0.02 is underutilisation; above 0.30 is overfitting.
            clip_fraction = (t.abs(ratio - 1) > self.cfg.ppo_clip).float().mean().item()

        return (
            actor_loss.item(),
            critic_loss.item(),
            entropies.mean().item(),
            ratio.mean().item(),
            advantages.mean().item(),
            explained_variance.item(),
            pre_clip_grad_norm,
            critic_weight_norm,
            approx_kl,
            clip_fraction,
        )


    def lr_finder(self, buffer, net, optimiser):
        print("(lr_finder) Starting learning rate tuning...")

        initial_state = deepcopy(net.state_dict())
        min_lr = self.cfg.lr_tuner.min_lr
        max_lr = self.cfg.lr_tuner.max_lr
        num_steps = self.cfg.lr_tuner.num_steps

        temp_net = deepcopy(net)
        temp_optimiser = optim.Adam(temp_net.parameters(), lr=min_lr)

        def lr_multiplier(step):
            if num_steps <= 1:
                return 1.0
            return (max_lr / min_lr) ** (step / (num_steps - 1))

        lr_scheduler = optim.lr_scheduler.LambdaLR(temp_optimiser, lr_lambda=lr_multiplier)

        minibatches = list(buffer.get_minibatches(self.cfg.minibatch_size))
        if not minibatches:
            raise RuntimeError("lr_finder received no minibatches")

        lr_history = []
        loss_history = []
        
        beta = 0.98  # smoothing factor
        avg_loss = 0.0
        best_loss = 0.0

        for step_idx in range(num_steps):
            batch = minibatches[step_idx % len(minibatches)]
            current_lr = temp_optimiser.param_groups[0]["lr"]

            actor_loss, critic_loss, entropy, _, _, _, _, _, _, _ = self.on_batch_update(
                temp_net,
                temp_optimiser,
                batch,
                optimize=True,
                scheduler=lr_scheduler,
            )

            # raw_loss = actor_loss + self.cfg.critic_loss_coef * critic_loss - self.entropy_loss_coef * entropy

            # In PPO, Actor Objective + Entropy behaves terribly in LR Searches
            # because large steps break the trust region ratio, causing loss to explode instantly.
            # Using just the Critic loss (MSE) behaves like standard supervised regression.
            raw_loss = critic_loss
            
            # Exponentially smooth the loss to handle batch variance
            avg_loss = beta * avg_loss + (1 - beta) * raw_loss
            smoothed_loss = avg_loss / (1 - beta**(step_idx + 1)) # bias correction

            # Stop if loss explodes (diverges)
            if step_idx > 0 and smoothed_loss > 4 * best_loss:
                break
                
            if step_idx == 0 or smoothed_loss < best_loss:
                best_loss = smoothed_loss

            lr_history.append(current_lr)
            loss_history.append(smoothed_loss)

        temp_net.load_state_dict(initial_state)

        if not loss_history:
            raise RuntimeError("(lr_finder) failed to produce loss values")

        min_idx = int(np.argmin(loss_history))
        min_loss_lr = lr_history[min_idx]
        suggested_lr = float(np.clip(min_loss_lr * self.cfg.lr_tuner.pick_lr_factor, min_lr, max_lr))

        for param_group in optimiser.param_groups:
            param_group["lr"] = suggested_lr

        self.learning_rate = suggested_lr

        print(f"(lr_finder) min_loss_lr: {min_loss_lr:.3e} | suggested_lr: {suggested_lr:.3e}")


    def record_episode(self, step, net_1, net_2):
        """Run one greedy episode and save video to self.video_dir."""

        rec_env = gym.make(self.gym_env_name, render_mode="rgb_array")
        rec_env = CRFlattenNormWrapper(rec_env)
        rec_env = HeatmapVisualizerWrapper(rec_env)
        rec_env = gym.wrappers.RecordVideo(
            rec_env,
            video_folder=self.video_dir,
            name_prefix=f"step{step:07d}",
            episode_trigger=lambda _: True,
            disable_logger=True,
        )
        
        state, _ = rec_env.reset()
        ep_return = np.zeros(2)

        try:
            while True:
                state_1, state_2 = self.split_observations(state)

                with t.no_grad():
                    val_1, skip_logits_1, deck_logits_1, pos_logits_1 = net_2(self._normalize_obs(state_1))
                    val_2, skip_logits_2, deck_logits_2, pos_logits_2 = net_1(self._normalize_obs(state_2))

                    if net_2.invalid_position_mask is not None:
                        pos_logits_1 = pos_logits_1.masked_fill(net_2.invalid_position_mask, float('-inf'))
                    if net_1.invalid_position_mask is not None:
                        pos_logits_2 = pos_logits_2.masked_fill(net_1.invalid_position_mask, float('-inf'))

                    skip_probs_1 = t.sigmoid(skip_logits_1).squeeze(0).cpu().numpy()
                    deck_probs_1 = F.softmax(deck_logits_1, dim=-1).squeeze(0).cpu().numpy()
                    pos_probs_1  = F.softmax(pos_logits_1, dim=-1).squeeze(0).cpu().numpy()

                    skip_probs_2 = t.sigmoid(skip_logits_2).squeeze(0).cpu().numpy()
                    deck_probs_2 = F.softmax(deck_logits_2, dim=-1).squeeze(0).cpu().numpy()
                    pos_probs_2  = F.softmax(pos_logits_2, dim=-1).squeeze(0).cpu().numpy()

                    skip_dist_1 = t.distributions.Bernoulli(logits=skip_logits_1)
                    deck_dist_1 = t.distributions.Categorical(logits=deck_logits_1)
                    pos_dist_1 = t.distributions.Categorical(logits=pos_logits_1)

                    skip_dist_2 = t.distributions.Bernoulli(logits=skip_logits_2)
                    deck_dist_2 = t.distributions.Categorical(logits=deck_logits_2)
                    pos_dist_2 = t.distributions.Categorical(logits=pos_logits_2)

                    action_1 = {
                        "skip": skip_dist_1.sample().detach(),
                        "deck_idx": deck_dist_1.sample().detach(),
                        "position": pos_dist_1.sample().detach(),
                    }
                    action_2 = {
                        "skip": skip_dist_2.sample().detach(),
                        "deck_idx": deck_dist_2.sample().detach(),
                        "position": pos_dist_2.sample().detach(),
                    }
            
                action = self.join_actions(action_1, action_2)
                rec_env.env.update(
                    value_1=val_1.squeeze(0).cpu().item(),
                    value_2=val_2.squeeze(0).cpu().item(),
                    skip_prob_1=skip_probs_1.item(),
                    skip_prob_2=skip_probs_2.item(),
                    deck_probs_1=deck_probs_1,
                    deck_probs_2=deck_probs_2,
                    pos_probs_1=pos_probs_1,
                    pos_probs_2=pos_probs_2,
                    action_1=action_1,
                    action_2=action_2,
                )


                state, reward, terminated, truncated, _ = rec_env.step(action)
                ep_return += np.array(reward)

                if terminated or truncated:
                    break

        except Exception as e:
            print(e)
        
        rec_env.close()
        print(f" step {step}:\t return: {ep_return}")

        # Keep only the 10 most recent videos (rolling window)
        videos = sorted(glob.glob(os.path.join(self.video_dir, "*.mp4")))
        for old in videos[:-10]:
            os.remove(old)

        metadatas = sorted(glob.glob(os.path.join(self.video_dir, "*.meta.json")))
        for old in metadatas[:-10]:
            os.remove(old)

        if self.wandb_logging:
            video_path = os.path.join(self.video_dir, f"step{step:07d}-episode-0.mp4")
            if os.path.exists(video_path):
                wandb.log({"video": wandb.Video(video_path, format="mp4")}, step=step)

        return ep_return

    # =================================================================
    # Training-state persistence
    # =================================================================

    def _training_state_path(self):
        return os.path.join(self.run_dir, "training_state.pt")


    def _save_training_state(self, global_step, net_1, optimiser_1, loaded_checkpoint_indices, opponent_elos):
        """Persist enough state to fully resume training."""
        state = {
            "global_step":        global_step,
            "current_elo":        self.current_elo,
            "net_1_state_dict":   net_1.state_dict(),
            "optimiser_state":    optimiser_1.state_dict(),
            "lr_tuned":           self.lr_tuned,
            "learning_rate":      self.learning_rate,
            "entropy_loss_coef":  self.entropy_loss_coef,
            "checkpoint_manager": self.checkpoint_manager.get_state(),
            "loaded_checkpoint_indices": loaded_checkpoint_indices,
            "opponent_elos":             opponent_elos,
        }
        tmp_path = self._training_state_path() + ".tmp"
        t.save(state, tmp_path)
        os.replace(tmp_path, self._training_state_path())  # atomic write
        print(f"(state) saved training state at step {global_step}")


    def _load_training_state(self, net_1, optimiser_1):
        """Load persisted state into an already-constructed net/optimiser.
        Returns the global_step to resume from (0 if no state found)."""
        path = self._training_state_path()
        if not os.path.exists(path):
            print(f"(state) no training_state.pt found in {self.run_dir}: starting fresh")
            return 0

        state = t.load(
            path, 
            weights_only=False, 
            map_location=self.device
        )
        net_1.load_state_dict(state["net_1_state_dict"])
        optimiser_1.load_state_dict(state["optimiser_state"])
        self.current_elo        = state["current_elo"]
        self.lr_tuned           = state["lr_tuned"]
        self.learning_rate      = state["learning_rate"]
        self.entropy_loss_coef  = state["entropy_loss_coef"]
        self.checkpoint_manager.load_state(state["checkpoint_manager"])
        
        self.loaded_checkpoint_indices = state.get("loaded_checkpoint_indices", [None] * self.num_envs)
        self.opponent_elos = state.get("opponent_elos", [self.cfg.elo.initial_rating] * self.num_envs)

        # Adjust loaded lists to match the current num_envs if it was changed on resume
        if len(self.loaded_checkpoint_indices) < self.num_envs:
            self.loaded_checkpoint_indices += [None] * (self.num_envs - len(self.loaded_checkpoint_indices))
        elif len(self.loaded_checkpoint_indices) > self.num_envs:
            self.loaded_checkpoint_indices = self.loaded_checkpoint_indices[:self.num_envs]

        if len(self.opponent_elos) < self.num_envs:
            self.opponent_elos += [self.cfg.elo.initial_rating] * (self.num_envs - len(self.opponent_elos))
        elif len(self.opponent_elos) > self.num_envs:
            self.opponent_elos = self.opponent_elos[:self.num_envs]


        global_step = state["global_step"]
        print(f"(state) resumed from step {global_step} (elo={self.current_elo:.1f})")
        return global_step

    # =================================================================

    def get_network_and_optimiser(self, weights=None):
        """
        Self-Play PPO samples from a pool of policies, so the weights arg 
        inits the network with those
        """

        network = make_network(self.cfg.network_type, **self.cfg.network.to_dict())
        
        if weights:
            network.load_state_dict(t.load(weights, weights_only=True))

        network.to(self.device)
        optimiser = optim.Adam(network.parameters(), lr=self.learning_rate)

        return network, optimiser


    def _batch_observations(self, obs_list):
        batched = {}
        for key in obs_list[0].keys():
            batched[key] = t.cat([obs[key] for obs in obs_list], dim=0)
        return batched


    def _unbatch_actions(self, actions_dict, N):
        return [
            {
                "skip": actions_dict["skip"][i].unsqueeze(0),
                "deck_idx": actions_dict["deck_idx"][i].unsqueeze(0),
                "position": actions_dict["position"][i].unsqueeze(0),
            }
            for i in range(N)
        ]


    def split_observations(self, obs):
        player_1_num_cards = len(obs["player_1_cards"])
        player_2_num_cards = len(obs["player_2_cards"])

        if player_1_num_cards == 0:
            player_1_cards_arr = np.zeros((0, self.env.flat_card_space.shape[0]), dtype=np.float32)
        else:
            player_1_cards_arr = np.array(obs["player_1_cards"], dtype=np.float32)
            
        if player_2_num_cards == 0:
            player_2_cards_arr = np.zeros((0, self.env.flat_card_space.shape[0]), dtype=np.float32)
        else:
            player_2_cards_arr = np.array(obs["player_2_cards"], dtype=np.float32)

        position_x_idx, position_y_idx = np.arange(*self.env.flattened_card_space_indices["position"])

        def pad_cards(cards_arr, num_cards):
            return F.pad(
                t.tensor(cards_arr),  # (X, card_dim)
                (0, 0, 0, self.max_num_objects - num_cards),  # (N, card_dim)
                "constant",
                0,
            ).unsqueeze(0)

        def rotate_entities_180(entities):
            rotated = entities.clone()
            rotated[..., position_x_idx] *= -1
            rotated[..., position_y_idx] *= -1
            return rotated

        player_1_cards = pad_cards(player_1_cards_arr, player_1_num_cards)
        player_2_cards = pad_cards(player_2_cards_arr, player_2_num_cards)

        player_1_crown_towers = t.tensor(np.array(obs["player_1_crown_towers"], dtype=np.float32)).unsqueeze(0)
        player_2_crown_towers = t.tensor(np.array(obs["player_2_crown_towers"], dtype=np.float32)).unsqueeze(0)

        game_completion_fraction = t.tensor(np.array(obs["game_completion_fraction"], dtype=np.float32)).reshape(-1, 1)

        # Network convention: each policy sees itself in player-1 view:
        # self at the top, opponent at the bottom, with the usual top-left x/y origin.
        obs_1 = {
            "game_completion_fraction": game_completion_fraction,
            "elixirs":                  t.tensor(np.array(obs["player_1_elixirs"], dtype=np.float32)).reshape(-1, 1),
            "my_cards":                 player_1_cards,
            "opponent_cards":           player_2_cards,
            "my_crown_towers":          player_1_crown_towers,
            "opponent_crown_towers":    player_2_crown_towers,
            "my_hand":                  t.tensor(np.array(obs["player_1_hand"], dtype=np.float32)).unsqueeze(0),
            "my_next_card":             t.tensor(np.array(obs["player_1_next_card"], dtype=np.float32)).unsqueeze(0),
        }

        obs_2 = {
            "game_completion_fraction": game_completion_fraction,
            "elixirs":                  t.tensor(np.array(obs["player_2_elixirs"], dtype=np.float32)).reshape(-1, 1),
            "my_cards":                 rotate_entities_180(player_2_cards),
            "opponent_cards":           rotate_entities_180(player_1_cards),
            "my_crown_towers":          rotate_entities_180(player_2_crown_towers),
            "opponent_crown_towers":    rotate_entities_180(player_1_crown_towers),
            "my_hand":                  t.tensor(np.array(obs["player_2_hand"], dtype=np.float32)).unsqueeze(0),
            "my_next_card":             t.tensor(np.array(obs["player_2_next_card"], dtype=np.float32)).unsqueeze(0),
        }

        return obs_1, obs_2


    def join_actions(self, action_1, action_2):
        def scalar_int(value):
            if hasattr(value, "detach"):
                return int(value.detach().cpu().item())
            return int(value)

        def policy_position_to_xy(action):
            pos_idx = scalar_int(action["position"])
            return pos_idx % self.arena.width, pos_idx // self.arena.width

        def rotate_xy_180(x, y):
            return self.arena.width - 1 - x, self.arena.height - 1 - y

        return {
            "player_1_skip":          scalar_int(action_1["skip"]),
            "player_1_card_idx":      scalar_int(action_1["deck_idx"]),
            "player_1_card_position": policy_position_to_xy(action_1),

            "player_2_skip":          scalar_int(action_2["skip"]),
            "player_2_card_idx":      scalar_int(action_2["deck_idx"]),
            "player_2_card_position": rotate_xy_180(*policy_position_to_xy(action_2)),
        }

    
    def _normalize_obs(self, obs_dict: dict) -> dict:
        """
        Normalize an obs dict (as produced by split_observations) before it enters
        the network, using the current EMA mean/std over the flat observation space.

        When obs_normalization is disabled or stats are not yet initialised (first
        rollout), the dict is returned unchanged.
        """
        if not self.cfg.obs_normalization or self.obs_norm_mean is None or self._state_shapes is None:
            return obs_dict

        # Flatten to a single (D,) tensor — same concatenation order as RolloutBuffer
        flat = t.cat([v.flatten() for v in obs_dict.values()])  # (D,)
        flat_normed = (flat - self.obs_norm_mean) / self.obs_norm_std

        # Unflatten back to a dict with the original per-key shapes
        result = {}
        offset = 0
        for key, shape in self._state_shapes.items():
            size = int(np.prod(shape))
            result[key] = flat_normed[offset : offset + size].reshape(shape)
            offset += size
        return result


    def set_seed(self, seed: int = 42):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        t.manual_seed(seed)
        t.cuda.manual_seed(seed)
        t.cuda.manual_seed_all(seed)
        t.backends.cudnn.deterministic = True
        t.backends.cudnn.benchmark = False
        t.use_deterministic_algorithms(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO for Clash Royale")

    # Run / Logging
    run_group = parser.add_argument_group("Run / Logging Settings")
    run_group.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Prefix for the run name; date-time will be appended."
    )
    run_group.add_argument(
        "--resume_run",
        type=str,
        default=None,
        metavar="RUN_NAME",
        help=(
            "Name of an existing run folder inside runs/ to resume. "
            "The full training state (net, optimiser, ELO, checkpoint manager) "
            "is restored from training_state.pt inside that folder."
        ),
    )
    run_group.add_argument(
        "--save_state_every",
        type=int,
        default=100_000,
        metavar="STEPS",
        help="Save a full training-state snapshot every N global env steps.",
    )

    # Flags / Mode
    flags_group = parser.add_argument_group("General Flags / Modes")
    flags_group.add_argument(
        "--use_lr_tuner",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable LR tuner."
    )
    flags_group.add_argument(
        "--overfit_mode",
        type=str,
        default=None,
        choices=['single-buffer', 'fixed-opponent', 'vs-random', 'vs-skip', 'vs-scripted'],
        help="Overfit mode to use."
    )
    flags_group.add_argument(
        "--wandb_logging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable wandb logging."
    )
    flags_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode."
    )
    flags_group.add_argument(
        "--profile",
        action="store_true",
        help="Run 2 PPO updates and print detailed timing stats on the 2nd (buffer collection, frame step, GAE, PPO). Exits after."
    )

    # Network Architecture Config Group
    net_group = parser.add_argument_group("Network Architecture Configuration")
    net_group.add_argument(
        "--network_type",
        type=str,
        default='deep_sets',
        choices=['deep_sets', 'transformer'],
        help="Network architecture to use."
    )
    net_group.add_argument(
        "--activation_fn",
        type=str,
        default='tanh',
        choices=['relu', 'tanh', 'elu'],
        help="Activation function to use in networks."
    )
    net_group.add_argument(
        "--disjoint_actor_critic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use disjoint networks for actor and critic."
    )
    net_group.add_argument(
        "--use_cnn_position_decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CNN position decoder (ConvTranspose2d) instead of Linear."
    )
    net_group.add_argument(
        "--use_layer_init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use orthogonal weight initialization."
    )
    net_group.add_argument(
        "--append_deck_info_to_position_head_input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Condition position head on chosen card embedding."
    )
    net_group.add_argument(
        "--use_attention_over_entities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use multi-head attention layer over entities in DeepSets."
    )
    net_group.add_argument(
        "--use_pointer_decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pointer scoring mechanism for deck selection."
    )

    # Reward Shaping
    reward_group = parser.add_argument_group("Reward Shaping Configuration")
    reward_group.add_argument(
        "--gae_gamma",
        type=float,
        default=0.997,
        help="Discount factor (gamma) for GAE."
    )
    reward_group.add_argument(
        "--step_penalty",
        type=float,
        default=0.0,
        help="Penalty applied at each step. A positive value here will add its negative to each step."
    )
    reward_group.add_argument(
        "--tower_damage_reward_scale",
        type=float,
        default=2e-4,
        help="Scale for tower damage reward."
    )
    reward_group.add_argument(
        "--tower_distruction_reward",
        type=float,
        default=0.5,
        help="Reward for destroying a tower."
    )
    reward_group.add_argument(
        "--winning_reward",
        type=float,
        default=5.0,
        help="Reward for winning the game."
    )

    # Environment
    env_group = parser.add_argument_group("Environment Settings")
    env_group.add_argument(
        "--num_envs",
        type=int,
        default=8,
        help="Number of parallel env workers for rollout collection."
    )

    # PPO
    ppo_group = parser.add_argument_group("PPO Algorithm Hyperparameters")
    ppo_group.add_argument(
        "--kl_threshold",
        type=float,
        default=0.01,
        metavar="KL",
        help="Max approximate KL divergence allowed per PPO update. The epoch loop exits early when exceeded. Default: 0.01."
    )
    ppo_group.add_argument(
        "--kl_early_stopping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable early stopping based on KL divergence."
    )
    ppo_group.add_argument(
        "--obs_normalization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalise observations per-minibatch using a running EMA of mean/variance over the state buffer."
    )
    ppo_group.add_argument(
        "--value_normalization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalise return targets before the critic MSE loss using a running EMA of mean/variance (values are denormalised for explained_variance logging)."
    )
    ppo_group.add_argument(
        "--num_ppo_epochs",
        type=int,
        default=6,
        metavar="K",
        help="Number of PPO gradient epochs per rollout (subject to KL early stopping)."
    )
    ppo_group.add_argument(
        "--drop_forced_skips",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Strip forced-skip steps (elixir < cheapest card) from the buffer before the PPO update."
    )
    ppo_group.add_argument(
        "--num_games_in_buffer",
        type=int,
        default=10,
        help="Number of games to keep in the rollout buffer."
    )

    # Optimisation
    opt_group = parser.add_argument_group("Optimization Settings")
    opt_group.add_argument(
        "--default_lr",
        type=float,
        default=1.5e-4,
        help="Default learning rate (or learning rate after LR tuner if enabled)."
    )
    opt_group.add_argument(
        "--minibatch_size",
        type=int,
        default=2048,
        help="Minibatch size for PPO updates."
    )

    args = parser.parse_args()

    trainer = Trainer(
        # Network
        network_type=args.network_type,
        activation_fn=args.activation_fn,
        disjoint_actor_critic=args.disjoint_actor_critic,
        use_cnn_position_decoder=args.use_cnn_position_decoder,
        use_layer_init=args.use_layer_init,
        append_deck_info_to_position_head_input=args.append_deck_info_to_position_head_input,
        use_attention_over_entities=args.use_attention_over_entities,
        use_pointer_decoder=args.use_pointer_decoder,

        # Run / Logging
        gym_env_name="ClashRoyaleEnv-v0",
        run_name=args.run_name,
        resume_run=args.resume_run,
        save_state_every=args.save_state_every,

        # Flags / Mode
        use_lr_tuner=args.use_lr_tuner,
        overfit_mode=args.overfit_mode,
        wandb_logging=args.wandb_logging,
        debug=args.debug,
        profile=args.profile,

        # Reward Shaping
        gae_gamma=args.gae_gamma,
        step_penalty=args.step_penalty,
        tower_damage_reward_scale=args.tower_damage_reward_scale,
        tower_distruction_reward=args.tower_distruction_reward,
        winning_reward=args.winning_reward,

        # Environment
        num_envs=args.num_envs,

        # PPO
        kl_threshold=args.kl_threshold,
        kl_early_stopping=args.kl_early_stopping,
        obs_normalization=args.obs_normalization,
        value_normalization=args.value_normalization,
        num_ppo_epochs=args.num_ppo_epochs,
        drop_forced_skips=args.drop_forced_skips,
        num_games_in_buffer=args.num_games_in_buffer,

        # Optimisation
        default_lr=args.default_lr,
        minibatch_size=args.minibatch_size,
    )

    trainer.train()
