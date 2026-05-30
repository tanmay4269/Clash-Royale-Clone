import wandb
import numpy as np
import torch as t
import plotly.graph_objects as go


class DiagnosticsLogger:
    def __init__(self, cfg, wandb_logging=True):
        self.cfg = cfg
        self.wandb_logging = wandb_logging

        # Resolve num_cards_in_deck safely to handle Dict configurations or missing key
        self.num_cards_in_deck = cfg.network.get("num_cards_in_deck")
        if not isinstance(self.num_cards_in_deck, int):
            self.num_cards_in_deck = cfg.network.get("num_cards_in_hand")
            if not isinstance(self.num_cards_in_deck, int):
                self.num_cards_in_deck = 4

        self.reset_episode_stats()
        self.reset_buffer_stats()

    def reset_episode_stats(self):
        self.current_ep_elixir_sum = 0
        self.current_ep_steps = 0
        self.ep_skips = 0
        self.ep_decks = []
        self.ep_pos_x = []
        self.ep_pos_y = []

    def reset_buffer_stats(self):
        self.buffer_games_completed = 0
        self.buffer_games_terminated = 0
        self.buffer_games_truncated = 0
        self.buffer_towers_killed_by_p1 = 0
        self.buffer_towers_killed_by_p2 = 0

        # Accumulated episode-level stats for aggregation at PPO update time.
        # With parallel envs, multiple episodes can end at the same global_step,
        # so logging each individually causes wandb to silently drop duplicates.
        # Instead we accumulate and log the mean per PPO update.
        self._ep_elos = []
        self._ep_returns = []
        self._ep_scores = []

        # Accumulated per-episode action/game diagnostics (parallel env path)
        self._ep_avg_elixirs = []
        self._ep_skip_ratios = []
        self._ep_deck_indices = []  # flat list across all episodes
        self._ep_pos_x = []
        self._ep_pos_y = []
        self._ep_durations = []

    def on_step(self, action_2, env):
        # Collect metrics for Player 2
        action_2_cpu = {k: v.cpu().numpy() for k, v in action_2.items()}
        self.current_ep_steps += 1
        
        # We unwrap twice to ensure we reach the underlying cr_gym_env if wrapped
        arena = env.unwrapped.arena
        self.current_ep_elixir_sum += arena.player_side_2.elixirs
        
        self.ep_skips += int(action_2_cpu["skip"].item())
        if action_2_cpu["skip"].item() == 0:
            self.ep_decks.append(int(action_2_cpu["deck_idx"].item()))
            pos = int(action_2_cpu["position"].item())
            self.ep_pos_x.append(pos % arena.width)
            self.ep_pos_y.append(pos // arena.width)

    def on_episode_end(self, env, terminated, truncated, current_elo, ep_return, score, global_step):
        self.buffer_games_completed += 1
        if terminated:
            self.buffer_games_terminated += 1
        if truncated:
            self.buffer_games_truncated += 1

        towers_killed_by_p1 = 0
        towers_killed_by_p2 = 0
        for t_name in ["king_tower", "princess_tower_1", "princess_tower_2"]:
            if float(env.unwrapped._cur_obs["player_2_crown_towers"][t_name]["health"][0]) <= 0:
                towers_killed_by_p1 += 1
            if float(env.unwrapped._cur_obs["player_1_crown_towers"][t_name]["health"][0]) <= 0:
                towers_killed_by_p2 += 1

        self.buffer_towers_killed_by_p1 += towers_killed_by_p1
        self.buffer_towers_killed_by_p2 += towers_killed_by_p2

        self._ep_elos.append(current_elo)
        self._ep_returns.append(ep_return)
        self._ep_scores.append(score)

        if self.wandb_logging:
            log_dict = {
                "game_diagnostics/ep_duration_frames": self.current_ep_steps,
                "game_diagnostics/avg_elixir_p1": self.current_ep_elixir_sum / max(1, self.current_ep_steps),
                "action_diagnostics/skip_ratio": self.ep_skips / max(1, self.current_ep_steps),
            }

            if self.ep_decks:
                log_dict["action_diagnostics/deck_idx_hist"] = wandb.Histogram(self.ep_decks)
                
                # Plotly stacked bar chart
                counts = np.bincount(self.ep_decks, minlength=self.num_cards_in_deck)
                proportions = counts / max(1, sum(counts))
                fig = go.Figure(data=[
                    go.Bar(name=f"Card {i}", x=["Deck"], y=[proportions[i]])
                    for i in range(self.num_cards_in_deck)
                ])
                fig.update_layout(barmode='stack', title="Deck Usage Proportions")
                log_dict["action_diagnostics/deck_stacked_chart"] = wandb.Html(fig.to_html(auto_play=False))

            if self.ep_pos_x:
                log_dict["action_diagnostics/pos_x_hist"] = wandb.Histogram(self.ep_pos_x)
                log_dict["action_diagnostics/pos_y_hist"] = wandb.Histogram(self.ep_pos_y)

            wandb.log(log_dict, step=global_step)

        self.reset_episode_stats()

    def on_episode_end_simple(self, terminated, truncated, current_elo, ep_return, score, episode_info=None):
        """
        Lightweight episode-end accumulator for parallel envs.
        Does not require direct env access, and does NOT log to wandb immediately.

        With parallel envs, multiple episodes can end within the same global_step
        tick. Calling wandb.log(..., step=X) multiple times with the same X causes
        silent overwrites: only the last call survives. Instead, we accumulate
        stats here and flush them as averages in on_ppo_update().

        episode_info: the ``info["episode"]`` dict emitted by ClashRoyaleEnv at
            episode end. Contains tower_kills, avg_elixir, skip_ratio, deck_indices,
            pos_x, pos_y, ep_steps.
        """
        self.buffer_games_completed += 1
        if terminated:
            self.buffer_games_terminated += 1
        if truncated:
            self.buffer_games_truncated += 1

        self._ep_elos.append(current_elo)
        self._ep_returns.append(ep_return)
        self._ep_scores.append(score)

        if episode_info is not None:
            self.buffer_towers_killed_by_p1 += episode_info.get("towers_killed_by_p2", 0)
            self.buffer_towers_killed_by_p2 += episode_info.get("towers_killed_by_p1", 0)
            self._ep_avg_elixirs.append(episode_info.get("avg_elixir_p1", 0.0))
            self._ep_skip_ratios.append(episode_info.get("skip_ratio", 0.0))
            self._ep_deck_indices.extend(episode_info.get("deck_indices", []))
            self._ep_pos_x.extend(episode_info.get("pos_x", []))
            self._ep_pos_y.extend(episode_info.get("pos_y", []))
            self._ep_durations.append(episode_info.get("ep_steps", 0))

        # Note: reset_episode_stats() is intentionally NOT called here because
        # on_episode_end_simple does not use the step-by-step accumulators
        # (ep_decks etc.): those live inside the env process. The buffer-level
        # accumulators above are flushed in on_ppo_update().

    def on_ppo_update(self, global_step, buffer, net_curr, net_init, net_prev):
        if not self.wandb_logging:
            self.reset_buffer_stats()
            return
        
        # --- Episode-level aggregates (flushed once per PPO update) ---
        if self._ep_elos:
            wandb.log({
                "elo":    self._ep_elos[-1],          # latest ELO (monotonically updated)
                "return": np.mean(self._ep_returns),  # mean return across all episodes in this buffer
                "score":  np.mean(self._ep_scores),   # win rate across all episodes in this buffer
            }, step=global_step)

        # --- Buffer-level game diagnostics ---
        game_diag = {
            "game_diagnostics/buffer_games_completed":  self.buffer_games_completed,
            "game_diagnostics/buffer_games_truncated":  self.buffer_games_truncated,
            "game_diagnostics/buffer_games_terminated": self.buffer_games_terminated,
            "game_diagnostics/avg_towers_killed_by_p1": self.buffer_towers_killed_by_p1 / max(1, self.buffer_games_completed),
            "game_diagnostics/avg_towers_killed_by_p2": self.buffer_towers_killed_by_p2 / max(1, self.buffer_games_completed),
        }

        if self._ep_avg_elixirs:
            game_diag["game_diagnostics/avg_elixir_p1"] = np.mean(self._ep_avg_elixirs)
        if self._ep_skip_ratios:
            game_diag["action_diagnostics/skip_ratio"] = np.mean(self._ep_skip_ratios)
        if self._ep_durations:
            game_diag["game_diagnostics/avg_ep_duration_frames"] = np.mean(self._ep_durations)
        if self._ep_deck_indices:
            game_diag["action_diagnostics/deck_idx_hist"] = wandb.Histogram(self._ep_deck_indices)
            counts = np.bincount(self._ep_deck_indices, minlength=self.num_cards_in_deck)
            proportions = counts / max(1, sum(counts))
            fig = go.Figure(data=[
                go.Bar(name=f"Card {i}", x=["Deck"], y=[proportions[i]])
                for i in range(self.num_cards_in_deck)
            ])
            fig.update_layout(barmode='stack', title="Deck Usage Proportions")
            game_diag["action_diagnostics/deck_stacked_chart"] = wandb.Html(fig.to_html(auto_play=False))
        if self._ep_pos_x:
            game_diag["action_diagnostics/pos_x_hist"] = wandb.Histogram(self._ep_pos_x)
            game_diag["action_diagnostics/pos_y_hist"] = wandb.Histogram(self._ep_pos_y)

        wandb.log(game_diag, step=global_step)

        self.reset_buffer_stats()

        # --- Per-head policy diagnostics ---
        with t.no_grad():
            batch = next(iter(buffer.get_minibatches(self.cfg.minibatch_size)))
            states_tensor = batch[0]
            device = next(net_curr.parameters()).device
            states_tensor = {k: v.to(device) for k, v in states_tensor.items()}

            fwd_curr = net_curr(states_tensor)
            fwd_init = net_init(states_tensor)
            fwd_prev = net_prev(states_tensor)

            is_pointer = len(fwd_curr) == 3  # Pointer: (value, card_logits, pos_logits)

            if is_pointer:
                # Unified 5-way card distribution (skip token is slot 4)
                _, card_log_curr, pos_log_curr = fwd_curr
                _, card_log_init, pos_log_init = fwd_init
                _, card_log_prev, pos_log_prev = fwd_prev

                card_dist_curr = t.distributions.Categorical(logits=card_log_curr)
                card_dist_init = t.distributions.Categorical(logits=card_log_init)
                card_dist_prev = t.distributions.Categorical(logits=card_log_prev)
                pos_dist_curr  = t.distributions.Categorical(logits=pos_log_curr)
                pos_dist_init  = t.distributions.Categorical(logits=pos_log_init)
                pos_dist_prev  = t.distributions.Categorical(logits=pos_log_prev)

                ent_card = card_dist_curr.entropy().mean().item()
                ent_pos  = pos_dist_curr.entropy().mean().item()

                kl_card_init = t.distributions.kl.kl_divergence(card_dist_curr, card_dist_init).mean().item()
                kl_pos_init  = t.distributions.kl.kl_divergence(pos_dist_curr,  pos_dist_init).mean().item()
                kl_card_prev = t.distributions.kl.kl_divergence(card_dist_curr, card_dist_prev).mean().item()
                kl_pos_prev  = t.distributions.kl.kl_divergence(pos_dist_curr,  pos_dist_prev).mean().item()

                wandb.log({
                    "per_head_diagnostics/entropy/card":             ent_card,
                    "per_head_diagnostics/entropy/position":         ent_pos,
                    "per_head_diagnostics/kl_vs_initial/card":       kl_card_init,
                    "per_head_diagnostics/kl_vs_initial/position":   kl_pos_init,
                    "per_head_diagnostics/kl_vs_pre_update/card":    kl_card_prev,
                    "per_head_diagnostics/kl_vs_pre_update/position": kl_pos_prev,
                }, step=global_step)

            else:
                # DeepSets: separate Bernoulli skip + Categorical deck heads
                _, skip_log_curr, deck_log_curr, pos_log_curr = fwd_curr
                _, skip_log_init, deck_log_init, pos_log_init = fwd_init
                _, skip_log_prev, deck_log_prev, pos_log_prev = fwd_prev

                skip_dist_curr = t.distributions.Bernoulli(logits=skip_log_curr)
                deck_dist_curr = t.distributions.Categorical(logits=deck_log_curr)
                pos_dist_curr  = t.distributions.Categorical(logits=pos_log_curr)

                skip_dist_init = t.distributions.Bernoulli(logits=skip_log_init)
                deck_dist_init = t.distributions.Categorical(logits=deck_log_init)
                pos_dist_init  = t.distributions.Categorical(logits=pos_log_init)

                skip_dist_prev = t.distributions.Bernoulli(logits=skip_log_prev)
                deck_dist_prev = t.distributions.Categorical(logits=deck_log_prev)
                pos_dist_prev  = t.distributions.Categorical(logits=pos_log_prev)

                ent_skip = skip_dist_curr.entropy().mean().item()
                ent_deck = deck_dist_curr.entropy().mean().item()
                ent_pos  = pos_dist_curr.entropy().mean().item()

                kl_skip_init = t.distributions.kl.kl_divergence(skip_dist_curr, skip_dist_init).mean().item()
                kl_deck_init = t.distributions.kl.kl_divergence(deck_dist_curr, deck_dist_init).mean().item()
                kl_pos_init  = t.distributions.kl.kl_divergence(pos_dist_curr,  pos_dist_init).mean().item()

                kl_skip_prev = t.distributions.kl.kl_divergence(skip_dist_curr, skip_dist_prev).mean().item()
                kl_deck_prev = t.distributions.kl.kl_divergence(deck_dist_curr, deck_dist_prev).mean().item()
                kl_pos_prev  = t.distributions.kl.kl_divergence(pos_dist_curr,  pos_dist_prev).mean().item()

                wandb.log({
                    "per_head_diagnostics/entropy/skip":              ent_skip,
                    "per_head_diagnostics/entropy/deck_idx":          ent_deck,
                    "per_head_diagnostics/entropy/position":          ent_pos,
                    "per_head_diagnostics/kl_vs_initial/skip":        kl_skip_init,
                    "per_head_diagnostics/kl_vs_initial/deck_idx":    kl_deck_init,
                    "per_head_diagnostics/kl_vs_initial/position":    kl_pos_init,
                    "per_head_diagnostics/kl_vs_pre_update/skip":     kl_skip_prev,
                    "per_head_diagnostics/kl_vs_pre_update/deck_idx": kl_deck_prev,
                    "per_head_diagnostics/kl_vs_pre_update/position": kl_pos_prev,
                }, step=global_step)
