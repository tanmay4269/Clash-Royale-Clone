import numpy as np
import torch as t


class RolloutBuffer:
    def __init__(self, n_steps, gae_gamma, gae_lambda):
        self.n_steps = n_steps
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda

        self.states      = None  # Lazy
        self.actions     = None
        self.log_probs   = t.zeros(n_steps, dtype=t.float32)
        self.rewards     = t.zeros(n_steps, dtype=t.float32)
        self.values      = t.zeros(n_steps, dtype=t.float32)
        self.dones       = t.zeros(n_steps, dtype=t.float32)
        self.advantages  = t.zeros(n_steps, dtype=t.float32)
        self.returns     = t.zeros(n_steps, dtype=t.float32)

        self.ptr = 0

        self.state_shapes, self.action_shapes = None, None  # cache for unfolding


    def push(
        self,
        state, action, log_prob, reward, value, 
        done
    ):
        state = self.flatten_dict(state, "state_shapes")
        action = self.flatten_dict(action, "action_shapes")

        if self.states is None:
            self.states = t.zeros((self.n_steps, state.shape[0]), dtype=t.float32)

        if self.actions is None:
            self.actions = t.zeros((self.n_steps, action.shape[0]), dtype=t.float32)

        self.states[self.ptr]    = state
        self.actions[self.ptr]   = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr]   = reward
        self.values[self.ptr]    = value
        self.dones[self.ptr]     = done

        self.ptr += 1


    def compute_gae(self, last_value, last_done):
        """last_value: float, last_done: float -> None (fills self.advantages, self.returns)"""
        n = self.ptr  # Use actual filled count, not pre-allocated size
        advantage = 0.0

        for i in range(n - 1, -1, -1):
            if i == n - 1:
                next_value = last_value
                next_done = last_done
            else:
                next_value = self.values[i + 1]
                next_done = self.dones[i]

            td_error = self.rewards[i] + self.gae_gamma * next_value * (1 - next_done) - self.values[i]
            advantage = td_error + self.gae_gamma * self.gae_lambda * (1 - next_done) * advantage

            self.advantages[i] = advantage

        # returns BEFORE normalizing advantages
        self.returns[:n] = self.advantages[:n] + self.values[:n]


    def get_minibatches(self, batch_size, obs_norm_mean=None, obs_norm_std=None, value_normalization=False):
        """
        Yields (states, actions, old_log_probs, advantages, returns).

        Advantages are always normalised per-minibatch (zero mean, unit std).

        obs_norm_mean / obs_norm_std: pre-computed EMA tensors (shape D,) from the Trainer,
            applied to the flat state tensor before unflattening.  Pass None (default) to skip.
        value_normalization: if True, normalises returns by the rollout-level mean/std.
            ret_mean and ret_std are always yielded (0.0 / 1.0 when off) so on_batch_update
            can unconditionally denormalise the critic output back to raw return scale for
            explained_variance logging.
        """
        n = self.ptr
        indices = np.random.permutation(n)

        if value_normalization:
            ret_mean = self.returns[:n].mean().item()
            ret_std  = max(self.returns[:n].std().item(), 1e-8)
        else:
            ret_mean, ret_std = 0.0, 1.0

        for start in range(0, n - batch_size + 1, batch_size):
            batch_idx = indices[start : start + batch_size]

            # Per-minibatch advantage normalisation (always on)
            mb_advantages = self.advantages[batch_idx]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Obs normalisation (EMA stats from Trainer, applied consistently at inference and update)
            mb_states = self.states[batch_idx]
            if obs_norm_mean is not None:
                mb_states = (mb_states - obs_norm_mean) / obs_norm_std

            # Value normalisation (rollout-level stats)
            mb_returns = self.returns[batch_idx]
            if value_normalization:
                mb_returns = (mb_returns - ret_mean) / ret_std

            yield (
                self.unflatten_dict(mb_states, self.state_shapes),
                self.unflatten_dict(self.actions[batch_idx], self.action_shapes),
                self.log_probs[batch_idx],
                mb_advantages,
                mb_returns,
            )


    @classmethod
    def merge(cls, buffers):
        """
        Merge multiple per-env buffers (each with correct per-env GAE) 
        into a single buffer for the PPO update.
        """
        filled = [b for b in buffers if b.ptr > 0]
        total = sum(b.ptr for b in filled)

        # Create a shell buffer: we'll overwrite its pre-allocated tensors
        merged = cls(
            n_steps=total, 
            gae_gamma=filled[0].gae_gamma, 
            gae_lambda=filled[0].gae_lambda
        )

        merged.states     = t.cat([b.states[:b.ptr]     for b in filled], dim=0)
        merged.actions    = t.cat([b.actions[:b.ptr]     for b in filled], dim=0)
        merged.log_probs  = t.cat([b.log_probs[:b.ptr]  for b in filled])
        merged.rewards    = t.cat([b.rewards[:b.ptr]     for b in filled])
        merged.values     = t.cat([b.values[:b.ptr]      for b in filled])
        merged.dones      = t.cat([b.dones[:b.ptr]       for b in filled])
        merged.advantages = t.cat([b.advantages[:b.ptr]  for b in filled])
        merged.returns    = t.cat([b.returns[:b.ptr]     for b in filled])

        merged.ptr = total
        merged.state_shapes  = filled[0].state_shapes
        merged.action_shapes = filled[0].action_shapes

        return merged


    def flatten_dict(self, data, cache_attr):
        if getattr(self, cache_attr) is None:
            setattr(self, cache_attr, {
                k: v.shape if isinstance(v, t.Tensor) else None
                for k, v in data.items()
            })

        return t.cat([
            v.flatten() if isinstance(v, t.Tensor) else t.tensor(v)
            for v in data.values()
        ])


    def unflatten_dict(self, flattened_array, shapes):
        unflattened_dict = {}
        current_index = 0
        
        for key, shape in shapes.items():
            if shape is None:
                unflattened_dict[key] = flattened_array[current_index]
                current_index += 1
            else:
                size = int(np.prod(shape))
                
                array_slice = flattened_array[:, current_index : current_index + size]
                unflattened_dict[key] = array_slice.reshape(-1, *shape[1:])
            
                current_index += size
            
        return unflattened_dict


    def forced_skip_mask(self, max_elixirs, deploy_cost_idx):
        """
        Returns a bool tensor (N,) that is True for every step where the agent
        was forced to skip: i.e. elixir < cost of every card in the deck.
        """
        states      = self.unflatten_dict(self.states[: self.ptr], self.state_shapes)
        norm_elixirs = states["elixirs"]                              # (N, 1)
        raw_elixirs  = (norm_elixirs + 1.0) / 2.0 * max_elixirs      # (N, 1)
        hand_normalized_costs = states["my_hand"][..., deploy_cost_idx]  # (N, 4)
        hand_raw_costs = hand_normalized_costs * (max_elixirs / 2.0) + (max_elixirs / 2.0)
        elixir_mask  = hand_raw_costs > raw_elixirs  # (N, 4)
        return elixir_mask.all(dim=-1)                                # (N,)


    def drop_forced_skips(self, max_elixirs, deploy_cost_idx):
        """
        Remove all forced-skip steps in-place and return (n_before, n_dropped).
        After this call, self.ptr reflects the compacted size and get_minibatches
        works unchanged: no other modifications needed.
        """
        if self.ptr == 0 or self.state_shapes is None:
            return 0, 0

        forced_skip = self.forced_skip_mask(max_elixirs, deploy_cost_idx)
        keep     = ~forced_skip
        n_before  = self.ptr
        n_dropped = int(forced_skip.sum().item())

        if n_dropped == 0:
            return n_before, 0

        self.states     = self.states[:n_before][keep]
        self.actions    = self.actions[:n_before][keep]
        self.log_probs  = self.log_probs[:n_before][keep]
        self.rewards    = self.rewards[:n_before][keep]
        self.values     = self.values[:n_before][keep]
        self.dones      = self.dones[:n_before][keep]
        self.advantages = self.advantages[:n_before][keep]
        self.returns    = self.returns[:n_before][keep]
        self.ptr        = n_before - n_dropped

        return n_before, n_dropped


    def __len__(self):
        return self.ptr


    def reset(self):
        self.ptr = 0
