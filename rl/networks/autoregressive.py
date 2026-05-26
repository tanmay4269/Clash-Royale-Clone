import torch as t
import torch.nn as nn

from rl.networks.transformer import TransformerActorCritic


class AutoregressiveActorCritic(TransformerActorCritic):
    """Level 4: Extends Level 3 with autoregressive action decoding — the position
    head is conditioned on the chosen card's embedding (state + card → position)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        activation_layer = self.get_activation(kwargs.get('activation_fn', 'relu'))

        # Override position head: takes trunk_mid_ch + d_model (CLS + chosen card embed)
        self.actor_position_net = self.make_position_head(
            kwargs.get('use_cnn_position_decoder', False),
            self.trunk_mid_ch + self.d_model,
            self.position_space_width, self.position_space_height,
            activation_layer,
        )

    def forward(self, obs):
        trunk_out, hand_out, value, skip_logits, deck_logits = self._encode(obs)
        # Default: condition position on mean of hand embeddings
        mean_hand = hand_out.mean(dim=1)
        pos_input = t.cat([trunk_out, mean_hand], dim=-1)
        pos_logits = self.actor_position_net(pos_input)
        return value, skip_logits, deck_logits, pos_logits

    def get_action_and_value(self, obs, action=None, invalid_deck_mask=None, invalid_position_mask=None):
        trunk_out, hand_out, value, skip_logits, deck_logits = self._encode(obs)

        if invalid_deck_mask is not None:
            deck_logits = deck_logits.masked_fill(invalid_deck_mask, float('-inf'))

        # Elixir masks
        raw_elixirs = (obs["elixirs"] + 1.0) / 2.0 * self.max_elixirs
        hand_normalized_costs = obs["my_hand"][..., self.deploy_cost_idx]
        hand_raw_costs = hand_normalized_costs * (self.max_elixirs / 2.0) + (self.max_elixirs / 2.0)
        elixir_mask = hand_raw_costs > raw_elixirs
        deck_logits = deck_logits.masked_fill(elixir_mask, float('-inf'))

        all_masked = elixir_mask.all(dim=-1)
        if all_masked.any():
            skip_logits = skip_logits.masked_fill(all_masked, 20.0)
            deck_logits = deck_logits.masked_fill(all_masked.unsqueeze(-1), 0.0)

        skip_dist = t.distributions.Bernoulli(logits=skip_logits)
        deck_dist = t.distributions.Categorical(logits=deck_logits)

        if action is None:
            action_skip = skip_dist.sample()
            action_deck = deck_dist.sample()
        else:
            action_skip = action["skip"].float()
            action_deck = action["deck_idx"].long()

        # Autoregressive: compute position logits conditioned on chosen card
        B = trunk_out.shape[0]
        chosen_embed = hand_out[t.arange(B, device=trunk_out.device), action_deck.long()]
        pos_input = t.cat([trunk_out, chosen_embed], dim=-1)
        pos_logits = self.actor_position_net(pos_input)

        if invalid_position_mask is not None:
            pos_logits = pos_logits.masked_fill(invalid_position_mask, float('-inf'))
        if self.invalid_position_mask is not None:
            pos_logits = pos_logits.masked_fill(self.invalid_position_mask, float('-inf'))

        pos_dist = t.distributions.Categorical(logits=pos_logits)

        if action is None:
            action_pos = pos_dist.sample()
        else:
            action_pos = action["position"].long()

        skip_log_prob = skip_dist.log_prob(action_skip)
        deck_log_prob = deck_dist.log_prob(action_deck)
        pos_log_prob  = pos_dist.log_prob(action_pos)

        is_skip = action_skip.bool()
        log_prob = skip_log_prob + t.where(is_skip, t.zeros_like(deck_log_prob), deck_log_prob + pos_log_prob)

        skip_entropy = skip_dist.entropy()
        deck_entropy = deck_dist.entropy()
        pos_entropy  = pos_dist.entropy()
        entropy = skip_entropy + t.where(is_skip, t.zeros_like(deck_entropy), deck_entropy + pos_entropy)

        action = {
            "skip": action_skip.detach(),
            "deck_idx": action_deck.detach(),
            "position": action_pos.detach(),
        }

        return action, log_prob, entropy, value
