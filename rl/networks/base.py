import numpy as np
import torch as t
import torch.nn as nn


class BaseActorCritic(nn.Module):
    def __init__(
        self,
        num_cards_in_deck=4,
        max_num_cards=10,
        position_space_width=18,
        position_space_height=32,
        invalid_position_mask=None,
        max_elixirs=10,
        deploy_cost_idx=2,
        **kwargs,
    ):
        super().__init__()
        self.num_cards_in_deck = num_cards_in_deck
        self.max_num_cards = max_num_cards
        self.position_space_width = position_space_width
        self.position_space_height = position_space_height
        self.invalid_position_mask = invalid_position_mask
        self.max_elixirs = max_elixirs
        self.deploy_cost_idx = deploy_cost_idx

    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    @staticmethod
    def get_activation(activation_fn):
        if activation_fn == 'relu':
            return nn.ReLU
        elif activation_fn == 'tanh':
            return nn.Tanh
        elif activation_fn == 'elu':
            return nn.ELU
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

    def make_position_head(self, use_cnn, input_ch, w, h, activation_layer):
        if not use_cnn:
            return nn.Sequential(
                self.layer_init(nn.Linear(input_ch, w * h), std=0.01)
            )

        assert w == 18 and h == 32, "CNN decoder is hardcoded to 32x18 position space"
        return nn.Sequential(
            nn.Unflatten(1, (input_ch, 1, 1)),
            self.layer_init(nn.ConvTranspose2d(input_ch, 64, kernel_size=(4, 3), stride=1, padding=0)),
            activation_layer(),
            self.layer_init(nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)),
            activation_layer(),
            self.layer_init(nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)),
            activation_layer(),
            self.layer_init(nn.ConvTranspose2d(16, 1, kernel_size=(4, 8), stride=(2, 1), padding=(1, 0)), std=0.01),
            nn.Flatten(),
        )

    def forward(self, obs):
        raise NotImplementedError

    def get_action_and_value(self, obs, action=None, invalid_deck_mask=None, invalid_position_mask=None):
        value, skip_logits, deck_logits, pos_logits = self(obs)

        if invalid_deck_mask is not None:
            deck_logits = deck_logits.masked_fill(invalid_deck_mask, float('-inf'))
        if invalid_position_mask is not None:
            pos_logits = pos_logits.masked_fill(invalid_position_mask, float('-inf'))
        if self.invalid_position_mask is not None:
            pos_logits = pos_logits.masked_fill(self.invalid_position_mask, float('-inf'))

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
        pos_dist  = t.distributions.Categorical(logits=pos_logits)

        if action is None:
            action_skip = skip_dist.sample()
            action_deck = deck_dist.sample()
            action_pos  = pos_dist.sample()
        else:
            action_skip = action["skip"].float()
            action_deck = action["deck_idx"].long()
            action_pos  = action["position"].long()

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
            "position": action_pos.detach()
        }

        return action, log_prob, entropy, value
