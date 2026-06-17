import numpy as np
import torch as t
import torch.nn as nn

from game.entity import EntityRegistry
from game.entities.troops import Knight, Giant, MiniPEKKA


class BaseActorCritic(nn.Module):
    def __init__(
        self,

        activation_fn='tanh',
        use_layer_init=True,
        use_cnn_position_decoder=True,
        use_learned_temperature=True,

        num_cards_in_hand=4,
        max_num_cards=20,
        position_space_width=18,
        position_space_height=32,

        invalid_position_mask=None,
        max_elixirs=10,
        deploy_cost_idx=2,
    ):
        super().__init__()

        self.activation_fn = activation_fn
        self.use_layer_init = use_layer_init
        self.use_cnn_position_decoder = use_cnn_position_decoder
        self.use_learned_temperature = use_learned_temperature

        self.num_cards_in_hand = num_cards_in_hand
        self.max_num_cards = max_num_cards
        self.position_space_width  = position_space_width
        self.position_space_height = position_space_height

        if invalid_position_mask is not None:
            self.register_buffer("invalid_position_mask", invalid_position_mask)
        else:
            self.invalid_position_mask = None

        self.max_elixirs = max_elixirs
        self.deploy_cost_idx = deploy_cost_idx

        # Learned temperature: each head gets its own temperature parameter.
        # Initialized to 1.0 (no-op). Softplus ensures positivity.
        if self.use_learned_temperature:
            self.log_temp_skip = nn.Parameter(t.zeros(1))  # softplus(0) ≈ 0.693
            self.log_temp_deck = nn.Parameter(t.zeros(1))
            self.log_temp_pos  = nn.Parameter(t.zeros(1))
    

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        if not self.use_layer_init:
            return layer

        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    def activation_layer(self):
        if self.activation_fn == 'relu':
            return nn.ReLU()
        elif self.activation_fn == 'tanh':
            return nn.Tanh()
        elif self.activation_fn == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_fn}")


    def make_position_head(
        self, 
        use_cnn_position_decoder, 
        input_ch, 
        position_space_width, 
        position_space_height
    ):
        if not use_cnn_position_decoder:
            return nn.Sequential(
                self.layer_init(nn.Linear(input_ch, position_space_width * position_space_height), std=0.01)
            )
        
        assert position_space_width == 18 and position_space_height == 32, "CNN decoder is hardcoded to 32x18 position space"

        # * Note: this has way larger capacity than the linear layer version, so ablate carefuly
        return nn.Sequential(
            # (input_ch) -> (input_ch,1,1)
            nn.Unflatten(1, (input_ch, 1, 1)),

            # (input_ch,1,1) -> (64,4,3)
            nn.ConvTranspose2d(
                in_channels=input_ch,
                out_channels=64,
                kernel_size=(4, 3),
                stride=1,
                padding=0
            ),
            nn.ReLU(),

            # (64,4,3) -> (32,8,6)
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # (32,8,6) -> (16,16,12)
            nn.ConvTranspose2d(
                32, 16,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # (16,16,12) -> (1,32,18)
            self.layer_init(nn.ConvTranspose2d(
                16, 1,
                kernel_size=(4, 7),
                stride=(2, 1),
                padding=(1, 0)
            ), std=0.01),

            # (1,32,18) -> (32*18)
            nn.Flatten(),
        )

    
    def _denorm_elixirs(self, obs):
        """Reverse the env's [-1, 1] normalization on elixirs and hand deploy costs.
        Returns (raw_elixirs, hand_raw_costs) both in original [0, max_elixirs] scale."""
        raw_elixirs = (obs["elixirs"] + 1.0) / 2.0 * self.max_elixirs  # (B, 1)
        hand_normalized_costs = obs["my_hand"][..., self.deploy_cost_idx]  # (B, 4)
        hand_raw_costs = hand_normalized_costs * (self.max_elixirs / 2.0) + (self.max_elixirs / 2.0)
        return raw_elixirs, hand_raw_costs


    def forward(self, obs):
        raise NotImplementedError


    def get_action_and_value(
        self, 
        obs, 
        action=None,
        invalid_deck_mask=None,
        invalid_position_mask=None,
    ):
        """
        obs: same as that taken by self.forward
        invalid_deck_mask: based on elixir or something more realistic like CR's random sampling in the deck
        invalid_position_mask: just your half of the arena is deployable into

        Elixir masking (sampling only, action is None):
          - Cards whose deploy cost exceeds current elixirs are masked to -inf.
          - If ALL deck cards are unaffordable, skip is forced (skip_logits → +inf)
            so the agent never wastes an action on a card it can't play.
        """
        value, skip_logits, deck_logits, pos_logits, pos_head_input = self(obs)

        # --- Learned temperature scaling ---
        if self.use_learned_temperature:
            # softplus ensures temperature is always positive
            skip_logits = skip_logits / t.nn.functional.softplus(self.log_temp_skip)
            deck_logits = deck_logits / t.nn.functional.softplus(self.log_temp_deck)
            # pos_logits scaled later (may be None when position head is conditioned on card)

        # --- Static masks ---
        if invalid_deck_mask is not None:
            deck_logits = deck_logits.masked_fill(invalid_deck_mask, float('-inf'))

        # --- Elixir masks ---
        raw_elixirs, hand_raw_costs = self._denorm_elixirs(obs)
        elixir_mask = hand_raw_costs > raw_elixirs  # (B, 4)
        deck_logits = deck_logits.masked_fill(elixir_mask, float('-inf'))

        all_masked = elixir_mask.all(dim=-1)  # (B,): can't afford anything
        if all_masked.any():
            # Use large finite value, NOT inf: Bernoulli(logits=inf).log_prob() → NaN.
            skip_logits = skip_logits.masked_fill(all_masked, 20.0)
            # Zero out fully-masked rows so Categorical doesn't receive all-inf logits.
            deck_logits = deck_logits.masked_fill(all_masked.unsqueeze(-1), 0.0)


        skip_dist = t.distributions.Bernoulli(logits=skip_logits)
        deck_dist = t.distributions.Categorical(logits=deck_logits)

        if action is None:
            action_skip = skip_dist.sample()
            action_deck = deck_dist.sample()
        else:
            action_skip = action["skip"].float()
            action_deck = action["deck_idx"].long()

        if pos_logits is None:
            trunk_out, hand_embeddings = pos_head_input
            B = trunk_out.shape[0]
            chosen_embed = hand_embeddings[t.arange(B, device=trunk_out.device), action_deck.long()]
            pos_logits = self.actor_position_net(
                t.cat([trunk_out, chosen_embed], dim=-1)
            )

        # Apply temperature to position logits (deferred from above when pos_logits was None)
        if self.use_learned_temperature:
            pos_logits = pos_logits / t.nn.functional.softplus(self.log_temp_pos)

        if invalid_position_mask is not None:
            pos_logits = pos_logits.masked_fill(invalid_position_mask, float('-inf'))
        if self.invalid_position_mask is not None:
            pos_logits = pos_logits.masked_fill(self.invalid_position_mask, float('-inf'))

        pos_dist = t.distributions.Categorical(logits=pos_logits)

        if action is None:
            action_pos = pos_dist.sample()
        else:
            action_pos = action["position"].long()

        # Log Probs
        skip_log_prob = skip_dist.log_prob(action_skip)
        deck_log_prob = deck_dist.log_prob(action_deck)
        pos_log_prob  = pos_dist.log_prob(action_pos)

        # Use where instead of multiplication: 0.0 * -inf = NaN in PyTorch.
        # When skipping, deck/pos choices are irrelevant: their log_probs are 0.
        is_skip = action_skip.bool()
        log_prob = skip_log_prob + t.where(is_skip, t.zeros_like(deck_log_prob), deck_log_prob + pos_log_prob)

        # Entropy
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
