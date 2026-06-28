from __future__ import annotations

import torch as t
import torch.nn as nn

from rl.networks.base import BaseActorCritic


class LegacyAttentionDeepSetsActorCritic(BaseActorCritic):
    """Compatibility model for run-30 attention checkpoints."""

    def __init__(
        self,
        entity_encoder_in_ch,
        entity_encoder_mid_ch=64,
        entity_encoder_out_ch=32,
        trunk_extra_in_ch=2,
        trunk_mid_ch=128,
        trunk_out_ch=128,
        num_cards_in_hand=4,
        max_num_cards=32,
        position_space_width=18,
        position_space_height=32,
        invalid_position_mask=None,
        max_elixirs=10,
        deploy_cost_idx=2,
    ):
        super().__init__(
            activation_fn="tanh",
            use_layer_init=True,
            use_cnn_position_decoder=True,
            use_learned_temperature=False,
            num_cards_in_hand=num_cards_in_hand,
            max_num_cards=max_num_cards,
            position_space_width=position_space_width,
            position_space_height=position_space_height,
            invalid_position_mask=invalid_position_mask,
            max_elixirs=max_elixirs,
            deploy_cost_idx=deploy_cost_idx,
        )

        def make_entity_encoder():
            return nn.Sequential(
                self.layer_init(nn.Linear(entity_encoder_in_ch, entity_encoder_mid_ch)),
                nn.LayerNorm(entity_encoder_mid_ch),
                self.activation_layer(),
                self.layer_init(nn.Linear(entity_encoder_mid_ch, entity_encoder_mid_ch)),
                nn.LayerNorm(entity_encoder_mid_ch),
                self.activation_layer(),
                self.layer_init(nn.Linear(entity_encoder_mid_ch, entity_encoder_out_ch)),
            )

        trunk_in_ch = trunk_extra_in_ch + (3 + 3 + 1 + 1 + num_cards_in_hand + 1) * entity_encoder_out_ch

        def make_trunk():
            return nn.Sequential(
                self.layer_init(nn.Linear(trunk_in_ch, trunk_mid_ch)),
                nn.LayerNorm(trunk_mid_ch),
                self.activation_layer(),
                self.layer_init(nn.Linear(trunk_mid_ch, trunk_out_ch)),
                nn.LayerNorm(trunk_out_ch),
                self.activation_layer(),
            )

        self.critic_entity_encoder = make_entity_encoder()
        self.actor_entity_encoder = make_entity_encoder()
        self.critic_trunk = make_trunk()
        self.actor_trunk = make_trunk()

        self.entity_attention = nn.MultiheadAttention(
            embed_dim=entity_encoder_out_ch,
            num_heads=4,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(entity_encoder_out_ch)

        self.critic_head = nn.Sequential(
            self.layer_init(nn.Linear(trunk_out_ch, 1), std=1.0)
        )
        self.actor_skip_net = nn.Sequential(
            self.layer_init(nn.Linear(trunk_out_ch, 1), std=0.01)
        )
        self.actor_deck_idx_net = nn.Sequential(
            self.layer_init(nn.Linear(trunk_out_ch, num_cards_in_hand), std=0.01)
        )
        self.actor_position_net = self.make_position_head(
            True,
            trunk_out_ch + num_cards_in_hand,
            position_space_width,
            position_space_height,
        )

    def apply_attention(self, embeddings, raw_entities):
        key_padding_mask = raw_entities.abs().sum(dim=-1) == 0
        all_masked = key_padding_mask.all(dim=-1, keepdim=True)
        key_padding_mask = key_padding_mask & ~all_masked
        attn_out, _ = self.entity_attention(
            embeddings,
            embeddings,
            embeddings,
            key_padding_mask=key_padding_mask,
        )
        return self.attention_norm(embeddings + attn_out)

    def get_trunk_input(self, obs, all_embeddings):
        my_card_embeddings = all_embeddings[:, 0 : self.max_num_cards]
        opponent_card_embeddings = all_embeddings[:, self.max_num_cards : 2 * self.max_num_cards]
        my_crown_tower_embeddings = all_embeddings[:, 2 * self.max_num_cards : 2 * self.max_num_cards + 3]
        opponent_crown_tower_embeddings = all_embeddings[:, 2 * self.max_num_cards + 3 : 2 * self.max_num_cards + 6]
        hand_start = 2 * self.max_num_cards + 6
        hand_embeddings = all_embeddings[:, hand_start : hand_start + self.num_cards_in_hand]
        next_card_embedding = all_embeddings[:, hand_start + self.num_cards_in_hand]

        def masked_mean(embeddings, entities):
            mask = (entities.abs().sum(dim=-1, keepdim=True) > 0).to(dtype=embeddings.dtype)
            count = mask.sum(dim=1).clamp(min=1.0)
            return (embeddings * mask).sum(dim=1) / count

        return t.cat(
            [
                obs["game_completion_fraction"],
                obs["elixirs"],
                my_crown_tower_embeddings.flatten(start_dim=1),
                opponent_crown_tower_embeddings.flatten(start_dim=1),
                masked_mean(my_card_embeddings, obs["my_cards"]),
                masked_mean(opponent_card_embeddings, obs["opponent_cards"]),
                hand_embeddings.flatten(start_dim=1),
                next_card_embedding,
            ],
            dim=-1,
        ).to(dtype=t.float32)

    def trunk_outputs(self, obs):
        all_entities = t.cat(
            [
                obs["my_cards"],
                obs["opponent_cards"],
                obs["my_crown_towers"],
                obs["opponent_crown_towers"],
                obs["my_hand"],
                obs["my_next_card"].unsqueeze(1),
            ],
            dim=1,
        ).to(dtype=t.float32)

        critic_embeddings = self.apply_attention(
            self.critic_entity_encoder(all_entities),
            all_entities,
        )
        actor_embeddings = self.apply_attention(
            self.actor_entity_encoder(all_entities),
            all_entities,
        )

        critic_trunk_input = self.get_trunk_input(obs, critic_embeddings)
        actor_trunk_input = self.get_trunk_input(obs, actor_embeddings)
        return self.critic_trunk(critic_trunk_input), self.actor_trunk(actor_trunk_input)

    def get_action_and_value(
        self,
        obs,
        action=None,
        invalid_deck_mask=None,
        invalid_position_mask=None,
    ):
        critic_out, actor_out = self.trunk_outputs(obs)
        value = self.critic_head(critic_out).squeeze(-1)
        skip_logits = self.actor_skip_net(actor_out).squeeze(-1)
        deck_logits = self.actor_deck_idx_net(actor_out)

        if invalid_deck_mask is not None:
            deck_logits = deck_logits.masked_fill(invalid_deck_mask, float("-inf"))

        raw_elixirs, hand_raw_costs = self._denorm_elixirs(obs)
        elixir_mask = hand_raw_costs > raw_elixirs
        deck_logits = deck_logits.masked_fill(elixir_mask, float("-inf"))
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

        deck_one_hot = t.nn.functional.one_hot(
            action_deck.long(),
            num_classes=self.num_cards_in_hand,
        ).to(dtype=actor_out.dtype)
        pos_logits = self.actor_position_net(t.cat([actor_out, deck_one_hot], dim=-1))

        if invalid_position_mask is not None:
            pos_logits = pos_logits.masked_fill(invalid_position_mask, float("-inf"))
        if self.invalid_position_mask is not None:
            pos_logits = pos_logits.masked_fill(self.invalid_position_mask, float("-inf"))

        pos_dist = t.distributions.Categorical(logits=pos_logits)
        if action is None:
            action_pos = pos_dist.sample()
        else:
            action_pos = action["position"].long()

        skip_log_prob = skip_dist.log_prob(action_skip)
        deck_log_prob = deck_dist.log_prob(action_deck)
        pos_log_prob = pos_dist.log_prob(action_pos)
        is_skip = action_skip.bool()
        log_prob = skip_log_prob + t.where(
            is_skip,
            t.zeros_like(deck_log_prob),
            deck_log_prob + pos_log_prob,
        )

        entropy = skip_dist.entropy() + t.where(
            is_skip,
            t.zeros_like(deck_dist.entropy()),
            deck_dist.entropy() + pos_dist.entropy(),
        )

        return {
            "skip": action_skip.detach(),
            "deck_idx": action_deck.detach(),
            "position": action_pos.detach(),
        }, log_prob, entropy, value
