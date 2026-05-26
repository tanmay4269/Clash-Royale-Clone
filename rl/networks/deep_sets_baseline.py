import torch as t
import torch.nn as nn

from rl.networks.base import BaseActorCritic


class DeepSetsBaseline(BaseActorCritic):
    """Original ActorCritic V1. Kept as legacy baseline."""

    def __init__(
        self,
        entity_encoder_in_ch,
        entity_encoder_mid_ch,
        entity_encoder_out_ch,
        trunk_extra_in_ch,
        trunk_mid_ch,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_encoder_in_ch, entity_encoder_mid_ch),
            nn.LayerNorm(entity_encoder_mid_ch),
            nn.ReLU(),

            nn.Linear(entity_encoder_mid_ch, entity_encoder_mid_ch),
            nn.LayerNorm(entity_encoder_mid_ch),
            nn.ReLU(),

            nn.Linear(entity_encoder_mid_ch, entity_encoder_out_ch),
        )

        self.trunk = nn.Sequential(
            nn.Linear(trunk_extra_in_ch + (3 + 3 + 1 + 1 + 4 + 1) * entity_encoder_out_ch, trunk_mid_ch),
            nn.LayerNorm(trunk_mid_ch),
            nn.ReLU(),

            nn.Linear(trunk_mid_ch, trunk_mid_ch),
            nn.LayerNorm(trunk_mid_ch),
            nn.ReLU(),
        )

        self.critic = nn.Sequential(nn.Linear(trunk_mid_ch, 1))
        self.actor_skip_net = nn.Sequential(nn.Linear(trunk_mid_ch, 1))
        self.actor_deck_idx_net = nn.Sequential(nn.Linear(trunk_mid_ch, self.num_cards_in_deck))
        self.actor_position_net = nn.Sequential(
            nn.Linear(trunk_mid_ch, self.position_space_width * self.position_space_height)
        )

    def forward(self, obs):
        all_entities = t.cat([
            obs["my_cards"],
            obs["opponent_cards"],
            obs["my_crown_towers"],
            obs["opponent_crown_towers"],
        ], dim=1).to(dtype=t.float32)

        all_embeddings = self.entity_encoder(all_entities)

        my_card_embeddings       = all_embeddings[:, 0 : self.max_num_cards]
        opponent_card_embeddings = all_embeddings[:, self.max_num_cards : 2 * self.max_num_cards]
        my_crown_tower_embeddings       = all_embeddings[:, 2 * self.max_num_cards : 2 * self.max_num_cards + 3]
        opponent_crown_tower_embeddings = all_embeddings[:, 2 * self.max_num_cards + 3 :]

        def masked_mean(embeddings, entities):
            mask = (entities.abs().sum(dim=-1, keepdim=True) > 0).to(dtype=embeddings.dtype)
            count = mask.sum(dim=1).clamp(min=1.0)
            return (embeddings * mask).sum(dim=1) / count

        hand_embeddings = self.entity_encoder(obs["my_hand"])
        next_card_embeddings = self.entity_encoder(obs["my_next_card"].unsqueeze(1)).squeeze(1)

        trunk_input = t.cat([
            obs["game_completion_fraction"],
            obs["elixirs"],
            my_crown_tower_embeddings.flatten(start_dim=1),
            opponent_crown_tower_embeddings.flatten(start_dim=1),
            masked_mean(my_card_embeddings, obs["my_cards"]),
            masked_mean(opponent_card_embeddings, obs["opponent_cards"]),
            hand_embeddings.flatten(start_dim=1),
            next_card_embeddings,
        ], dim=-1).to(dtype=t.float32)

        trunk_out = self.trunk(trunk_input)

        value = self.critic(trunk_out).squeeze(-1)
        skip_logits = self.actor_skip_net(trunk_out).squeeze(-1)
        deck_logits = self.actor_deck_idx_net(trunk_out)
        pos_logits = self.actor_position_net(trunk_out)

        return value, skip_logits, deck_logits, pos_logits
