import numpy as np

import torch as t
import torch.nn as nn

from rl.networks.base import BaseActorCritic


class DeepSetsActorCritic(BaseActorCritic):
    """Original ActorCriticV2. The standard Deep Sets architecture with orthogonal init,
    activation selection, optional disjoint actor/critic, and CNN position decoder."""

    def __init__(
        self,
        entity_encoder_in_ch,
        entity_encoder_mid_ch,
        entity_encoder_out_ch,
        trunk_extra_in_ch,
        trunk_mid_ch,

        # activation_fn='relu',
        activation_fn='tanh',
        
        # disjoint_actor_critic=False,
        disjoint_actor_critic=True,
        
        # use_cnn_position_decoder=False,
        use_cnn_position_decoder=True,
        
        # use_last_layer_norms=False,
        use_last_layer_norms=True,
        
        # append_deck_info_to_position_head_input=False,
        append_deck_info_to_position_head_input=True,

        **kwargs,
    ):
        super().__init__(**kwargs)

        self.entity_encoder_out_ch = entity_encoder_out_ch
        self.trunk_mid_ch = trunk_mid_ch
        self.disjoint_actor_critic = disjoint_actor_critic
        self.use_last_layer_norms = use_last_layer_norms
        self.append_deck_info_to_position_head_input = append_deck_info_to_position_head_input

        activation_layer = self.get_activation(activation_fn)

        def make_entity_encoder():
            return nn.Sequential(
                self.layer_init(nn.Linear(entity_encoder_in_ch, entity_encoder_mid_ch)),
                nn.LayerNorm(entity_encoder_mid_ch),
                activation_layer(),

                self.layer_init(nn.Linear(entity_encoder_mid_ch, entity_encoder_mid_ch)),
                nn.LayerNorm(entity_encoder_mid_ch),
                activation_layer(),

                self.layer_init(nn.Linear(entity_encoder_mid_ch, entity_encoder_out_ch)),
                # nn.LayerNorm(entity_encoder_out_ch),
                # activation_layer(),
            )

        def make_trunk():
            return nn.Sequential(
                self.layer_init(nn.Linear(trunk_extra_in_ch + (3 + 3 + 1 + 1 + 4 + 1) * entity_encoder_out_ch, trunk_mid_ch)),
                nn.LayerNorm(trunk_mid_ch),
                activation_layer(),

                self.layer_init(nn.Linear(trunk_mid_ch, trunk_mid_ch)),
                nn.LayerNorm(trunk_mid_ch),
                activation_layer(),
            )

        if not self.disjoint_actor_critic:
            self.shared_entity_encoder = make_entity_encoder()
            self.shared_trunk = make_trunk()
        else:
            self.critic_entity_encoder = make_entity_encoder()
            self.actor_entity_encoder  = make_entity_encoder()
            self.critic_trunk = make_trunk()
            self.actor_trunk  = make_trunk()

        self.critic_head = nn.Sequential(self.layer_init(nn.Linear(trunk_mid_ch, 1), std=1.0))
        self.actor_skip_net = nn.Sequential(self.layer_init(nn.Linear(trunk_mid_ch, 1), std=0.01))
        self.actor_deck_idx_net = nn.Sequential(self.layer_init(nn.Linear(trunk_mid_ch, self.num_cards_in_deck), std=0.01))

        self.actor_position_net = self.make_position_head(
            use_cnn_position_decoder,
            trunk_mid_ch + (self.num_cards_in_deck if self.append_deck_info_to_position_head_input else 0),
            self.position_space_width, self.position_space_height,
            activation_layer,
        )

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        if not self.use_last_layer_norms:
            return layer

        return super().layer_init(layer, std=std, bias_const=bias_const)

    def get_trunk_input(self, obs, all_embeddings, encoder):
        my_card_embeddings       = all_embeddings[:, 0 : self.max_num_cards]
        opponent_card_embeddings = all_embeddings[:, self.max_num_cards : 2 * self.max_num_cards]
        my_crown_tower_embeddings       = all_embeddings[:, 2 * self.max_num_cards : 2 * self.max_num_cards + 3]
        opponent_crown_tower_embeddings = all_embeddings[:, 2 * self.max_num_cards + 3 :]

        def masked_mean(embeddings, entities):
            mask = (entities.abs().sum(dim=-1, keepdim=True) > 0).to(dtype=embeddings.dtype)
            count = mask.sum(dim=1).clamp(min=1.0)
            return (embeddings * mask).sum(dim=1) / count

        hand_embeddings = encoder(obs["my_hand"])
        next_card_embeddings = encoder(obs["my_next_card"].unsqueeze(1)).squeeze(1)

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

        return trunk_input

    def forward(self, obs):
        all_entities = t.cat([
            obs["my_cards"],
            obs["opponent_cards"],
            obs["my_crown_towers"],
            obs["opponent_crown_towers"],
        ], dim=1).to(dtype=t.float32)

        if not self.disjoint_actor_critic:
            all_embeddings = self.shared_entity_encoder(all_entities)
            shared_trunk_input = self.get_trunk_input(obs, all_embeddings, self.shared_entity_encoder)
            shared_trunk_out   = self.shared_trunk(shared_trunk_input)
            critic_head_input  = shared_trunk_out
            actor_heads_inputs = shared_trunk_out
        else:
            critic_embeddings = self.critic_entity_encoder(all_entities)
            actor_embeddings  = self.actor_entity_encoder(all_entities)
            critic_trunk_input = self.get_trunk_input(obs, critic_embeddings, self.critic_entity_encoder)
            actor_trunk_input  = self.get_trunk_input(obs, actor_embeddings, self.actor_entity_encoder)
            critic_head_input  = self.critic_trunk(critic_trunk_input)
            actor_heads_inputs = self.actor_trunk(actor_trunk_input)

        value = self.critic_head(critic_head_input).squeeze(-1)
        skip_logits = self.actor_skip_net(actor_heads_inputs).squeeze(-1)
        deck_logits = self.actor_deck_idx_net(actor_heads_inputs)

        if self.append_deck_info_to_position_head_input:
            actor_heads_inputs = t.cat([actor_heads_inputs, deck_logits], dim=-1)
        pos_logits = self.actor_position_net(actor_heads_inputs)

        return value, skip_logits, deck_logits, pos_logits
