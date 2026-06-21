import numpy as np
import torch as t
import torch.nn as nn

from rl.networks.base import BaseActorCritic


class DeepSetsActorCritic(BaseActorCritic):
    def __init__(
        self, 

        entity_encoder_in_ch, 
        entity_encoder_mid_ch, 
        entity_encoder_out_ch,

        trunk_extra_in_ch,
        trunk_out_ch,

        activation_fn='tanh',
        disjoint_actor_critic=True,
        use_cnn_position_decoder=True,
        use_layer_init=True,
        use_learned_temperature=True,
        append_deck_info_to_position_head_input=True,
        use_attention_over_entities=True,
        use_pointer_decoder=True,

        num_cards_in_hand=4,
        max_num_cards=20,
        position_space_width=18,
        position_space_height=32,

        invalid_position_mask=None,
        max_elixirs=10,
        deploy_cost_idx=2,
    ):
        super().__init__(
            activation_fn=activation_fn,
            use_layer_init=use_layer_init,

            use_cnn_position_decoder=use_cnn_position_decoder,
            use_learned_temperature=use_learned_temperature,

            num_cards_in_hand=num_cards_in_hand,
            max_num_cards=max_num_cards,
            position_space_width=position_space_width,
            position_space_height=position_space_height,

            invalid_position_mask=invalid_position_mask,
            max_elixirs=max_elixirs,
            deploy_cost_idx=deploy_cost_idx,
        )

        self.disjoint_actor_critic = disjoint_actor_critic
        self.append_deck_info_to_position_head_input = append_deck_info_to_position_head_input
        self.use_attention_over_entities = use_attention_over_entities
        self.use_pointer_decoder = use_pointer_decoder

        def make_entity_encoder():
            return nn.Sequential(
                self.layer_init(nn.Linear(entity_encoder_in_ch, entity_encoder_mid_ch)),
                nn.LayerNorm(entity_encoder_mid_ch),
                self.activation_layer(),

                self.layer_init(nn.Linear(entity_encoder_mid_ch, entity_encoder_mid_ch)),
                nn.LayerNorm(entity_encoder_mid_ch),
                self.activation_layer(),

                self.layer_init(nn.Linear(entity_encoder_mid_ch, entity_encoder_out_ch)),
                nn.LayerNorm(entity_encoder_out_ch),
                self.activation_layer(),
            )

        def make_trunk():
            trunk_in_ch = trunk_extra_in_ch + (3 + 3 + 1 + 1 + 4 + 1) * entity_encoder_out_ch
            trunk_mid_ch = trunk_in_ch // 2 

            return nn.Sequential(
                self.layer_init(nn.Linear(trunk_in_ch, trunk_mid_ch)),
                nn.LayerNorm(trunk_mid_ch),
                self.activation_layer(),

                self.layer_init(nn.Linear(trunk_mid_ch, trunk_mid_ch)),
                nn.LayerNorm(trunk_mid_ch),
                self.activation_layer(),

                self.layer_init(nn.Linear(trunk_mid_ch, trunk_out_ch)),
                nn.LayerNorm(trunk_out_ch),
                self.activation_layer(),
            )

        if not self.disjoint_actor_critic:
            self.shared_entity_encoder = make_entity_encoder()
            self.shared_trunk = make_trunk()
        else:
            self.critic_entity_encoder = make_entity_encoder()
            self.actor_entity_encoder  = make_entity_encoder()

            self.critic_trunk = make_trunk()
            self.actor_trunk = make_trunk()

        if self.use_attention_over_entities:
            def make_attention_layer():
                return nn.MultiheadAttention(
                    embed_dim=entity_encoder_out_ch,
                    num_heads=4,
                    batch_first=True,
                )
            
            if not self.disjoint_actor_critic:
                self.shared_entity_attention = make_attention_layer()
                self.shared_attention_norm = nn.LayerNorm(entity_encoder_out_ch)
            else:
                self.critic_entity_attention = make_attention_layer()
                self.actor_entity_attention  = make_attention_layer()

                self.critic_attention_norm = nn.LayerNorm(entity_encoder_out_ch)
                self.actor_attention_norm = nn.LayerNorm(entity_encoder_out_ch)


        # Specific Heads
        self.critic_head = nn.Sequential(
            self.layer_init(nn.Linear(trunk_out_ch, 1), std=1.0)
        )

        if not self.use_pointer_decoder:
            self.actor_skip_net = nn.Sequential(
                self.layer_init(nn.Linear(trunk_out_ch, 1), std=0.01)
            )

            self.actor_deck_idx_net = nn.Sequential(
                self.layer_init(nn.Linear(trunk_out_ch, num_cards_in_hand), std=0.01)
            )
        else:
            self.skip_token = nn.Parameter(t.zeros(entity_encoder_out_ch))
            self.pointer_query = self.layer_init(nn.Linear(trunk_out_ch, entity_encoder_out_ch), std=0.01)

        self.actor_position_net = self.make_position_head(
            use_cnn_position_decoder,
            trunk_out_ch + (entity_encoder_out_ch if self.append_deck_info_to_position_head_input else 0), 
            position_space_width, 
            position_space_height, 
        )

    
    def apply_attention(self, embeddings, raw_entities, attention_layer, attention_norm):
        key_padding_mask = (raw_entities.abs().sum(dim=-1) == 0)
        
        all_masked = key_padding_mask.all(dim=-1, keepdim=True)  # To avoid NaN when all entities are masked
        key_padding_mask = key_padding_mask & ~all_masked

        attn_out, _ = attention_layer(
            embeddings, embeddings, embeddings,
            key_padding_mask=key_padding_mask,
        )
        return attention_norm(embeddings + attn_out)


    def get_trunk_input(self, obs, all_embeddings):
        my_card_embeddings       = all_embeddings[:, 0 : self.max_num_cards]
        opponent_card_embeddings = all_embeddings[:, self.max_num_cards : 2 * self.max_num_cards]

        my_crown_tower_embeddings       = all_embeddings[:, 2 * self.max_num_cards : 2 * self.max_num_cards + 3]
        opponent_crown_tower_embeddings = all_embeddings[:, 2 * self.max_num_cards + 3 : 2 * self.max_num_cards + 6]

        hand_start = 2 * self.max_num_cards + 6
        hand_embeddings = all_embeddings[:, hand_start : hand_start + self.num_cards_in_hand]
        next_card_embeddings = all_embeddings[:, hand_start + self.num_cards_in_hand]  # (B, entity_encoder_out_ch)

        def masked_mean(embeddings, entities):
            mask = (entities.abs().sum(dim=-1, keepdim=True) > 0).to(dtype=embeddings.dtype)
            count = mask.sum(dim=1).clamp(min=1.0)
            return (embeddings * mask).sum(dim=1) / count

        trunk_input = t.cat([
            obs["game_completion_fraction"],
            obs["elixirs"],
            my_crown_tower_embeddings.flatten(start_dim=1),                 # (B, 3 * entity_encoder_out_ch)
            opponent_crown_tower_embeddings.flatten(start_dim=1),           # (B, 3 * entity_encoder_out_ch)
            masked_mean(my_card_embeddings, obs["my_cards"]),               # (B, entity_encoder_out_ch)
            masked_mean(opponent_card_embeddings, obs["opponent_cards"]),   # (B, entity_encoder_out_ch)
            hand_embeddings.flatten(start_dim=1),                           # (B, 4 * entity_encoder_out_ch)
            next_card_embeddings,                                           # (B, entity_encoder_out_ch)
        ], dim=-1).to(dtype=t.float32)                                      # (B, trunk_extra_in_ch + 13 * entity_encoder_out_ch) 

        return trunk_input, hand_embeddings


    def forward(self, obs):
        """
        obs, which is just one player's, is expected to be a dict with:
        - game_completion_fraction: (B, 1)
        - elixirs: (B, 1)
        - my_cards: (B, N, card_dim)
            - where N is the upper cap on number of entities at once on the arena
            - zero padding is used
        - opponent_cards: (B, N, card_dim)
        - my_crown_towers: (B, 3, card_dim)
        - opponent_crown_towers: (B, 3, card_dim)
        """
        
        all_entities = t.cat([
            obs["my_cards"], 
            obs["opponent_cards"], 
            obs["my_crown_towers"], 
            obs["opponent_crown_towers"], 
            obs["my_hand"],
            obs["my_next_card"].unsqueeze(1),
        ], dim=1).to(dtype=t.float32)

        if not self.disjoint_actor_critic:
            all_embeddings = self.shared_entity_encoder(all_entities)

            if self.use_attention_over_entities:
                all_embeddings = self.apply_attention(all_embeddings, all_entities, self.shared_entity_attention, self.shared_attention_norm)

            shared_trunk_input, shared_hand_embeddings = self.get_trunk_input(obs, all_embeddings)
            shared_trunk_out   = self.shared_trunk(shared_trunk_input)

            critic_head_input  = shared_trunk_out
            actor_heads_inputs = shared_trunk_out
            actor_hand_embeddings = shared_hand_embeddings
        else:
            critic_embeddings = self.critic_entity_encoder(all_entities)
            actor_embeddings  = self.actor_entity_encoder(all_entities)

            if self.use_attention_over_entities:
                critic_embeddings = self.apply_attention(critic_embeddings, all_entities, self.critic_entity_attention, self.critic_attention_norm)
                actor_embeddings  = self.apply_attention(actor_embeddings, all_entities, self.actor_entity_attention, self.actor_attention_norm)

            critic_trunk_input, _ = self.get_trunk_input(obs, critic_embeddings)
            actor_trunk_input, actor_hand_embeddings  = self.get_trunk_input(obs, actor_embeddings)

            critic_head_input = self.critic_trunk(critic_trunk_input)
            actor_heads_inputs  = self.actor_trunk(actor_trunk_input)

        value = self.critic_head(critic_head_input).squeeze(-1)  # (B,)

        if not self.use_pointer_decoder:
            skip_logits = self.actor_skip_net(actor_heads_inputs).squeeze(-1)  # (B,)
            deck_logits = self.actor_deck_idx_net(actor_heads_inputs)
        else:
            hand_embeddings = actor_hand_embeddings

            B = actor_heads_inputs.shape[0]
            skip_embed = self.skip_token.unsqueeze(0).expand(B, -1).unsqueeze(1)
            candidates = t.cat([skip_embed, hand_embeddings], dim=1)  # (B, 5, E)

            query = self.pointer_query(actor_heads_inputs).unsqueeze(1)  # (B, 1, E)
            card_logits = (query * candidates).sum(dim=-1).squeeze(1)    # (B, 5)
            
            skip_logits = card_logits[:, 0]
            deck_logits = card_logits[:, 1:]

        if self.append_deck_info_to_position_head_input:
            return value, skip_logits, deck_logits, None, (actor_heads_inputs, actor_hand_embeddings) 

        pos_logits = self.actor_position_net(actor_heads_inputs)

        return value, skip_logits, deck_logits, pos_logits, None


if __name__ == "__main__":
    from itertools import product

    # quick action critic net sanity check with dummy data
    for disjoint_actor_critic, activation_fn, use_cnn_position_decoder, use_layer_init, append_deck_info_to_position_head_input, use_attention_over_entities, use_pointer_decoder in \
        list(product([False, True], ['relu', 'tanh', 'elu'], [False, True], [False, True], [False, True], [False, True], [False, True])):

        print(f"Config: disjoint_actor_critic={disjoint_actor_critic}, activation_fn={activation_fn}, use_cnn_position_decoder={use_cnn_position_decoder}, use_layer_init={use_layer_init}, append_deck_info_to_position_head_input={append_deck_info_to_position_head_input}, use_attention_over_entities={use_attention_over_entities}, use_pointer_decoder={use_pointer_decoder}")

        net = DeepSetsActorCritic(
            entity_encoder_in_ch=26,
            entity_encoder_mid_ch=64,
            entity_encoder_out_ch=32,

            trunk_extra_in_ch=2,
            trunk_out_ch=128,

            activation_fn=activation_fn,
            disjoint_actor_critic=disjoint_actor_critic,
            use_cnn_position_decoder=use_cnn_position_decoder,
            use_layer_init=use_layer_init,
            append_deck_info_to_position_head_input=append_deck_info_to_position_head_input,
            use_attention_over_entities=use_attention_over_entities,
            use_pointer_decoder=use_pointer_decoder,

            num_cards_in_hand=4,
            max_num_cards=10,
            position_space_width=18,
            position_space_height=32,
            deploy_cost_idx=2,
        )

        dummy_obs = {
            "game_completion_fraction": t.tensor([[0.5]]),
            "elixirs": t.tensor([[0.5]]),
            "my_cards": t.zeros((1, 10, 26)),
            "opponent_cards": t.zeros((1, 10, 26)),
            "my_crown_towers": t.zeros((1, 3, 26)),
            "opponent_crown_towers": t.zeros((1, 3, 26)),
            "my_hand": t.zeros((1, 4, 26)),
            "my_next_card": t.zeros((1, 26)),
        }

        action, log_prob, entropy, value = net.get_action_and_value(dummy_obs)
        print(f"Action: {action}, Log Prob: {log_prob}, Entropy: {entropy}, Value: {value}")