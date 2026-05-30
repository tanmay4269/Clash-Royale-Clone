import torch as t
import torch.nn as nn

from rl.networks.base import BaseActorCritic


class TransformerEncoderPart(nn.Module):
    def __init__(
        self, 
        layer_init, 
        entity_encoder_in_ch, 
        d_model, 
        num_segments, 
        num_heads, 
        num_layers
    ):
        super().__init__()

        # Input projections
        self.meta_proj = layer_init(nn.Linear(2, d_model))
        self.entity_proj = layer_init(nn.Linear(entity_encoder_in_ch, d_model))

        # Segment and CLS embeddings
        self.segment_embedding = nn.Embedding(num_segments, d_model)
        self.cls_token = nn.Parameter(t.randn(1, 1, d_model) * 0.01)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )


class TransformerActorCritic(BaseActorCritic):
    """Full bidirectional transformer encoder (BERT-style) with segment embeddings,
    [CLS] token for global pooling, and pointer scoring for deck selection."""

    def __init__(
        self,

        entity_encoder_in_ch,
        
        trunk_out_ch=128,

        d_model=64,
        num_heads=4,
        num_layers=2,
        num_segments=5,

        activation_fn='relu',
        
        # disjoint_actor_critic=False,
        disjoint_actor_critic=True,

        # use_layer_init=False,
        use_layer_init=True,
         
        # use_cnn_position_decoder=False,
        use_cnn_position_decoder=True,

        # append_deck_info_to_position_head_input=False,
        append_deck_info_to_position_head_input=True,

        # use_pointer_decoder=False,
        use_pointer_decoder=True,
        
        num_cards_in_hand=4,
        max_num_cards=32,
        position_space_width=18,
        position_space_height=32,

        invalid_position_mask=None,
        max_elixirs=10,
        deploy_cost_idx=2,
        
        **kwargs  # to ignore excess args ppo_trainer throws
    ):
        super().__init__(
            activation_fn=activation_fn,
            use_layer_init=use_layer_init,

            use_cnn_position_decoder=use_cnn_position_decoder,

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
        self.use_pointer_decoder = use_pointer_decoder

        self.d_model = d_model
        self.trunk_out_ch = trunk_out_ch

        def make_transformer_encoder():
            return TransformerEncoderPart(
                self.layer_init,
                entity_encoder_in_ch,
                d_model,
                num_segments,
                num_heads,
                num_layers
            )

        def make_post_transformer():
            return nn.Sequential(
                self.layer_init(nn.Linear(d_model, trunk_out_ch)),
                nn.LayerNorm(trunk_out_ch),
                self.activation_layer(),
            )

        if not self.disjoint_actor_critic:
            self.shared_transformer_encoder = make_transformer_encoder()
            self.shared_post_transformer = make_post_transformer()
        else:
            self.critic_transformer_encoder = make_transformer_encoder()
            self.actor_transformer_encoder  = make_transformer_encoder()

            self.critic_post_transformer = make_post_transformer()
            self.actor_post_transformer  = make_post_transformer()

        # Heads
        self.critic_head = nn.Sequential(
            self.layer_init(nn.Linear(trunk_out_ch, 1), std=1.0)
        )
        
        if not self.use_pointer_decoder:
            self.actor_skip_net = nn.Sequential(
                self.layer_init(nn.Linear(trunk_out_ch, 1))
            )
            
            self.actor_deck_idx_net = nn.Sequential(
                self.layer_init(nn.Linear(d_model, 1))
            )
        else:
            self.skip_token = nn.Parameter(t.zeros(d_model))
            self.pointer_query = self.layer_init(nn.Linear(trunk_out_ch, d_model))

        self.actor_position_net = self.make_position_head(
            use_cnn_position_decoder,
            trunk_out_ch + (num_cards_in_hand if self.append_deck_info_to_position_head_input else 0), 
            self.position_space_width, 
            self.position_space_height,
        )


    def encode_with(self, obs, encoder, post_transformer):
        B = obs["my_cards"].shape[0]
        device = obs["my_cards"].device
        N = self.max_num_cards

        # Token construction
        meta = encoder.meta_proj(t.cat([
            obs["game_completion_fraction"], obs["elixirs"]
        ], dim=-1)).unsqueeze(1)

        all_entity_features = t.cat([
            obs["my_crown_towers"],
            obs["opponent_crown_towers"],
            obs["my_cards"],
            obs["opponent_cards"],
            obs["my_hand"],
            obs["my_next_card"].unsqueeze(1),
        ], dim=1).to(dtype=t.float32)
        entity_tokens = encoder.entity_proj(all_entity_features)

        cls = encoder.cls_token.expand(B, -1, -1)
        tokens = t.cat([cls, meta, entity_tokens], dim=1)

        # Segment IDs: CLS, meta, towers, deployed, hand+next
        seg_ids = t.cat([
            t.full((B, 1),                          0, dtype=t.long, device=device),
            t.full((B, 1),                          1, dtype=t.long, device=device),
            t.full((B, 6),                          2, dtype=t.long, device=device),
            t.full((B, 2 * N),                      3, dtype=t.long, device=device),
            t.full((B, self.num_cards_in_hand + 1), 4, dtype=t.long, device=device),
        ], dim=1)
        tokens = tokens + encoder.segment_embedding(seg_ids)

        # Padding mask for zero-padded deployed card slots
        card_padding = t.cat([
            obs["my_cards"], obs["opponent_cards"]
        ], dim=1).abs().sum(dim=-1) == 0

        padding_mask = t.cat([
            t.zeros(B, 1 + 1 + 6, dtype=t.bool, device=device),
            card_padding,
            t.zeros(B, self.num_cards_in_hand + 1, dtype=t.bool, device=device),
        ], dim=1)

        # Transformer
        encoded = encoder.transformer_encoder(tokens, src_key_padding_mask=padding_mask)

        trunk_out = post_transformer(encoded[:, 0])

        hand_start = 1 + 1 + 6 + 2 * N
        hand_out = encoded[:, hand_start : hand_start + self.num_cards_in_hand]
        return trunk_out, hand_out


    def forward(self, obs):
        if not self.disjoint_actor_critic:
            critic_trunk_out, hand_out = self.encode_with(obs, self.shared_transformer_encoder, self.shared_post_transformer)
            actor_trunk_out = critic_trunk_out
            actor_hand_out = hand_out
        else:
            critic_trunk_out, _ = self.encode_with(obs, self.critic_transformer_encoder, self.critic_post_transformer)
            actor_trunk_out, actor_hand_out = self.encode_with(obs, self.actor_transformer_encoder, self.actor_post_transformer)

        value = self.critic_head(critic_trunk_out).squeeze(-1)
        
        if not self.use_pointer_decoder:
            skip_logits = self.actor_skip_net(actor_trunk_out).squeeze(-1)
            deck_logits = self.actor_deck_idx_net(actor_hand_out).squeeze(-1)
        else:
            B = actor_trunk_out.shape[0]
            skip_embed = self.skip_token.unsqueeze(0).expand(B, -1).unsqueeze(1)
            candidates = t.cat([skip_embed, actor_hand_out], dim=1)  # (B, 5, d_model)

            query = self.pointer_query(actor_trunk_out).unsqueeze(1)  # (B, 1, d_model)
            card_logits = (query * candidates).sum(dim=-1).squeeze(1) # (B, 5)
            
            skip_logits = card_logits[:, 0]
            deck_logits = card_logits[:, 1:]

        if self.append_deck_info_to_position_head_input:
            return value, skip_logits, deck_logits, None, actor_trunk_out

        pos_logits = self.actor_position_net(actor_trunk_out)
        return value, skip_logits, deck_logits, pos_logits, None


if __name__ == "__main__":
    from itertools import product

    # quick action critic net sanity check with dummy data
    for disjoint_actor_critic, activation_fn, use_cnn_position_decoder, use_layer_init, append_deck_info_to_position_head_input, use_pointer_decoder in \
        list(product([False, True], ['relu', 'tanh', 'elu'], [False, True], [False, True], [False, True], [False, True])):

        print(f"Config: disjoint_actor_critic={disjoint_actor_critic}, activation_fn={activation_fn}, use_cnn_position_decoder={use_cnn_position_decoder}, use_layer_init={use_layer_init}, append_deck_info_to_position_head_input={append_deck_info_to_position_head_input}, use_pointer_decoder={use_pointer_decoder}")

        net = TransformerActorCritic(
            entity_encoder_in_ch=26,
            
            trunk_out_ch=128,

            d_model=64,
            num_heads=4,
            num_layers=2,
            num_segments=5,

            activation_fn=activation_fn,
            disjoint_actor_critic=disjoint_actor_critic,
            use_layer_init=use_layer_init,
            use_cnn_position_decoder=use_cnn_position_decoder,
            append_deck_info_to_position_head_input=append_deck_info_to_position_head_input,
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
