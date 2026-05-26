import torch as t
import torch.nn as nn

from rl.networks.base import BaseActorCritic


class TransformerActorCritic(BaseActorCritic):
    """Level 3: Full bidirectional transformer encoder (BERT-style) with segment embeddings,
    [CLS] token for global pooling, and pointer scoring for deck selection."""

    def __init__(
        self,
        entity_encoder_in_ch,
        trunk_mid_ch=128,
        d_model=64,
        num_heads=4,
        num_layers=2,
        num_segments=5,
        use_cnn_position_decoder=False,
        activation_fn='relu',
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.trunk_mid_ch = trunk_mid_ch

        activation_layer = self.get_activation(activation_fn)

        # Input projections
        self.meta_proj = self.layer_init(nn.Linear(2, d_model))
        self.entity_proj = self.layer_init(nn.Linear(entity_encoder_in_ch, d_model))

        # Segment and CLS embeddings
        self.segment_embedding = nn.Embedding(num_segments, d_model)
        self.cls_token = nn.Parameter(t.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CLS → trunk-dim projection
        self.post_transformer = nn.Sequential(
            self.layer_init(nn.Linear(d_model, trunk_mid_ch)),
            nn.LayerNorm(trunk_mid_ch),
            activation_layer(),
        )

        # Heads
        self.critic_head = nn.Sequential(self.layer_init(nn.Linear(trunk_mid_ch, 1), std=1.0))
        self.actor_skip_net = nn.Sequential(self.layer_init(nn.Linear(trunk_mid_ch, 1), std=0.01))
        self.deck_score = nn.Sequential(self.layer_init(nn.Linear(d_model, 1), std=0.01))

        self.actor_position_net = self.make_position_head(
            use_cnn_position_decoder,
            trunk_mid_ch,
            self.position_space_width, self.position_space_height,
            activation_layer,
        )

    def _encode(self, obs):
        B = obs["my_cards"].shape[0]
        device = obs["my_cards"].device
        N = self.max_num_cards

        # Token construction
        meta = self.meta_proj(t.cat([
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
        entity_tokens = self.entity_proj(all_entity_features)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = t.cat([cls, meta, entity_tokens], dim=1)

        # Segment IDs: 0=CLS, 1=meta, 2=towers, 3=deployed, 4=hand+next
        seg_ids = t.cat([
            t.zeros(B, 1, dtype=t.long, device=device),
            t.full((B, 1), 1, dtype=t.long, device=device),
            t.full((B, 6), 2, dtype=t.long, device=device),
            t.full((B, 2 * N), 3, dtype=t.long, device=device),
            t.full((B, self.num_cards_in_deck + 1), 4, dtype=t.long, device=device),
        ], dim=1)
        tokens = tokens + self.segment_embedding(seg_ids)

        # Padding mask for zero-padded deployed card slots
        card_padding = t.cat([
            obs["my_cards"], obs["opponent_cards"]
        ], dim=1).abs().sum(dim=-1) == 0
        padding_mask = t.cat([
            t.zeros(B, 1 + 1 + 6, dtype=t.bool, device=device),
            card_padding,
            t.zeros(B, self.num_cards_in_deck + 1, dtype=t.bool, device=device),
        ], dim=1)

        # Transformer
        encoded = self.transformer_encoder(tokens, src_key_padding_mask=padding_mask)

        trunk_out = self.post_transformer(encoded[:, 0])

        hand_start = 1 + 1 + 6 + 2 * N
        hand_out = encoded[:, hand_start : hand_start + self.num_cards_in_deck]

        value = self.critic_head(trunk_out).squeeze(-1)
        skip_logits = self.actor_skip_net(trunk_out).squeeze(-1)
        deck_logits = self.deck_score(hand_out).squeeze(-1)

        return trunk_out, hand_out, value, skip_logits, deck_logits

    def forward(self, obs):
        trunk_out, hand_out, value, skip_logits, deck_logits = self._encode(obs)
        pos_logits = self.actor_position_net(trunk_out)
        return value, skip_logits, deck_logits, pos_logits
