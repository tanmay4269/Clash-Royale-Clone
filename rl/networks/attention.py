import torch as t
import torch.nn as nn

from rl.networks.deep_sets import DeepSetsActorCritic


class AttentionActorCritic(DeepSetsActorCritic):
    """Level 2: Adds a single bidirectional MultiHeadAttention layer over entity
    embeddings before the Deep Sets pool, enabling relational reasoning between entities."""

    def __init__(self, num_attention_heads=4, **kwargs):
        super().__init__(**kwargs)

        E = self.entity_encoder_out_ch
        self.entity_attention = nn.MultiheadAttention(
            embed_dim=E,
            num_heads=num_attention_heads,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(E)

    def _apply_attention(self, embeddings, raw_entities):
        key_padding_mask = (raw_entities.abs().sum(dim=-1) == 0)
        attn_out, _ = self.entity_attention(
            embeddings, embeddings, embeddings,
            key_padding_mask=key_padding_mask,
        )
        return self.attention_norm(embeddings + attn_out)

    def forward(self, obs):
        all_entities = t.cat([
            obs["my_cards"],
            obs["opponent_cards"],
            obs["my_crown_towers"],
            obs["opponent_crown_towers"],
        ], dim=1).to(dtype=t.float32)

        if not self.disjoint_actor_critic:
            all_embeddings = self.shared_entity_encoder(all_entities)
            all_embeddings = self._apply_attention(all_embeddings, all_entities)
            shared_trunk_input = self.get_trunk_input(obs, all_embeddings, self.shared_entity_encoder)
            shared_trunk_out   = self.shared_trunk(shared_trunk_input)
            critic_head_input  = shared_trunk_out
            actor_heads_inputs = shared_trunk_out
        else:
            critic_embeddings = self.critic_entity_encoder(all_entities)
            actor_embeddings  = self.actor_entity_encoder(all_entities)
            critic_embeddings = self._apply_attention(critic_embeddings, all_entities)
            actor_embeddings  = self._apply_attention(actor_embeddings, all_entities)
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
