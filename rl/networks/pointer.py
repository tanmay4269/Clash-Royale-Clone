import torch as t
import torch.nn as nn

from rl.networks.deep_sets import DeepSetsActorCritic


class PointerActorCritic(DeepSetsActorCritic):
    """Level 1: Replaces separate skip + deck heads with a unified 5-way pointer
    (4 hand cards + 1 learnable skip token). Eliminates the separate Bernoulli skip head."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        E = self.entity_encoder_out_ch
        D = self.trunk_mid_ch

        del self.actor_skip_net
        del self.actor_deck_idx_net

        self.skip_token = nn.Parameter(t.zeros(E))
        self.pointer_query = self.layer_init(nn.Linear(D, E), std=0.01)

    def forward(self, obs):
        all_entities = t.cat([
            obs["my_cards"],
            obs["opponent_cards"],
            obs["my_crown_towers"],
            obs["opponent_crown_towers"],
        ], dim=1).to(dtype=t.float32)

        if not self.disjoint_actor_critic:
            all_embeddings = self.shared_entity_encoder(all_entities)
            trunk_input = self.get_trunk_input(obs, all_embeddings, self.shared_entity_encoder)
            trunk_out = self.shared_trunk(trunk_input)
            hand_embeddings = self.shared_entity_encoder(obs["my_hand"])
        else:
            actor_embeddings = self.actor_entity_encoder(all_entities)
            critic_embeddings = self.critic_entity_encoder(all_entities)
            actor_trunk_input = self.get_trunk_input(obs, actor_embeddings, self.actor_entity_encoder)
            critic_trunk_input = self.get_trunk_input(obs, critic_embeddings, self.critic_entity_encoder)
            trunk_out = self.actor_trunk(actor_trunk_input)
            critic_out = self.critic_trunk(critic_trunk_input)
            hand_embeddings = self.actor_entity_encoder(obs["my_hand"])

        B = trunk_out.shape[0]

        # Pointer: query against hand cards + skip token
        skip_embed = self.skip_token.unsqueeze(0).expand(B, -1).unsqueeze(1)
        candidates = t.cat([hand_embeddings, skip_embed], dim=1)  # (B, 5, E)
        query = self.pointer_query(trunk_out).unsqueeze(1)        # (B, 1, E)
        card_logits = (query * candidates).sum(dim=-1).squeeze(1) # (B, 5)

        if not self.disjoint_actor_critic:
            value = self.critic_head(trunk_out).squeeze(-1)
        else:
            value = self.critic_head(critic_out).squeeze(-1)

        if self.append_deck_info_to_position_head_input:
            pos_input = t.cat([trunk_out, card_logits[:, :self.num_cards_in_deck]], dim=-1)
        else:
            pos_input = trunk_out
        pos_logits = self.actor_position_net(pos_input)

        return value, card_logits, pos_logits

    def get_action_and_value(self, obs, action=None, invalid_deck_mask=None, invalid_position_mask=None):
        value, card_logits, pos_logits = self(obs)

        B = card_logits.shape[0]

        if invalid_position_mask is not None:
            pos_logits = pos_logits.masked_fill(invalid_position_mask, float('-inf'))
        if self.invalid_position_mask is not None:
            pos_logits = pos_logits.masked_fill(self.invalid_position_mask, float('-inf'))

        # Elixir mask on first 4 slots (hand cards); slot 4 (skip) is always valid
        raw_elixirs = (obs["elixirs"] + 1.0) / 2.0 * self.max_elixirs
        hand_normalized_costs = obs["my_hand"][..., self.deploy_cost_idx]
        hand_raw_costs = hand_normalized_costs * (self.max_elixirs / 2.0) + (self.max_elixirs / 2.0)
        elixir_mask = hand_raw_costs > raw_elixirs  # (B, 4)
        card_logits[:, :self.num_cards_in_deck] = card_logits[:, :self.num_cards_in_deck].masked_fill(elixir_mask, float('-inf'))

        if invalid_deck_mask is not None:
            card_logits[:, :self.num_cards_in_deck] = card_logits[:, :self.num_cards_in_deck].masked_fill(invalid_deck_mask, float('-inf'))

        card_dist = t.distributions.Categorical(logits=card_logits)
        pos_dist  = t.distributions.Categorical(logits=pos_logits)

        if action is None:
            action_card = card_dist.sample()
            action_pos  = pos_dist.sample()
        else:
            # Reconstruct 5-way card action from skip + deck_idx
            action_card = t.where(action["skip"].bool(), t.tensor(self.num_cards_in_deck, device=card_logits.device), action["deck_idx"].long())
            action_pos = action["position"].long()

        is_skip = (action_card == self.num_cards_in_deck)

        card_log_prob = card_dist.log_prob(action_card)
        pos_log_prob  = pos_dist.log_prob(action_pos)
        log_prob = card_log_prob + t.where(is_skip, t.zeros_like(pos_log_prob), pos_log_prob)

        card_entropy = card_dist.entropy()
        pos_entropy  = pos_dist.entropy()
        entropy = card_entropy + t.where(is_skip, t.zeros_like(pos_entropy), pos_entropy)

        action_skip = is_skip.float()
        action_deck = t.where(is_skip, t.zeros_like(action_card), action_card)

        action = {
            "skip": action_skip.detach(),
            "deck_idx": action_deck.detach(),
            "position": action_pos.detach(),
        }

        return action, log_prob, entropy, value
