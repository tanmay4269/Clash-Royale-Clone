import numpy as np
import torch as t
import torch.nn as nn

from utils import EntityRegistry
from entities.troops import Knight, Giant, MiniPEKKA


class ActorCritic(nn.Module):
    def __init__(
        self, 
        
        entity_encoder_in_ch, 
        entity_encoder_mid_ch, 
        entity_encoder_out_ch,

        trunk_extra_in_ch,
        trunk_mid_ch,

        num_cards_in_deck,
        max_num_cards,
        position_space_width,
        position_space_height,

        invalid_position_mask=None,
        max_elixirs=10,
    ):
        super().__init__()

        self.max_num_cards = max_num_cards
        self.position_space_width  = position_space_width
        self.position_space_height = position_space_height
        
        self.invalid_position_mask = invalid_position_mask
        self.max_elixirs = max_elixirs

        # Deck order must match cr_gym_env.step(): idx 0=Knight, 1=Giant, 2=MiniPEKKA
        _deck_classes = [Knight, Giant, MiniPEKKA]
        _costs = [EntityRegistry._dummy_instances[cls.__name__].deploy_cost for cls in _deck_classes]
        self.register_buffer("deck_deploy_costs", t.tensor(_costs, dtype=t.float32))


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
            nn.Linear(trunk_extra_in_ch + (3 + 3 + 1 + 1) * entity_encoder_out_ch, trunk_mid_ch),
            nn.LayerNorm(trunk_mid_ch),
            nn.ReLU(),

            nn.Linear(trunk_mid_ch, trunk_mid_ch),
            nn.LayerNorm(trunk_mid_ch),
            nn.ReLU(),
        )

        self.critic = nn.Sequential(
            nn.Linear(trunk_mid_ch, 1)
        )

        self.actor_skip_net = nn.Sequential(
            nn.Linear(trunk_mid_ch, 1)
        )

        self.actor_deck_idx_net = nn.Sequential(
            nn.Linear(trunk_mid_ch, num_cards_in_deck)
        )

        self.actor_position_net = nn.Sequential(
            nn.Linear(trunk_mid_ch, position_space_width * position_space_height)
        )


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

        trunk_input = t.cat([
            obs["game_completion_fraction"],
            obs["elixirs"],
            my_crown_tower_embeddings.flatten(start_dim=1),                 # (B, 3 * entity_encoder_out_ch)
            opponent_crown_tower_embeddings.flatten(start_dim=1),           # (B, 3 * entity_encoder_out_ch)
            masked_mean(my_card_embeddings, obs["my_cards"]),               # (B, entity_encoder_out_ch)
            masked_mean(opponent_card_embeddings, obs["opponent_cards"]),   # (B, entity_encoder_out_ch)
        ], dim=-1).to(dtype=t.float32)  # (B, trunk_extra_in_ch + entity_encoder_out_ch)

        trunk_out = self.trunk(trunk_input)

        value = self.critic(trunk_out).squeeze(-1)  # (B,)

        skip_logits = self.actor_skip_net(trunk_out).squeeze(-1)  # (B,)
        deck_logits = self.actor_deck_idx_net(trunk_out)
        pos_logits = self.actor_position_net(trunk_out)
        
        return value, skip_logits, deck_logits, pos_logits


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
        value, skip_logits, deck_logits, pos_logits = self(obs)

        # --- Static masks ---
        if invalid_deck_mask is not None:
            deck_logits = deck_logits.masked_fill(invalid_deck_mask, float('-inf'))
        if invalid_position_mask is not None:
            pos_logits = pos_logits.masked_fill(invalid_position_mask, float('-inf'))
        if self.invalid_position_mask is not None:
            pos_logits = pos_logits.masked_fill(self.invalid_position_mask, float('-inf'))

        # --- Elixir masks ---
        raw_elixirs = (obs["elixirs"] + 1.0) / 2.0 * self.max_elixirs  # (B, 1)
        elixir_mask = self.deck_deploy_costs.unsqueeze(0) > raw_elixirs  # (B, num_cards)
        deck_logits = deck_logits.masked_fill(elixir_mask, float('-inf'))

        all_masked = elixir_mask.all(dim=-1)  # (B,): can't afford anything
        if all_masked.any():
            # Use large finite value, NOT inf: Bernoulli(logits=inf).log_prob() → NaN.
            skip_logits = skip_logits.masked_fill(all_masked, 20.0)
            # Zero out fully-masked rows so Categorical doesn't receive all-inf logits.
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


class ActorCriticV2(nn.Module):
    def __init__(
        self, 

        entity_encoder_in_ch, 
        entity_encoder_mid_ch, 
        entity_encoder_out_ch,

        trunk_extra_in_ch,
        trunk_mid_ch,

        activation_fn='relu',
        disjoint_actor_critic=False,
        use_cnn_position_decoder=False,
        use_last_layer_norms=False,
        append_deck_info_to_position_head_input=False,

        num_cards_in_deck=3,
        max_num_cards=3,
        position_space_width=18,
        position_space_height=32,

        invalid_position_mask=None,
        max_elixirs=10,
    ):
        super().__init__()

        self.disjoint_actor_critic = disjoint_actor_critic
        self.append_deck_info_to_position_head_input = append_deck_info_to_position_head_input

        self.max_num_cards = max_num_cards
        self.position_space_width  = position_space_width
        self.position_space_height = position_space_height
        
        self.invalid_position_mask = invalid_position_mask
        self.max_elixirs = max_elixirs

        # Activation function selection
        if activation_fn == 'relu':
            activation_layer = nn.ReLU
        elif activation_fn == 'tanh':
            activation_layer = nn.Tanh
        elif activation_fn == 'elu':
            activation_layer = nn.ELU
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")
        

        gain = np.sqrt(2)
        # gain = nn.init.calculate_gain(activation_fn)  # Doesn't support ELU
        def layer_init(layer, std=gain, bias_const=0.0):
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)
            return layer

        def make_entity_encoder():
            return nn.Sequential(
                layer_init(nn.Linear(entity_encoder_in_ch, entity_encoder_mid_ch)),
                nn.LayerNorm(entity_encoder_mid_ch),
                activation_layer(),

                layer_init(nn.Linear(entity_encoder_mid_ch, entity_encoder_mid_ch)),
                nn.LayerNorm(entity_encoder_mid_ch),
                activation_layer(),

                layer_init(nn.Linear(entity_encoder_mid_ch, entity_encoder_out_ch)),
                nn.LayerNorm(entity_encoder_out_ch),
                activation_layer(),
            )

        def make_trunk():
            return nn.Sequential(
                layer_init(nn.Linear(trunk_extra_in_ch + (3 + 3 + 1 + 1) * entity_encoder_out_ch, trunk_mid_ch)),
                nn.LayerNorm(trunk_mid_ch),
                activation_layer(),

                layer_init(nn.Linear(trunk_mid_ch, trunk_mid_ch)),
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
            self.actor_trunk = make_trunk()

        # Specific Heads
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(trunk_mid_ch, 1), std=1.0)
        )

        self.actor_skip_net = nn.Sequential(
            layer_init(nn.Linear(trunk_mid_ch, 1), std=0.01)
        )

        self.actor_deck_idx_net = nn.Sequential(
            layer_init(nn.Linear(trunk_mid_ch, num_cards_in_deck), std=0.01)
        )

        self.actor_position_net = self.make_position_head(
            use_cnn_position_decoder,
            trunk_mid_ch + (num_cards_in_deck if self.append_deck_info_to_position_head_input else 0), 
            position_space_width, 
            position_space_height, 
            layer_init,
            activation_layer
        )

        # Deck order must match cr_gym_env.step(): idx 0=Knight, 1=Giant, 2=MiniPEKKA
        _deck_classes = [Knight, Giant, MiniPEKKA]
        _costs = [EntityRegistry._dummy_instances[cls.__name__].deploy_cost for cls in _deck_classes]
        self.register_buffer("deck_deploy_costs", t.tensor(_costs, dtype=t.float32))


    def make_position_head(self, use_cnn_position_decoder, input_ch, position_space_width, position_space_height, layer_init, activation_layer):
        if not use_cnn_position_decoder:
            # return nn.Sequential(
            #     layer_init(nn.Linear(input_ch, 384)),
            #     activation_layer(),

            #     layer_init(nn.Linear(384, 384)),
            #     activation_layer(),

            #     layer_init(nn.Linear(384, 256)),
            #     activation_layer(),

            #     layer_init(nn.Linear(256, 32 * 18), std=0.01),
            # )

            return nn.Sequential(
                layer_init(nn.Linear(input_ch, position_space_width * position_space_height), std=0.01)
            )
        
        assert position_space_width == 18 and position_space_height == 32, "CNN decoder is hardcoded to 32x18 position space"

        # * Note: this has way larger capacity than the linear layer version, so ablate carefuly
        # ! If position logits misbehave in early training, revisit the orthogonal initialisation here coz theoretically its wrong!
        return nn.Sequential(
            # (input_ch) -> (input_ch,1,1)
            nn.Unflatten(1, (input_ch, 1, 1)),

            # (input_ch,1,1) -> (64,4,3)
            layer_init(nn.ConvTranspose2d(
                in_channels=input_ch,
                out_channels=64,
                kernel_size=(4, 3),
                stride=1,
                padding=0
            )),
            activation_layer(),

            # (64,4,3) -> (32,8,6)
            layer_init(nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            )),
            activation_layer(),

            # (32,8,6) -> (16,16,12)
            layer_init(nn.ConvTranspose2d(
                32, 16,
                kernel_size=4,
                stride=2,
                padding=1
            )),
            activation_layer(),

            # (16,16,12) -> (1,32,18)
            layer_init(nn.ConvTranspose2d(
                16, 1,
                kernel_size=(4, 8),
                stride=(2, 1),
                padding=(1, 0)
            ), std=0.01),

            # (1,32,18) -> (32*18)
            nn.Flatten(),
        )

    def get_trunk_input(self, obs, all_embeddings):
        my_card_embeddings       = all_embeddings[:, 0 : self.max_num_cards]
        opponent_card_embeddings = all_embeddings[:, self.max_num_cards : 2 * self.max_num_cards]

        my_crown_tower_embeddings       = all_embeddings[:, 2 * self.max_num_cards : 2 * self.max_num_cards + 3]
        opponent_crown_tower_embeddings = all_embeddings[:, 2 * self.max_num_cards + 3 :]

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
        ], dim=-1).to(dtype=t.float32)                                      # (B, trunk_extra_in_ch + entity_encoder_out_ch) 

        return trunk_input


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
        ], dim=1).to(dtype=t.float32)

        if not self.disjoint_actor_critic:
            all_embeddings = self.shared_entity_encoder(all_entities)

            shared_trunk_input = self.get_trunk_input(obs, all_embeddings)
            shared_trunk_out   = self.shared_trunk(shared_trunk_input)

            critic_head_input  = shared_trunk_out
            actor_heads_inputs = shared_trunk_out
        else:
            critic_embeddings = self.critic_entity_encoder(all_entities)
            actor_embeddings  = self.actor_entity_encoder(all_entities)

            critic_trunk_input = self.get_trunk_input(obs, critic_embeddings)
            actor_trunk_input  = self.get_trunk_input(obs, actor_embeddings)

            critic_head_input = self.critic_trunk(critic_trunk_input)
            actor_heads_inputs  = self.actor_trunk(actor_trunk_input)

        value = self.critic_head(critic_head_input).squeeze(-1)  # (B,)

        skip_logits = self.actor_skip_net(actor_heads_inputs).squeeze(-1)  # (B,)
        deck_logits = self.actor_deck_idx_net(actor_heads_inputs)
        
        if self.append_deck_info_to_position_head_input:
            actor_heads_inputs = t.cat([actor_heads_inputs, deck_logits], dim=-1)
        pos_logits = self.actor_position_net(actor_heads_inputs)
        
        return value, skip_logits, deck_logits, pos_logits


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
        value, skip_logits, deck_logits, pos_logits = self(obs)

        # --- Static masks ---
        if invalid_deck_mask is not None:
            deck_logits = deck_logits.masked_fill(invalid_deck_mask, float('-inf'))
        if invalid_position_mask is not None:
            pos_logits = pos_logits.masked_fill(invalid_position_mask, float('-inf'))
        if self.invalid_position_mask is not None:
            pos_logits = pos_logits.masked_fill(self.invalid_position_mask, float('-inf'))

        # --- Elixir masks ---
        raw_elixirs = (obs["elixirs"] + 1.0) / 2.0 * self.max_elixirs  # (B, 1)
        elixir_mask = self.deck_deploy_costs.unsqueeze(0) > raw_elixirs  # (B, num_cards)
        deck_logits = deck_logits.masked_fill(elixir_mask, float('-inf'))

        all_masked = elixir_mask.all(dim=-1)  # (B,): can't afford anything
        if all_masked.any():
            # Use large finite value, NOT inf: Bernoulli(logits=inf).log_prob() → NaN.
            skip_logits = skip_logits.masked_fill(all_masked, 20.0)
            # Zero out fully-masked rows so Categorical doesn't receive all-inf logits.
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


class BotNet:
    def __init__(self, bot_type, invalid_position_mask, num_cards_in_deck, position_space_width, position_space_height):
        self.bot_type = bot_type  # 'random', 'skip', or 'scripted'
        self.invalid_position_mask = invalid_position_mask
        self.num_cards_in_deck = num_cards_in_deck
        self.position_space_width = position_space_width
        self.position_space_height = position_space_height
        self.position_space_size = position_space_width * position_space_height
        self.toggle = False
        
    def _get_logits(self, B):
        skip_logits = t.zeros((B, 1))
        deck_logits = t.zeros((B, self.num_cards_in_deck))
        pos_logits = t.zeros((B, self.position_space_size))
        
        if self.bot_type == 'skip':
            skip_logits = t.full((B, 1), 100.0)  # Always skip
        elif self.bot_type == 'scripted':
            skip_logits = t.full((B, 1), 100.0)  # Default to skip
            deck_logits[:, 0] = 100.0            # Always pick card 0
            
            mid_y = self.position_space_height // 4  # middle of player's valid half
            mid_x = self.position_space_width // 2
            
            x_offset = 5 if self.toggle else -5
            
            target_x = max(0, min(self.position_space_width - 1, mid_x + x_offset))
            target_idx = mid_y * self.position_space_width + target_x
            
            if self.invalid_position_mask is not None and not self.invalid_position_mask[target_idx]:
                pos_logits[:, target_idx] = 100.0
            elif self.invalid_position_mask is not None:
                valid_indices = (~self.invalid_position_mask).nonzero(as_tuple=True)[0]
                if len(valid_indices) > 0:
                    center_valid_idx = valid_indices[len(valid_indices) // 2]
                    pos_logits[:, center_valid_idx] = 100.0
            
        if self.invalid_position_mask is not None:
            pos_logits = pos_logits.masked_fill(self.invalid_position_mask, float('-inf'))
            
        return skip_logits, deck_logits, pos_logits

    def _update_toggle_and_skip(self, x, skip_logits):
        if self.bot_type != 'scripted':
            return skip_logits
             
        # Card 0 is assumed to cost 5 (Giant), 4 (Mini Pekka), 3 (Knight), or max 10. 
        # Safest way without manual coupling is to only unskip when elixirs is >= 5.0
        B = x["elixirs"].shape[0] if isinstance(x, dict) else 1
        for b in range(B):
            elixir = x["elixirs"][b, 0].item() if isinstance(x, dict) else 0.0
            if elixir >= 0.5:  # Threshold for our scripted card
                skip_logits[b, 0] = -100.0  # Un-skip, command a spawn
                # Since we are successfully commanding a spawn, flip toggle for NEXT time.
                self.toggle = not self.toggle
                
        return skip_logits

    def get_action_and_value(self, x, action=None):
        B = x["my_cards"].shape[0] if isinstance(x, dict) else 1
        skip_logits, deck_logits, pos_logits = self._get_logits(B)
        skip_logits = self._update_toggle_and_skip(x, skip_logits)
            
        action = {
            "skip": t.distributions.Bernoulli(logits=skip_logits).sample().detach(),
            "deck_idx": t.distributions.Categorical(logits=deck_logits).sample().detach(),
            "position": t.distributions.Categorical(logits=pos_logits).sample().detach(),
        }
        return action, None, None, None
        
    def __call__(self, x):
        B = x["my_cards"].shape[0] if isinstance(x, dict) else 1
        value = t.zeros((B,))
        skip_logits, deck_logits, pos_logits = self._get_logits(B)
        skip_logits = self._update_toggle_and_skip(x, skip_logits)
        # Skip logits are expected as shape (B,)
        return value, skip_logits.squeeze(-1), deck_logits, pos_logits


if __name__ == "__main__":
    from itertools import product

    # quick action critic net sanity check with dummy data
    for disjoint_actor_critic, activation_fn, use_cnn_position_decoder, use_last_layer_norms, append_deck_info_to_position_head_input in \
        list(product([False, True], ['relu', 'tanh', 'elu'], [False, True], [False, True], [False, True])):

        net = ActorCritic(
            entity_encoder_in_ch=26,
            entity_encoder_mid_ch=64,
            entity_encoder_out_ch=32,

            trunk_extra_in_ch=2,
            trunk_mid_ch=128,

            activation_fn=activation_fn,
            disjoint_actor_critic=disjoint_actor_critic,
            use_cnn_position_decoder=use_cnn_position_decoder,
            use_last_layer_norms=use_last_layer_norms,
            append_deck_info_to_position_head_input=append_deck_info_to_position_head_input,

            num_cards_in_deck=3,
            max_num_cards=10,
            position_space_width=18,
            position_space_height=32,
        )

        dummy_obs = {
            "game_completion_fraction": t.tensor([[0.5]]),
            "elixirs": t.tensor([[0.5]]),
            "my_cards": t.zeros((1, 10, 26)),
            "opponent_cards": t.zeros((1, 10, 26)),
            "my_crown_towers": t.zeros((1, 3, 26)),
            "opponent_crown_towers": t.zeros((1, 3, 26)),
        }

        action, log_prob, entropy, value = net.get_action_and_value(dummy_obs)
        print(f"Action: {action}, Log Prob: {log_prob}, Entropy: {entropy}, Value: {value}")
