import torch as t


class BotNet:
    def __init__(self, bot_type, invalid_position_mask, num_cards_in_deck, position_space_width, position_space_height):
        self.bot_type = bot_type
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
            skip_logits = t.full((B, 1), 100.0)
        elif self.bot_type == 'scripted':
            skip_logits = t.full((B, 1), 100.0)
            deck_logits[:, 0] = 100.0

            mid_y = self.position_space_height // 4
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

        B = x["elixirs"].shape[0] if isinstance(x, dict) else 1
        for b in range(B):
            elixir = x["elixirs"][b, 0].item() if isinstance(x, dict) else 0.0
            if elixir >= 0.5:
                skip_logits[b, 0] = -100.0
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
        return value, skip_logits.squeeze(-1), deck_logits, pos_logits
