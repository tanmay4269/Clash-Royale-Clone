from __future__ import annotations

import os
import time
from dataclasses import dataclass

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import numpy as np
import pygame
import torch as t
import torch.nn.functional as F

import rl.env.cr_gym_env
from game.entities.spells.arrows import Arrows
from game.entities.spells.fireball import Fireball
from game.entities.troops.archer import Archers
from game.entities.troops.giant import Giant
from game.entities.troops.knight import Knight
from game.entities.troops.minion import Minions
from game.entities.troops.mini_pekka import MiniPEKKA
from game.entities.troops.musketeer import Musketeer
from rl.env.cr_flatten_norm_wrapper import CRFlattenNormWrapper
from rl.networks import make_network
from rl.networks.botnet import BotNet
from web_play.legacy_networks import LegacyAttentionDeepSetsActorCritic


CARD_BY_NAME = {
    "Knight": Knight,
    "Giant": Giant,
    "MiniPEKKA": MiniPEKKA,
    "Musketeer": Musketeer,
    "Archers": Archers,
    "Fireball": Fireball,
    "Arrows": Arrows,
    "Minions": Minions,
}


@dataclass(frozen=True)
class OpponentSpec:
    kind: str
    label: str
    checkpoint_path: str | None = None
    architecture: str = "deep_sets"
    architecture_label: str = "Built-in bot"


def split_observations(obs, env, arena, max_num_objects):
    flat_card_space = env.flat_card_space
    card_dim = flat_card_space.shape[0]
    position_x_idx, position_y_idx = np.arange(
        *env.flattened_card_space_indices["position"]
    )

    p1_cards = obs["player_1_cards"]
    p2_cards = obs["player_2_cards"]

    n1 = len(p1_cards)
    n2 = len(p2_cards)

    arr1 = np.array(p1_cards, dtype=np.float32) if n1 else np.zeros((0, card_dim), dtype=np.float32)
    arr2 = np.array(p2_cards, dtype=np.float32) if n2 else np.zeros((0, card_dim), dtype=np.float32)

    def pad(arr, n):
        return F.pad(
            t.tensor(arr),
            (0, 0, 0, max_num_objects - n),
            "constant",
            0,
        ).unsqueeze(0)

    def rot180(entities):
        rotated = entities.clone()
        rotated[..., position_x_idx] *= -1
        rotated[..., position_y_idx] *= -1
        return rotated

    p1t = pad(arr1, n1)
    p2t = pad(arr2, n2)

    ct1 = t.tensor(np.array(obs["player_1_crown_towers"], dtype=np.float32)).unsqueeze(0)
    ct2 = t.tensor(np.array(obs["player_2_crown_towers"], dtype=np.float32)).unsqueeze(0)
    gcf = t.tensor(np.array(obs["game_completion_fraction"], dtype=np.float32)).reshape(-1, 1)

    obs_1 = {
        "game_completion_fraction": gcf,
        "elixirs": t.tensor(np.array(obs["player_1_elixirs"], dtype=np.float32)).reshape(-1, 1),
        "my_cards": p1t,
        "opponent_cards": p2t,
        "my_crown_towers": ct1,
        "opponent_crown_towers": ct2,
        "my_hand": t.tensor(np.array(obs["player_1_hand"], dtype=np.float32)).unsqueeze(0),
        "my_next_card": t.tensor(np.array(obs["player_1_next_card"], dtype=np.float32)).unsqueeze(0),
    }

    obs_2 = {
        "game_completion_fraction": gcf,
        "elixirs": t.tensor(np.array(obs["player_2_elixirs"], dtype=np.float32)).reshape(-1, 1),
        "my_cards": rot180(p2t),
        "opponent_cards": rot180(p1t),
        "my_crown_towers": rot180(ct2),
        "opponent_crown_towers": rot180(ct1),
        "my_hand": t.tensor(np.array(obs["player_2_hand"], dtype=np.float32)).unsqueeze(0),
        "my_next_card": t.tensor(np.array(obs["player_2_next_card"], dtype=np.float32)).unsqueeze(0),
    }

    return obs_1, obs_2


def scalar_int(value):
    return int(value.detach().cpu().item()) if hasattr(value, "detach") else int(value)


def action_position(action, arena):
    idx = scalar_int(action["position"])
    return idx % arena.width, idx // arena.width


def apply_action_to_arena(arena, joined_action):
    for idx in [1, 2]:
        if joined_action[f"player_{idx}_skip"] == 1:
            continue

        owner = arena.player_side_1 if idx == 1 else arena.player_side_2
        x, y = joined_action[f"player_{idx}_card_position"]
        card_idx = joined_action[f"player_{idx}_card_idx"]

        if 0 <= card_idx < len(owner.hand):
            card_cls = owner.hand[card_idx]
            card = card_cls(owner, y, x)

            if arena.deploy_entity(card):
                owner.add_object(card)
                owner.use_card(card_idx)


class WebMatch:
    def __init__(self, opponent: OpponentSpec):
        pygame.init()
        pygame.font.init()
        pygame.display.set_mode((1, 1))

        self._env_raw = gym.make("ClashRoyaleEnv-v0")
        self._env_wrap = CRFlattenNormWrapper(self._env_raw)
        self._env_raw.reset()
        self.arena = self._env_raw.unwrapped.arena
        self.arena._debug_active_player = 2
        self.arena.player_side_2.active_card_idx = 0

        self.max_num_objects = self.arena.max_num_objects
        self.opponent = opponent
        self._bot = self._make_bot(opponent)
        self._last_tick = time.monotonic()
        self.finished = False
        self.terminated = False
        self.truncated = False
        self.value_1 = 0.0
        self.value_2 = 0.0

        self.scale_factor = 1.5
        self.eval_bar_width = 30
        self.arena_width = self.arena.width * self.arena.tile_size
        self.arena_height = self.arena.height * self.arena.tile_size
        self.panel_width = 200
        self.surface_width = int((self.eval_bar_width + self.arena_width + self.panel_width) * self.scale_factor)
        self.surface_height = int(self.arena_height * self.scale_factor)
        self.surface = pygame.Surface((self.surface_width, self.surface_height))

    def _invalid_position_mask(self):
        arena = self.arena
        scale = arena.tile_size
        occupancy_grid = arena.cell_occupancy
        tiled = np.where(occupancy_grid == 1, 1, 0)[scale // 2 :: scale, scale // 2 :: scale]
        mask = tiled.astype(bool).T
        mask[arena.height // 2 :, :] = True
        return t.tensor(mask).flatten()

    def _make_bot(self, opponent: OpponentSpec):
        mask_tensor = self._invalid_position_mask()
        if opponent.kind == "checkpoint":
            network_kwargs = {
                "entity_encoder_in_ch": self._env_wrap.flat_card_space.shape[0],
                "entity_encoder_mid_ch": 64,
                "entity_encoder_out_ch": 32,
                "trunk_extra_in_ch": 2,
                "trunk_mid_ch": 128,
                "trunk_out_ch": 128,
                "num_cards_in_deck": self._env_raw.unwrapped.NUM_CARDS_IN_HAND,
                "num_cards_in_hand": self._env_raw.unwrapped.NUM_CARDS_IN_HAND,
                "max_num_cards": self.arena.max_num_objects,
                "position_space_width": self.arena.width,
                "position_space_height": self.arena.height,
                "invalid_position_mask": mask_tensor,
                "max_elixirs": 10,
                "deploy_cost_idx": self._env_wrap.flattened_card_space_indices["deploy_cost"][0],
            }
            if opponent.architecture == "legacy_deep_sets_attention":
                legacy_kwargs = dict(network_kwargs)
                legacy_kwargs.pop("num_cards_in_deck", None)
                network = LegacyAttentionDeepSetsActorCritic(**legacy_kwargs)
            else:
                if opponent.architecture == "transformer":
                    network_kwargs["use_pointer_decoder"] = False
                network = make_network(opponent.architecture, **network_kwargs)
            state_dict = t.load(opponent.checkpoint_path, map_location="cpu", weights_only=True)
            network.load_state_dict(state_dict)
            network.eval()
            return network

        return BotNet(
            bot_type=opponent.kind,
            invalid_position_mask=mask_tensor,
            num_cards_in_deck=4,
            position_space_width=self.arena.width,
            position_space_height=self.arena.height,
        )

    def _get_obs_wrapped(self):
        raw_obs = self._env_raw.unwrapped._get_obs()
        return self._env_wrap.observation(raw_obs)

    def _bot_action(self):
        obs_wrapped = self._get_obs_wrapped()
        obs_1, obs_2 = split_observations(
            obs_wrapped,
            self._env_wrap,
            self.arena,
            self.max_num_objects,
        )
        with t.no_grad():
            action, _, _, value_1 = self._bot.get_action_and_value(obs_1)
            self.value_1 = self._tensor_scalar(value_1)
            if self.opponent.kind == "checkpoint":
                _, _, _, value_2 = self._bot.get_action_and_value(obs_2)
                self.value_2 = self._tensor_scalar(value_2)
            else:
                self.value_2 = 0.0
        return action

    @staticmethod
    def _tensor_scalar(value) -> float:
        if value is None:
            return 0.0
        if hasattr(value, "detach"):
            return float(value.detach().cpu().reshape(-1)[0].item())
        return float(value)

    def tick(self):
        if self.finished:
            return

        now = time.monotonic()
        dt = min(now - self._last_tick, 1 / 15)
        self._last_tick = now

        action_1 = self._bot_action()
        x, y = action_position(action_1, self.arena)
        joined = {
            "player_1_skip": scalar_int(action_1["skip"]),
            "player_1_card_idx": scalar_int(action_1["deck_idx"]),
            "player_1_card_position": (x, y),
            "player_2_skip": 1,
            "player_2_card_idx": 0,
            "player_2_card_position": (0, 0),
        }
        apply_action_to_arena(self.arena, joined)

        self.terminated, self.truncated = self.arena.update(dt)
        self.finished = self.terminated or self.truncated

    def select_card(self, index: int):
        if 0 <= index < len(self.arena.player_side_2.hand):
            self.arena._debug_active_player = 2
            self.arena.player_side_2.active_card_idx = index

    def deploy(self, canvas_x: int, canvas_y: int) -> bool:
        if self.finished:
            return False

        x = int(canvas_x / self.scale_factor) - self.eval_bar_width
        y = int(canvas_y / self.scale_factor)

        if self._select_hud_card(x, y):
            return True

        if not (0 <= x < self.arena_width and 0 <= y < self.arena_height):
            return False

        self.arena._debug_active_player = 2
        before_hand = list(self.arena.player_side_2.hand)
        before_elixir = self.arena.player_side_2.elixirs
        if self._is_invalid_human_deploy_position(x, y):
            return False
        self.arena.on_click((x, y))
        after_hand = list(self.arena.player_side_2.hand)
        after_elixir = self.arena.player_side_2.elixirs
        return before_hand != after_hand or before_elixir != after_elixir

    def _is_invalid_human_deploy_position(self, x: int, y: int) -> bool:
        tile_col = x // self.arena.tile_size
        tile_row = y // self.arena.tile_size
        if not (0 <= tile_col < self.arena.width and 0 <= tile_row < self.arena.height):
            return True

        scale = self.arena.tile_size
        occupancy_grid = self.arena.cell_occupancy
        tiled = np.where(occupancy_grid == 1, 1, 0)[scale // 2 :: scale, scale // 2 :: scale]
        mask = tiled.astype(bool).T
        mask[: self.arena.height // 2, :] = True
        return bool(mask[tile_row, tile_col])

    def _select_hud_card(self, x: int, y: int) -> bool:
        panel_x = self.arena_width
        if x < panel_x:
            return False

        title_x = panel_x + 7
        card_y = 368
        card_w = 30
        card_h = 42
        hit_pad_x = 8
        hit_pad_y = 10

        for idx in range(len(self.arena.player_side_2.hand)):
            card_x = title_x + idx * 36
            if (
                card_x - hit_pad_x <= x <= card_x + card_w + hit_pad_x
                and card_y - hit_pad_y <= y <= card_y + card_h + hit_pad_y
            ):
                self.select_card(idx)
                return True

        return False

    def frame_rgba(self) -> bytes:
        self.surface.fill((24, 24, 28))
        arena_hud_surface = pygame.Surface((
            int((self.arena_width + self.panel_width) * self.scale_factor),
            self.surface_height,
        ))
        arena_hud_surface.fill((24, 24, 28))
        self._draw_eval_bar(self.surface)
        self.arena.render(arena_hud_surface, scale_factor=self.scale_factor)
        self.surface.blit(arena_hud_surface, (int(self.eval_bar_width * self.scale_factor), 0))
        if self.finished:
            self._draw_result_overlay(self.surface)
        return pygame.image.tobytes(self.surface, "RGBA")

    def _draw_eval_bar(self, surface):
        height = self.surface_height
        width = int(self.eval_bar_width * self.scale_factor)
        win_prob = 1.0 / (1.0 + np.exp(-float(np.clip(self.value_2, -20, 20))))
        red_h = int(height * (1.0 - win_prob))

        pygame.draw.rect(surface, (200, 40, 40), (0, 0, width, red_h))
        pygame.draw.rect(surface, (40, 100, 200), (0, red_h, width, height - red_h))
        pygame.draw.line(surface, (255, 255, 255), (0, height // 2), (width, height // 2), max(1, int(2 * self.scale_factor)))

        font = pygame.font.SysFont(None, int(15 * self.scale_factor), bold=True)
        red_text = font.render(f"{(1.0 - win_prob) * 100:.0f}", True, (255, 255, 255))
        blue_text = font.render(f"{win_prob * 100:.0f}", True, (255, 255, 255))
        if red_h > red_text.get_height() + 6:
            surface.blit(red_text, ((width - red_text.get_width()) // 2, max(3, red_h // 2 - red_text.get_height() // 2)))
        if height - red_h > blue_text.get_height() + 6:
            surface.blit(blue_text, ((width - blue_text.get_width()) // 2, red_h + (height - red_h - blue_text.get_height()) // 2))

    def _draw_result_overlay(self, surface):
        overlay = pygame.Surface((self.surface_width, self.surface_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 145))
        surface.blit(overlay, (0, 0))

        if self.arena.winner == 2:
            title = "YOU WIN"
            color = (139, 206, 247)
        elif self.arena.winner == 1:
            title = "YOU LOSE"
            color = (239, 68, 68)
        else:
            title = "DRAW"
            color = (242, 244, 247)

        title_font = pygame.font.SysFont(None, int(54 * self.scale_factor), bold=True)
        sub_font = pygame.font.SysFont(None, int(18 * self.scale_factor))
        title_surface = title_font.render(title, True, color)
        sub_surface = sub_font.render("Start a new match from the top bar.", True, (230, 230, 230))
        cx = self.surface_width // 2
        cy = self.surface_height // 2
        surface.blit(title_surface, (cx - title_surface.get_width() // 2, cy - title_surface.get_height()))
        surface.blit(sub_surface, (cx - sub_surface.get_width() // 2, cy + 8))

    def state(self) -> dict:
        p1 = self.arena.player_side_1
        p2 = self.arena.player_side_2
        return {
            "opponent": {
                "label": self.opponent.label,
                "kind": self.opponent.kind,
                "architecture": self.opponent.architecture,
                "architectureLabel": self.opponent.architecture_label,
            },
            "canvas": {
                "width": self.surface_width,
                "height": self.surface_height,
                "evalBarWidth": int(self.eval_bar_width * self.scale_factor),
                "arenaWidth": int(self.arena_width * self.scale_factor),
                "arenaHeight": int(self.arena_height * self.scale_factor),
            },
            "time": round(self.arena.elapsed_time, 1),
            "finished": self.finished,
            "winner": self.arena.winner,
            "blue": {
                "elixir": round(p2.elixirs, 1),
                "hand": [card.__name__ for card in p2.hand],
                "next": p2.next_card.__name__ if p2.next_card else None,
                "activeCardIndex": p2.active_card_idx,
            },
            "red": {
                "elixir": round(p1.elixirs, 1),
                "hand": [card.__name__ for card in p1.hand],
                "next": p1.next_card.__name__ if p1.next_card else None,
            },
        }
