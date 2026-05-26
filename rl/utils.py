from typing import List

import numpy as np
import gymnasium as gym
import pygame

from game.entity import EntityType, EntityRegistry


class AverageMeter:
    def __init__(self, max_samples = None):
        self.max_samples = max_samples
        self.samples: List[float] = []

    def add_sample(self, value: float):
        self.samples.append(value)
        if self.max_samples is not None and len(self.samples) > self.max_samples:
            self.samples.pop(0)

    def average(self) -> float:
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)

    def window_average(self, window_size: int) -> float:
        if not self.samples:
            return 0.0
        return sum(self.samples[-window_size:]) / min(window_size, len(self.samples))


class HeatmapVisualizerWrapper(gym.Wrapper):
    def __init__(self, env, extra_width=150):
        super().__init__(env)
        self.debug_policy = None
        self.extra_width = extra_width
        self.arena_w = self.unwrapped.arena.width
        self.arena_h = self.unwrapped.arena.height
        self._frame_idx = 0

    def update(
        self,
        *,
        player_idx,
        skip_prob,
        deck_probs,
        pos_probs,
        pos_logits=None,
        action=None,
        env_action=None,
    ):
        self.debug_policy = {
            "player_idx": player_idx,
            "skip_prob": float(skip_prob),
            "deck_probs": np.asarray(deck_probs, dtype=np.float32),
            "pos_probs": np.asarray(pos_probs, dtype=np.float32),
            "pos_logits": None if pos_logits is None else np.asarray(pos_logits, dtype=np.float32),
            "action": action,
            "env_action": env_action,
        }

    def _policy_values_to_arena(self, values, player_idx):
        values_2d = np.asarray(values, dtype=np.float32).reshape(self.arena_h, self.arena_w)
        if player_idx == 2:
            return np.flip(values_2d, axis=(0, 1))
        return values_2d

    def _normalize_heatmap_values(self, values_2d):
        finite = np.isfinite(values_2d)
        normalized = np.zeros_like(values_2d, dtype=np.float32)
        if not np.any(finite):
            return normalized

        finite_values = values_2d[finite]
        lo = float(np.min(finite_values))
        hi = float(np.max(finite_values))
        span = hi - lo
        if span <= 1e-8:
            normalized[finite] = 0.15
            return normalized

        normalized[finite] = (values_2d[finite] - lo) / span
        return np.sqrt(np.clip(normalized, 0.0, 1.0))

    def _policy_index_to_arena_cell(self, pos_idx, player_idx):
        pos_idx = int(pos_idx)
        row = pos_idx // self.arena_w
        col = pos_idx % self.arena_w
        if player_idx == 2:
            row = self.arena_h - 1 - row
            col = self.arena_w - 1 - col
        return row, col

    def _action_position(self, debug_policy):
        player_idx = debug_policy["player_idx"]
        env_action = debug_policy.get("env_action")
        if env_action is not None:
            x, y = env_action[f"player_{player_idx}_card_position"]
            return int(y), int(x)

        action = debug_policy.get("action")
        if action is None:
            return None

        pos = action["position"]
        if hasattr(pos, "detach"):
            pos = pos.detach().cpu().item()
        return self._policy_index_to_arena_cell(pos, player_idx)

    def render(self):
        frame = self.env.render()
        if frame is None:
            return frame

        H, W, C = frame.shape
        new_W = W + self.extra_width

        surface = pygame.Surface((new_W, H))
        img = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surface.blit(img, (0, 0))

        pygame.draw.rect(surface, (30, 30, 30), (W, 0, self.extra_width, H))

        if self.debug_policy is None:
            return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
            
        debug_policy = self.debug_policy
        player_idx = debug_policy["player_idx"]
        skip_prob = debug_policy["skip_prob"]
        deck_probs = debug_policy["deck_probs"]
        pos_probs = debug_policy["pos_probs"]
        pos_logits = debug_policy["pos_logits"]

        heatmap_surface = pygame.Surface((W, H), pygame.SRCALPHA)
        heatmap_values_2d = self._policy_values_to_arena(pos_probs, player_idx)

        tile_h = H // self.arena_h
        tile_w = W // self.arena_w

        max_p = np.max(heatmap_values_2d)
        norm_values = heatmap_values_2d / max_p if max_p > 0 else heatmap_values_2d

        for r in range(self.arena_h):
            for c in range(self.arena_w):
                p = norm_values[r, c]
                color = (255, 0, 0, int(p * 200))
                pygame.draw.rect(heatmap_surface, color, (c * tile_w, r * tile_h, tile_w, tile_h))

        action_cell = self._action_position(debug_policy)
        if action_cell is not None:
            a_r, a_c = action_cell
            pygame.draw.rect(heatmap_surface, (255, 255, 0, 255), (a_c * tile_w, a_r * tile_h, tile_w, tile_h), 2)

        top_pos_idx = int(np.argmax(pos_probs))
        top_r, top_c = self._policy_index_to_arena_cell(top_pos_idx, player_idx)
        pygame.draw.rect(heatmap_surface, (0, 255, 255, 255), (top_c * tile_w, top_r * tile_h, tile_w, tile_h), 2)

        surface.blit(heatmap_surface, (0, 0))

        pygame.font.init()
        font = pygame.font.SysFont(None, 24)

        finite_logits = None
        logit_span = None
        if pos_logits is not None:
            finite_logits = pos_logits[np.isfinite(pos_logits)]
            if finite_logits.size:
                logit_span = float(np.max(finite_logits) - np.min(finite_logits))

        action_cell = self._action_position(debug_policy)
        action_prob = None
        action_logit = None
        if action_cell is not None:
            action_r, action_c = action_cell
            action_idx = action_r * self.arena_w + action_c
            if player_idx == 2:
                action_idx = (self.arena_h - 1 - action_r) * self.arena_w + (self.arena_w - 1 - action_c)
            action_prob = float(pos_probs[action_idx])
            if pos_logits is not None:
                action_logit = float(pos_logits[action_idx])

        self._frame_idx += 1
        frame_txt = font.render(f"Frame: {self._frame_idx}", True, (200, 200, 200))
        surface.blit(frame_txt, (W + 10, 20))

        player_txt = font.render(f"P{player_idx}", True, (200, 200, 200))
        surface.blit(player_txt, (W + 10, 45))

        skip_txt = font.render(f"Skip: {skip_prob:.2f}", True, (200, 200, 200))
        surface.blit(skip_txt, (W + 10, 70))

        top_txt = font.render(f"Top: {top_c},{top_r}", True, (200, 200, 200))
        surface.blit(top_txt, (W + 10, 95))

        top_prob_txt = font.render(f"P: {pos_probs[top_pos_idx]:.3f}", True, (200, 200, 200))
        surface.blit(top_prob_txt, (W + 10, 120))

        if logit_span is not None:
            span_txt = font.render(f"L span: {logit_span:.3f}", True, (200, 200, 200))
            surface.blit(span_txt, (W + 10, 145))

        if action_prob is not None:
            action_prob_txt = font.render(f"A p: {action_prob:.3f}", True, (200, 200, 200))
            surface.blit(action_prob_txt, (W + 10, 170))

        if action_logit is not None and np.isfinite(action_logit):
            action_logit_txt = font.render(f"A l: {action_logit:.3f}", True, (200, 200, 200))
            surface.blit(action_logit_txt, (W + 10, 195))

        for i, p in enumerate(deck_probs):
            y = 230 + i * 60
            pygame.draw.rect(surface, (80, 80, 80), (W + 10, y, 40, 40))

            fill_h = int(40 * p)
            pygame.draw.rect(surface, (0, 200, 0), (W + 10, y + 40 - fill_h, 40, fill_h))

            txt_c = font.render(f"C{i}", True, (255, 255, 255))
            surface.blit(txt_c, (W + 60, y + 5))
            txt_p = font.render(f"{p:.2f}", True, (255, 255, 255))
            surface.blit(txt_p, (W + 60, y + 25))

        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
