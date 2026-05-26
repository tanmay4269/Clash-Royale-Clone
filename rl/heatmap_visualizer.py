import numpy as np
import gymnasium as gym
import pygame
import torch as t
from game.entity import EntityType

class HeatmapVisualizerWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        pygame.init()
        self.debug_policy = None
        self.arena_w = self.unwrapped.arena.width
        self.arena_h = self.unwrapped.arena.height
        self._frame_idx = 0

    def update(
        self,
        *,
        value_1,
        value_2,
        skip_prob_1,
        skip_prob_2,
        deck_probs_1,
        deck_probs_2,
        pos_probs_1,
        pos_probs_2,
        action_1=None,
        action_2=None,
    ):
        self.debug_policy = {
            "value_1": float(value_1),
            "value_2": float(value_2),
            "skip_prob_1": float(skip_prob_1),
            "skip_prob_2": float(skip_prob_2),
            "deck_probs_1": np.asarray(deck_probs_1, dtype=np.float32),
            "deck_probs_2": np.asarray(deck_probs_2, dtype=np.float32),
            "pos_probs_1": np.asarray(pos_probs_1, dtype=np.float32),
            "pos_probs_2": np.asarray(pos_probs_2, dtype=np.float32),
            "action_1": action_1,
            "action_2": action_2,
        }

    def _policy_index_to_arena_cell(self, pos_idx, player_idx):
        pos_idx = int(pos_idx)
        row = pos_idx // self.arena_w
        col = pos_idx % self.arena_w
        if player_idx == 2:
            row = self.arena_h - 1 - row
            col = self.arena_w - 1 - col
        return row, col

    def _action_position(self, action, player_idx):
        if action is None:
            return None
        pos = action.get("position")
        if pos is None:
            return None
        if hasattr(pos, "detach"):
            pos = pos.detach().cpu().item()
        return self._policy_index_to_arena_cell(pos, player_idx)

    def render(self):
        # We don't just call self.env.render() because that is 288x512 and crops HUD panel.
        # Instead, we render the entire screen ourselves.
        # Screen layout:
        # [ 30 px Eval Bar ] [ 288 px Arena ] [ 200 px HUD Panel ]
        # Total width = 518, Height = 512.
        
        W = 288
        H = 512
        PANEL_WIDTH = 200
        EVAL_BAR_WIDTH = 30
        
        total_W = EVAL_BAR_WIDTH + W + PANEL_WIDTH
        
        # Create final surface
        surface = pygame.Surface((total_W, H))
        surface.fill((24, 24, 28))
        
        # Create surface for Arena + HUD panel
        arena_hud_surface = pygame.Surface((W + PANEL_WIDTH, H))
        
        # Render the arena and HUD panel directly onto arena_hud_surface
        self.unwrapped.arena.render(arena_hud_surface, scale_factor=1.0)
        
        if self.debug_policy is not None:
            dp = self.debug_policy
            
            # --- 1. Draw Heatmaps on Arena ---
            heatmap_surface = pygame.Surface((W, H), pygame.SRCALPHA)
            
            tile_h = H // self.arena_h
            tile_w = W // self.arena_w
            
            # Normalization
            max_p1 = np.max(dp["pos_probs_1"])
            norm_p1 = dp["pos_probs_1"] / max_p1 if max_p1 > 0 else dp["pos_probs_1"]
            norm_p1_2d = norm_p1.reshape(self.arena_h, self.arena_w)
            
            # Player 2 needs 180 rotation
            max_p2 = np.max(dp["pos_probs_2"])
            norm_p2 = dp["pos_probs_2"] / max_p2 if max_p2 > 0 else dp["pos_probs_2"]
            norm_p2_2d = norm_p2.reshape(self.arena_h, self.arena_w)
            norm_p2_2d_rot = np.flip(norm_p2_2d, axis=(0, 1))
            
            # Draw overlays
            for r in range(self.arena_h):
                for c in range(self.arena_w):
                    p1 = norm_p1_2d[r, c]
                    p2 = norm_p2_2d_rot[r, c]
                    
                    if p1 > 0:
                        # Red overlay for Player 1
                        pygame.draw.rect(heatmap_surface, (230, 40, 40, int(p1 * 130)), (c * tile_w, r * tile_h, tile_w, tile_h))
                    if p2 > 0:
                        # Blue overlay for Player 2
                        pygame.draw.rect(heatmap_surface, (40, 100, 230, int(p2 * 130)), (c * tile_w, r * tile_h, tile_w, tile_h))
            
            # Chosen actions / Top probability cells
            # Player 1 Top Probability (Cyan)
            top_pos_1 = int(np.argmax(dp["pos_probs_1"]))
            top_r1, top_c1 = self._policy_index_to_arena_cell(top_pos_1, 1)
            pygame.draw.rect(heatmap_surface, (0, 255, 255, 255), (top_c1 * tile_w, top_r1 * tile_h, tile_w, tile_h), 2)
            
            # Player 2 Top Probability (Orange)
            top_pos_2 = int(np.argmax(dp["pos_probs_2"]))
            top_r2, top_c2 = self._policy_index_to_arena_cell(top_pos_2, 2)
            pygame.draw.rect(heatmap_surface, (255, 165, 0, 255), (top_c2 * tile_w, top_r2 * tile_h, tile_w, tile_h), 2)
            
            # Player 1 Action (Yellow)
            action_cell_1 = self._action_position(dp["action_1"], 1)
            if action_cell_1 is not None:
                a_r1, a_c1 = action_cell_1
                pygame.draw.rect(heatmap_surface, (255, 255, 0, 255), (a_c1 * tile_w, a_r1 * tile_h, tile_w, tile_h), 2)
                
            # Player 2 Action (Magenta)
            action_cell_2 = self._action_position(dp["action_2"], 2)
            if action_cell_2 is not None:
                a_r2, a_c2 = action_cell_2
                pygame.draw.rect(heatmap_surface, (255, 0, 255, 255), (a_c2 * tile_w, a_r2 * tile_h, tile_w, tile_h), 2)
                
            # Blit heatmap to the arena portion
            arena_hud_surface.blit(heatmap_surface, (0, 0))
            
            # --- 2. Draw deck probability bars inside HUD panel ---
            if not pygame.font.get_init():
                pygame.font.init()
            font_small = pygame.font.SysFont(None, 12)
            font_medium = pygame.font.SysFont(None, 16)
            
            # Constants from _draw_player_hud_block
            title_x = W + 7
            
            # Player 1 (y_offset = 20)
            card_y_1 = 20 + 48
            card_w = 30
            card_h = 42
            bar_h = 40
            
            for i, p in enumerate(dp["deck_probs_1"]):
                card_x = title_x + i * 36
                bar_y = card_y_1 + card_h + 3
                # Draw vertical progress bar
                pygame.draw.rect(arena_hud_surface, (40, 40, 45), (card_x, bar_y, card_w, bar_h), border_radius=2)
                fill_h = int(bar_h * p)
                if fill_h > 0:
                    pygame.draw.rect(arena_hud_surface, (0, 220, 100), (card_x, bar_y + bar_h - fill_h, card_w, fill_h), border_radius=2)
                pygame.draw.rect(arena_hud_surface, (100, 100, 100), (card_x, bar_y, card_w, bar_h), width=1, border_radius=2)
                
                # Draw percentage text below
                txt = font_small.render(f"{p*100:.0f}%", True, (200, 200, 200))
                arena_hud_surface.blit(txt, (card_x + (card_w - txt.get_width()) // 2, bar_y + bar_h + 1))
                
            # Player 1 Skip Prob
            next_x = title_x + 4 * 36 + 6
            next_w = 22
            skip_bar_y_1 = card_y_1 + card_h + 3
            pygame.draw.rect(arena_hud_surface, (40, 40, 45), (next_x, skip_bar_y_1, next_w, bar_h), border_radius=2)
            fill_skip_1 = int(bar_h * dp["skip_prob_1"])
            if fill_skip_1 > 0:
                pygame.draw.rect(arena_hud_surface, (150, 150, 150), (next_x, skip_bar_y_1 + bar_h - fill_skip_1, next_w, fill_skip_1), border_radius=2)
            pygame.draw.rect(arena_hud_surface, (100, 100, 100), (next_x, skip_bar_y_1, next_w, bar_h), width=1, border_radius=2)
            
            skip_txt_1 = font_small.render(f"{dp['skip_prob_1']*100:.0f}%", True, (150, 150, 150))
            arena_hud_surface.blit(skip_txt_1, (next_x + (next_w - skip_txt_1.get_width()) // 2, skip_bar_y_1 + bar_h + 1))
            
            # Player 2 (y_offset = 320)
            card_y_2 = 320 + 48
            for i, p in enumerate(dp["deck_probs_2"]):
                card_x = title_x + i * 36
                bar_y = card_y_2 + card_h + 3
                # Draw vertical progress bar
                pygame.draw.rect(arena_hud_surface, (40, 40, 45), (card_x, bar_y, card_w, bar_h), border_radius=2)
                fill_h = int(bar_h * p)
                if fill_h > 0:
                    pygame.draw.rect(arena_hud_surface, (0, 180, 255), (card_x, bar_y + bar_h - fill_h, card_w, fill_h), border_radius=2)
                pygame.draw.rect(arena_hud_surface, (100, 100, 100), (card_x, bar_y, card_w, bar_h), width=1, border_radius=2)
                
                # Draw percentage text below
                txt = font_small.render(f"{p*100:.0f}%", True, (200, 200, 200))
                arena_hud_surface.blit(txt, (card_x + (card_w - txt.get_width()) // 2, bar_y + bar_h + 1))
                
            # Player 2 Skip Prob
            skip_bar_y_2 = card_y_2 + card_h + 3
            pygame.draw.rect(arena_hud_surface, (40, 40, 45), (next_x, skip_bar_y_2, next_w, bar_h), border_radius=2)
            fill_skip_2 = int(bar_h * dp["skip_prob_2"])
            if fill_skip_2 > 0:
                pygame.draw.rect(arena_hud_surface, (150, 150, 150), (next_x, skip_bar_y_2 + bar_h - fill_skip_2, next_w, fill_skip_2), border_radius=2)
            pygame.draw.rect(arena_hud_surface, (100, 100, 100), (next_x, skip_bar_y_2, next_w, bar_h), width=1, border_radius=2)
            
            skip_txt_2 = font_small.render(f"{dp['skip_prob_2']*100:.0f}%", True, (150, 150, 150))
            arena_hud_surface.blit(skip_txt_2, (next_x + (next_w - skip_txt_2.get_width()) // 2, skip_bar_y_2 + bar_h + 1))

            # --- 3. Draw vertical Eval/Win-rate Bar ---
            # win_prob for Player 1 (Red on top, Blue on bottom)
            win_prob = 1.0 / (1.0 + np.exp(-dp["value_1"]))
            fill_h = int(H * win_prob)
            
            # Player 1 is red, Player 2 is blue
            p1_color = (200, 40, 40)
            p2_color = (40, 100, 200)
            
            # Red top (P1), Blue bottom (P2)
            pygame.draw.rect(surface, p1_color, (0, 0, EVAL_BAR_WIDTH, fill_h))
            pygame.draw.rect(surface, p2_color, (0, fill_h, EVAL_BAR_WIDTH, H - fill_h))
            
            # Mid divider line (50%)
            pygame.draw.line(surface, (255, 255, 255), (0, H // 2), (EVAL_BAR_WIDTH, H // 2), 2)
            
            # Text inside Eval Bar (centered)
            y_mid_1 = fill_h // 2
            y_mid_2 = fill_h + (H - fill_h) // 2
            
            if fill_h > 25:
                p1_txt = font_medium.render(f"{win_prob*100:.0f}%", True, (255, 255, 255))
                surface.blit(p1_txt, (EVAL_BAR_WIDTH // 2 - p1_txt.get_width() // 2, y_mid_1 - p1_txt.get_height() // 2))
                
            if (H - fill_h) > 25:
                p2_txt = font_medium.render(f"{(1 - win_prob)*100:.0f}%", True, (255, 255, 255))
                surface.blit(p2_txt, (EVAL_BAR_WIDTH // 2 - p2_txt.get_width() // 2, y_mid_2 - p2_txt.get_height() // 2))
                
            self._frame_idx += 1

        else:
            # Fallback if no debug_policy set yet
            pygame.draw.rect(surface, (40, 40, 45), (0, 0, EVAL_BAR_WIDTH, H))
            pygame.draw.line(surface, (255, 255, 255), (0, H // 2), (EVAL_BAR_WIDTH, H // 2), 2)
            
        # Blit Arena + HUD to main surface
        surface.blit(arena_hud_surface, (EVAL_BAR_WIDTH, 0))
        
        # Convert and transpose to HWC for Gym/RecordVideo
        frame_3d = pygame.surfarray.array3d(surface)
        return np.transpose(frame_3d, (1, 0, 2))
