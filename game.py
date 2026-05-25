"""
game.py: Manual test harness for the Clash Royale arena.

Player 1: Human (keyboard + mouse, same as before).
Player 2: AI bot driven by BotNet (overfit modes: random / skip / scripted) or trained model from checkpoint.

Controls
--------
1 / 2       : switch active player (human deploy side)
K / G / P   : pick active card  (Knight / Giant / MiniPEKKA)
R / S / C   : change bot mode   (Random / Skip / Scripted)
Left-click  : deploy active card at mouse position

Usage
-----
  python game.py                     # default: random bot
  python game.py --opponent skip
  python game.py --opponent scripted
  python game.py --opponent /path/to/checkpoint.pt   # load trained model
"""

import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

import psutil
import os
import numpy as np

import torch as t
import torch.nn.functional as F

from utils import *
from arena import Arena
from network import BotNet, ActorCritic

import gymnasium as gym
import cr_gym_env
from cr_flatten_norm_wrapper import CRFlattenNormWrapper

from entities.troops.knight    import Knight
from entities.troops.giant     import Giant
from entities.troops.mini_pekka import MiniPEKKA


def split_observations(obs, env, arena, max_num_objects):
    """Return (obs_1, obs_2) dicts formatted for the network."""
    flat_card_space  = env.flat_card_space
    card_dim         = flat_card_space.shape[0]
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
            "constant", 0,
        ).unsqueeze(0)

    def rot180(entities):
        r = entities.clone()
        r[..., position_x_idx] *= -1
        r[..., position_y_idx] *= -1
        return r

    p1t = pad(arr1, n1)
    p2t = pad(arr2, n2)

    ct1 = t.tensor(np.array(obs["player_1_crown_towers"], dtype=np.float32)).unsqueeze(0)
    ct2 = t.tensor(np.array(obs["player_2_crown_towers"], dtype=np.float32)).unsqueeze(0)

    gcf = t.tensor(np.array(obs["game_completion_fraction"], dtype=np.float32)).reshape(-1, 1)

    obs_1 = {
        "game_completion_fraction": gcf,
        "elixirs":                  t.tensor(np.array(obs["player_1_elixirs"], dtype=np.float32)).reshape(-1, 1),
        "my_cards":                 p1t,
        "opponent_cards":           p2t,
        "my_crown_towers":          ct1,
        "opponent_crown_towers":    ct2,
    }

    obs_2 = {
        "game_completion_fraction": gcf,
        "elixirs":                  t.tensor(np.array(obs["player_2_elixirs"], dtype=np.float32)).reshape(-1, 1),
        "my_cards":                 rot180(p2t),
        "opponent_cards":           rot180(p1t),
        "my_crown_towers":          rot180(ct2),
        "opponent_crown_towers":    rot180(ct1),
    }

    return obs_1, obs_2


def join_actions(action_1, action_2, arena):
    """Merge two per-player action dicts into the env-style dict."""
    def scalar_int(v):
        return int(v.detach().cpu().item()) if hasattr(v, "detach") else int(v)

    def pos_to_xy(action):
        idx = scalar_int(action["position"])
        return idx % arena.width, idx // arena.width

    def rot_xy_180(x, y):
        return arena.width - 1 - x, arena.height - 1 - y

    return {
        "player_1_skip":          scalar_int(action_1["skip"]),
        "player_1_card_idx":      scalar_int(action_1["deck_idx"]),
        "player_1_card_position": pos_to_xy(action_1),

        "player_2_skip":          scalar_int(action_2["skip"]),
        "player_2_card_idx":      scalar_int(action_2["deck_idx"]),
        "player_2_card_position": rot_xy_180(*pos_to_xy(action_2)),
    }


CARD_MAP = {0: Knight, 1: Giant, 2: MiniPEKKA}


def apply_action_to_arena(arena, joined_action):
    """Deploy cards for both players based on a joined action dict."""
    for idx in [1, 2]:
        if joined_action[f"player_{idx}_skip"] == 1:
            continue

        owner = arena.player_side_1 if idx == 1 else arena.player_side_2
        x, y  = joined_action[f"player_{idx}_card_position"]
        card_cls = CARD_MAP.get(joined_action[f"player_{idx}_card_idx"], Knight)
        card = card_cls(owner, y, x)

        if arena.deploy_entity(card):
            owner.add_object(card)


class Game:
    def __init__(self, opponent_mode: str = "random"):
        self._env_raw  = gym.make("ClashRoyaleEnv-v0")
        self._env_wrap = CRFlattenNormWrapper(self._env_raw)

        # Share the arena from the gym env so render / update / click
        # all go to the exact same object.
        self._env_raw.reset()
        self.arena = self._env_raw.unwrapped.arena

        self.max_num_objects = self.arena.max_num_objects

        self.opponent_mode = opponent_mode
        self._bot          = self._make_bot(opponent_mode)

        self.width  = self.arena.width  * self.arena.tile_size
        self.height = self.arena.height * self.arena.tile_size

        pygame.init()
        self.screen  = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Clash Royale: vs {opponent_mode} bot")
        self.clock   = pygame.time.Clock()
        self.running = True
        self.dt      = 0

        self.arena._debug_active_player = 1
        self.arena._debug_active_card   = Knight

        print(self._help_text())

    def _make_bot(self, mode: str) -> BotNet:
        arena = self.arena
        scale            = arena.tile_size
        occupancy_grid   = arena.cell_occupancy
        tiled            = np.where(occupancy_grid == 1, 1, 0)[scale//2::scale, scale//2::scale]
        mask             = tiled.astype(bool).T
        mask[arena.height//2:, :] = True   # opponent's half always invalid for bot (player-2 view is mirrored)
        mask_tensor = t.tensor(mask).flatten()

        # Check if mode is a checkpoint path
        checkpoint_path = os.path.expanduser(mode)
        if os.path.isfile(checkpoint_path):
            # Load ActorCritic model from checkpoint
            network = ActorCritic(
                entity_encoder_in_ch=16,
                entity_encoder_mid_ch=64,
                entity_encoder_out_ch=32,

                trunk_extra_in_ch=2,
                trunk_mid_ch=128,
                
                num_cards_in_deck=self._env_raw.unwrapped.NUM_CARDS_IN_DECK,
                max_num_cards=self.arena.max_num_objects,
                position_space_width=arena.width,
                position_space_height=arena.height,
                invalid_position_mask=mask_tensor,
                max_elixirs=10,
            )
            try:
                state_dict = t.load(checkpoint_path, map_location=None if t.cuda.is_available() else 'cpu', weights_only=True)
                network.load_state_dict(state_dict)
                print(f"Loaded checkpoint: {checkpoint_path}")
                return network
            except Exception as e:
                print(f"Error loading checkpoint {checkpoint_path}: {e}")
                exit(1)

        # Default: create a BotNet with the specified mode
        return BotNet(
            bot_type              = mode,
            invalid_position_mask = mask_tensor,
            num_cards_in_deck     = self._env_raw.unwrapped.NUM_CARDS_IN_DECK,
            position_space_width  = arena.width,
            position_space_height = arena.height,
        )

    def _get_obs_wrapped(self):
        """Pull a fresh observation from the arena via the wrapper."""
        raw_obs = self._env_raw.unwrapped._get_obs()
        return self._env_wrap.observation(raw_obs)

    def _bot_action(self, obs_wrapped):
        """Ask the bot for player-2's action."""
        _, obs_2 = split_observations(
            obs_wrapped, self._env_wrap, self.arena, self.max_num_objects
        )
        with t.no_grad():
            action_2, _, _, _ = self._bot.get_action_and_value(obs_2)
        return action_2

    @staticmethod
    def _help_text():
        return (
            "\n=== Clash Royale Manual Test ===\n"
            "  1 / 2        -> switch deploy side (player 1 / 2)\n"
            "  K / G / P    -> set card  (Knight / Giant / MiniPEKKA)\n"
            "  R / S / C    -> bot mode  (Random / Skip / Scripted)\n"
            "  Left-click   -> deploy card\n"
            "================================\n"
        )

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.arena.on_click()

            elif event.type == pygame.KEYDOWN:
                # Player-side selection
                if event.key == pygame.K_1:
                    self.arena._debug_active_player = 1
                    print("Active player -> 1")
                elif event.key == pygame.K_2:
                    self.arena._debug_active_player = 2
                    print("Active player -> 2")

                # Card selection
                elif event.key == pygame.K_k:
                    self.arena._debug_active_card = Knight
                    print("Active card -> Knight")
                elif event.key == pygame.K_g:
                    self.arena._debug_active_card = Giant
                    print("Active card -> Giant")
                elif event.key == pygame.K_p:
                    self.arena._debug_active_card = MiniPEKKA
                    print("Active card -> MiniPEKKA")

                # Bot-mode switching
                elif event.key == pygame.K_r:
                    self.opponent_mode = "random"
                    self._bot = self._make_bot("random")
                    pygame.display.set_caption("Clash Royale: vs random bot")
                    print("Bot mode -> random")
                elif event.key == pygame.K_s:
                    self.opponent_mode = "skip"
                    self._bot = self._make_bot("skip")
                    pygame.display.set_caption("Clash Royale: vs skip bot")
                    print("Bot mode -> skip")
                elif event.key == pygame.K_c:
                    self.opponent_mode = "scripted"
                    self._bot = self._make_bot("scripted")
                    pygame.display.set_caption("Clash Royale: vs scripted bot")
                    print("Bot mode -> scripted")

        self.arena.render(self.screen)

        obs_wrapped = self._get_obs_wrapped()
        action_2    = self._bot_action(obs_wrapped)

        # Human player 1 always skips in the bot-driven path;
        # the human deploys manually via on_click().
        action_1_skip = {"skip": t.tensor(1), "deck_idx": t.tensor(0), "position": t.tensor(0)}
        joined = join_actions(action_1_skip, action_2, self.arena)

        # Only apply the bot's deploy (player_2); human clicks are handled by arena.on_click
        bot_action = {
            "player_1_skip":          1,          # human side: no auto deploy
            "player_1_card_idx":      0,
            "player_1_card_position": (0, 0),
            "player_2_skip":          joined["player_2_skip"],
            "player_2_card_idx":      joined["player_2_card_idx"],
            "player_2_card_position": joined["player_2_card_position"],
        }
        apply_action_to_arena(self.arena, bot_action)

        terminated, truncated = self.arena.update(self.dt)
        if terminated or truncated:
            self.running = False
            return

        pygame.display.flip()
        self.dt = self.clock.tick(60) / 1000

        ### * DEBUG * ###
        if False:
            process = psutil.Process(os.getpid())
            memory  = process.memory_info().rss
            print(f"FPS: {self.clock.get_fps():.2f}\t RAM: {memory/1024/1024:.2f} MB")

    def run(self):
        while self.running:
            self.update()
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual test: play vs a bot")
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        help="Bot opponent mode ('random', 'skip', 'scripted') or path to checkpoint file to load",
    )
    args = parser.parse_args()

    game = Game(opponent_mode=args.opponent)
    game.run()