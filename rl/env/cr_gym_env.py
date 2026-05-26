from game.utils import *
from rl.env.cr_flatten_norm_wrapper import CRFlattenNormWrapper

import os
import time

PROFILE_ENV = os.environ.get("PROFILE_ENV") == "1"

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from game.arena import Arena
from game.entity import EntityType, Entity
from game.entities.buildings.crown_tower import CrownTower

from game.entities.troops import *


register(
    id="ClashRoyaleEnv-v0",
    entry_point="rl.env.cr_gym_env:ClashRoyaleEnv",
)

class ClashRoyaleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self, 
        render_mode=None,
        step_penalty=0.0,
        tower_damage_reward_scale=1/5000.0,
        tower_distruction_reward=0.5,
        winning_reward=5.0
    ):
        self.arena = Arena()

        ### Observation Space ###
        INT_MAX = 1_000_000
        FLOAT_MAX = 1_000_000.0

        def get_entity_space():
            return spaces.Dict({
                "deploy_cost": spaces.Box(0.0, self.arena.player_side_1.max_elixirs, shape=(1,), dtype=np.float32),
                "deploy_delay": spaces.Box(0.0, 10.0, shape=(1,), dtype=np.float32),
                
                "entity_type": spaces.Discrete(EntityType.num_types()), 

                "target_types": spaces.MultiBinary(EntityType.num_types()), 

                # x, y
                "position": spaces.Box(  
                    low=np.array([0.0, 0.0], dtype=np.float32),
                    high=np.array([
                        self.arena.width * self.arena.tile_size,
                        self.arena.height * self.arena.tile_size
                    ], dtype=np.float32),
                    dtype=np.float32
                ),
                
                # "health": spaces.Box(0.0, FLOAT_MAX),
                "health": spaces.Box(0.0, EntityRegistry.aggregate("health")["max"], shape=(1,), dtype=np.float32),
                "hitpoints": spaces.Box(0.0, EntityRegistry.aggregate("hitpoints")["max"], shape=(1,), dtype=np.float32),

                "damage": spaces.Box(0.0, EntityRegistry.aggregate("damage")["max"], shape=(1,), dtype=np.float32),
                "attack_radius_cells": spaces.Box(0.0, EntityRegistry.aggregate("attack_radius_cells")["max"], shape=(1,), dtype=np.float32),

                "hit_speed": spaces.Box(0.0, EntityRegistry.aggregate("hit_speed")["max"], shape=(1,), dtype=np.float32),
                "first_hit_speed": spaces.Box(0.0, EntityRegistry.aggregate("first_hit_speed")["max"], shape=(1,), dtype=np.float32),
                
                # Maybe add what stage of its hit cycle is it if the model can't seem to figure this out
            })

        crown_towers_space = spaces.Dict({
            "king_tower": get_entity_space(),
            "princess_tower_1": get_entity_space(),
            "princess_tower_2": get_entity_space(),
        })

        self.observation_space = spaces.Dict({
            "game_completion_fraction": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "player_1_elixirs": spaces.Discrete(self.arena.player_side_1.max_elixirs + 1),
            "player_1_cards": spaces.Sequence(get_entity_space()),
            "player_1_crown_towers": crown_towers_space,

            "player_2_elixirs": spaces.Discrete(self.arena.player_side_2.max_elixirs + 1),
            "player_2_cards": spaces.Sequence(get_entity_space()),
            "player_2_crown_towers": crown_towers_space,
        })

        ### Action Space ###
        self.NUM_CARDS_IN_DECK = 3  # TODO: ofc find a better way
        position_space = spaces.Box(  
            low=np.array([0.0, 0.0]),
            high=np.array([
                self.arena.width,
                self.arena.height
            ]),
            dtype=int
        )

        self.action_space = spaces.Dict({
            "player_1_skip": spaces.Discrete(2), # bool
            "player_1_card_idx": spaces.Discrete(self.NUM_CARDS_IN_DECK),
            "player_1_card_position": position_space,

            "player_2_skip": spaces.Discrete(2), # bool
            "player_2_card_idx": spaces.Discrete(self.NUM_CARDS_IN_DECK),
            "player_2_card_position": position_space,
        })

        ### Rewards ###
        self.step_penalty = step_penalty
        self.tower_damage_reward_scale = tower_damage_reward_scale
        self.tower_distruction_reward = tower_distruction_reward
        self.winning_reward = winning_reward


        ### Helpers ###
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.screen = None
        self.clock = None

        self.FIXED_DT = 1/4  # 4 fps simulator

        self._cur_obs = None

        # Per-episode accumulators (populated during step, read in _get_info at episode end)
        self._ep_steps = 0
        self._ep_elixir_sum = 0.0
        self._ep_skips = 0
        self._ep_decks = []
        self._ep_pos_x = []
        self._ep_pos_y = []


    def _get_obs(self):
        obs = {}

        ### Basics ###
        obs["game_completion_fraction"] = np.array([self.arena.elapsed_time / self.arena.game_duration], dtype=np.float32)

        obs["player_1_elixirs"] = np.int64(self.arena.player_side_1.elixirs)
        obs["player_2_elixirs"] = np.int64(self.arena.player_side_2.elixirs)

        ### Cards ###
        obs["player_1_cards"], obs["player_2_cards"] = [], []
        for obj in self.arena.objects:
            if isinstance(obj, CrownTower):
                continue
            
            owner = obs["player_1_cards"] if obj.owner == self.arena.player_side_1 else obs["player_2_cards"]
            owner.append({
                "deploy_cost": np.array([obj.deploy_cost], dtype=np.float32),
                "deploy_delay": np.array([obj.deploy_delay], dtype=np.float32),

                "entity_type": obj.entity_type,
                "target_types": np.array([
                    int(EntityType.GROUND   in obj.target_types),
                    int(EntityType.AIR      in obj.target_types),
                    int(EntityType.BUILDING in obj.target_types),
                ], dtype=np.int8),

                "position": np.array([obj.position.x, obj.position.y], dtype=np.float32),
                
                "health": np.array([obj.health], dtype=np.float32),
                "hitpoints": np.array([obj.hitpoints], dtype=np.float32),
                
                "damage": np.array([obj.damage], dtype=np.float32),
                "attack_radius_cells": np.array([obj.attack_radius_cells], dtype=np.float32),

                "hit_speed": np.array([obj.hit_speed], dtype=np.float32),
                "first_hit_speed": np.array([obj.first_hit_speed], dtype=np.float32),
            })


        ### Crown Towers ###
        obs["player_1_crown_towers"], obs["player_2_crown_towers"] = {}, {}

        # TODO: this is too hidious, should have a getter in the entity class
        for side_name in ["player_1", "player_2"]:
            side = self.arena.player_side_1 if side_name == "player_1" else \
                self.arena.player_side_2

            for tower_name in ["king_tower", "princess_tower_1", "princess_tower_2"]:
                tower = None
                if tower_name == "king_tower":
                    tower = side.king_tower
                elif tower_name == "princess_tower_1":
                    tower = side.princess_tower_1
                else:
                    tower = side.princess_tower_2

                obs[f"{side_name}_crown_towers"][tower_name] = {
                    "deploy_cost": tower.deploy_cost,
                    "deploy_delay": tower.deploy_delay,

                    "entity_type": tower.entity_type,
                    "target_types": np.array([
                        int(EntityType.GROUND   in tower.target_types),
                        int(EntityType.AIR      in tower.target_types),
                        int(EntityType.BUILDING in tower.target_types),
                    ], dtype=np.int8),
                    
                    "position": np.array([tower.position.x, tower.position.y]),

                    "health": tower.health,
                    "hitpoints": tower.hitpoints,

                    "damage": tower.damage,
                    "attack_radius_cells": tower.attack_radius_cells,

                    "hit_speed": tower.hit_speed,
                    "first_hit_speed": tower.first_hit_speed
                }

        return obs


    def _get_info(self, terminated=False, truncated=False):
        """
        Returns diagnostic info that crosses the process boundary for parallel envs.
        At episode end (terminated or truncated) we also include per-episode aggregates
        so the main process can log them without needing direct env access.
        """
        info = {}

        if terminated or truncated:
            # Tower kills: count towers whose health dropped to 0
            towers_killed_by_p1 = sum(
                1 for t_name in ["king_tower", "princess_tower_1", "princess_tower_2"]
                if float(self._cur_obs["player_2_crown_towers"][t_name]["health"]) <= 0
            )
            towers_killed_by_p2 = sum(
                1 for t_name in ["king_tower", "princess_tower_1", "princess_tower_2"]
                if float(self._cur_obs["player_1_crown_towers"][t_name]["health"]) <= 0
            )

            winner = self.arena.winner  # 1, 2, or None (draw / truncated)
            sudden_death_win = (
                terminated
                and winner is not None
                and self.arena.has_sudden_death_started
                and not (  # king tower still alive → princess-tower kill in SD
                    float(self._cur_obs["player_1_crown_towers"]["king_tower"]["health"]) <= 0
                    or float(self._cur_obs["player_2_crown_towers"]["king_tower"]["health"]) <= 0
                )
            )

            info["episode"] = {
                "towers_killed_by_p1": towers_killed_by_p1,
                "towers_killed_by_p2": towers_killed_by_p2,
                "winner":              winner,
                "sudden_death_win":    sudden_death_win,
                "avg_elixir_p1":       self._ep_elixir_sum / max(1, self._ep_steps),
                "skip_ratio":          self._ep_skips / max(1, self._ep_steps),
                "deck_indices":        list(self._ep_decks),
                "pos_x":               list(self._ep_pos_x),
                "pos_y":               list(self._ep_pos_y),
                "ep_steps":            self._ep_steps,
            }

        return info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.arena = Arena()

        # Reset per-episode accumulators
        self._ep_steps = 0
        self._ep_elixir_sum = 0.0
        self._ep_skips = 0
        self._ep_decks = []
        self._ep_pos_x = []
        self._ep_pos_y = []

        self._cur_obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return self._cur_obs, info


    def step(self, action):
        # Accumulate per-step diagnostics for player 1
        self._ep_steps += 1
        self._ep_elixir_sum += float(self.arena.player_side_1.elixirs)
        p1_skip = int(action["player_1_skip"])
        self._ep_skips += p1_skip
        if p1_skip == 0:
            self._ep_decks.append(int(action["player_1_card_idx"]))
            pos = action["player_1_card_position"]
            self._ep_pos_x.append(int(pos[0]))
            self._ep_pos_y.append(int(pos[1]))

        for idx in [1, 2]:
            if action[f"player_{idx}_skip"] == 1:
                continue

            owner = self.arena.player_side_1 if idx == 1 \
                else self.arena.player_side_2

            x, y = action[f"player_{idx}_card_position"]
            
            # TODO: use entity registers instead
            card = None
            if action[f"player_{idx}_card_idx"] == 0:
                card = Knight
            elif action[f"player_{idx}_card_idx"] == 1:
                card = Giant
            elif action[f"player_{idx}_card_idx"] == 2:
                card = MiniPEKKA

            card_instance = card(owner, y, x)
            if self.arena.deploy_entity(card_instance):
                owner.add_object(card_instance)

        prev_obs = self._cur_obs  # Saving for reward calculation
        
        # For stable physics updates
        terminated, truncated = False, False
        if PROFILE_ENV:
            t0 = time.perf_counter()
            
        for _ in range(10):
            _terminated, _truncated = self.arena.update(self.FIXED_DT / 10)
            terminated, truncated = terminated or _terminated, truncated or _truncated
            
        if PROFILE_ENV:
            print(f"(PROFILE) arena.update (10x): {(time.perf_counter() - t0)*1000:.3f} ms")
        
        if PROFILE_ENV:
            t0 = time.perf_counter()
            
        self._cur_obs = self._get_obs()
        
        if PROFILE_ENV:
            print(f"(PROFILE) cr_gym_env._get_obs: {(time.perf_counter() - t0)*1000:.3f} ms")


        reward = self._get_reward(prev_obs, terminated, truncated)
        info = self._get_info(terminated=terminated, truncated=truncated)

        if terminated or truncated:
            # Reset accumulators so a fresh episode starts clean after auto-reset
            self._ep_steps = 0
            self._ep_elixir_sum = 0.0
            self._ep_skips = 0
            self._ep_decks = []
            self._ep_pos_x = []
            self._ep_pos_y = []

        if self.render_mode == "human":
            self._render_frame()

        return self._cur_obs, reward, terminated, truncated, info


    def _get_reward(self, prev_obs, terminated=False, truncated=False):
        player_1_reward, player_2_reward = -self.step_penalty, -self.step_penalty
        
        if prev_obs is None:
            return player_1_reward, player_2_reward

        # 1. Tower HP change (shaping) + 2. Tower destruction bonus
        for idx in [1, 2]:
            cur_dict  = self._cur_obs[f"player_{idx}_crown_towers"]
            prev_dict = prev_obs[f"player_{idx}_crown_towers"]

            for tower_str in cur_dict.keys():
                cur_health  = float(cur_dict[tower_str]["health"])
                prev_health = float(prev_dict[tower_str]["health"])
                delta = cur_health - prev_health  # negative when damage dealt

                # HP delta (shaping)
                if idx == 1:
                    player_1_reward += delta * self.tower_damage_reward_scale
                    player_2_reward -= delta * self.tower_damage_reward_scale
                else:
                    player_1_reward -= delta * self.tower_damage_reward_scale
                    player_2_reward += delta * self.tower_damage_reward_scale

                # Tower destruction bonus
                if prev_health > 0 and cur_health <= 0:
                    if idx == 1:  # P1's tower destroyed → bad for P1
                        player_1_reward -= self.tower_distruction_reward
                        player_2_reward += self.tower_distruction_reward
                    else:         # P2's tower destroyed → good for P1
                        player_1_reward += self.tower_distruction_reward
                        player_2_reward -= self.tower_distruction_reward

        # 3. Game outcome  (covers termination AND 5:00 tiebreaker truncation)
        if terminated or truncated:
            winner = self.arena.winner  # set by Arena.update(); 1, 2, or None
            if winner == 1:
                player_1_reward += self.winning_reward
                player_2_reward -= self.winning_reward
            elif winner == 2:
                player_1_reward -= self.winning_reward
                player_2_reward += self.winning_reward
            # winner == None → true draw, no win bonus

        return player_1_reward, player_2_reward


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def _render_frame(self):
        screen_w = self.arena.width  * self.arena.tile_size
        screen_h = self.arena.height * self.arena.tile_size

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_w, screen_h))
            else:  
                # rgb_array: use an offscreen surface, no display needed
                self.screen = pygame.Surface((screen_w, screen_h))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        self.arena.render(self.screen)

        if self.render_mode == "human":
            self.arena.render(self.screen)
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])  # TODO: do i even need this!?
        else:  # rgb_array
            rgb_array = pygame.surfarray.array3d(self.screen)
            rgb_array = rgb_array.transpose(1, 0, 2)  # pygame is (w,h,c), numpy expects (h,w,c)
            return rgb_array


    def close(self):
        if self.screen is not None:
            if self.render_mode == "human":
                pygame.display.quit()
            pygame.quit()
            self.screen = None


if __name__ == "__main__":
    # quick test

    # env = ClashRoyaleEnv()  # Works but the latter has some useful checks
    env = gym.make("ClashRoyaleEnv-v0", render_mode="rgb_array")

    env = CRFlattenNormWrapper(env)

    env = gym.wrappers.RecordVideo(
        env,
        video_folder="./videos",
        name_prefix=f"debug",
        episode_trigger=lambda _: True,
    )
    
    state, _ = env.reset()
    done = False

    player_1_spawn_cooldown = 0.0
    player_2_spawn_cooldown = 0.0

    player_1_spawn = {
        "player_1_skip": 0,
        "player_1_card_idx": 0,
        "player_1_card_position": np.array([9, 32-10]),

        "player_2_skip": 1,
        "player_2_card_idx": 0,
        "player_2_card_position": np.array([9, 10]),
    }

    player_2_spawn = {
        "player_1_skip": 1,
        "player_1_card_idx": 0,
        "player_1_card_position": np.array([9, 32-10]),

        "player_2_skip": 0,
        "player_2_card_idx": 0,
        "player_2_card_position": np.array([9, 10]),
    }

    try:
        while not done:
            if player_1_spawn_cooldown > 0.25:
                player_1_spawn_cooldown = 0.0
                action = player_1_spawn
            elif player_2_spawn_cooldown > 0.2:
                player_2_spawn_cooldown = 0.0
                action = player_2_spawn
            else:
                action = env.action_space.sample()
                action["player_1_skip"] = 1
                action["player_2_skip"] = 1

            next_state, reward, termination, truncated, _ = env.step(action)

            player_1_spawn_cooldown += 1/100
            # player_2_spawn_cooldown += 1/100

            print("-----")
            print("action", action)
            print("next_state", next_state)

            done = termination or truncated
    except Exception as e:
        print(e)

    env.close()
