from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable


CHECKPOINT_RE = re.compile(r"checkpoint_(?P<idx>\d+)(?:_(?P<elo>-?\d+))?\.pt$")

ARCHITECTURE_ALIASES = {
    "deep_sets": "deep_sets",
    "deepsets": "deep_sets",
    "transformer": "transformer",
    "legacy_deep_sets_attention": "legacy_deep_sets_attention",
    "attention": "legacy_deep_sets_attention",
}

ARCHITECTURE_LABELS = {
    "deep_sets": "Deep Sets",
    "legacy_deep_sets_attention": "Deep Sets attention",
    "transformer": "Transformer",
}

ARCHITECTURE_COLORS = {
    "legacy_deep_sets_attention": "#1f77b4",
    "deep_sets": "#2ca02c",
    "transformer": "#d62728",
}


@dataclass(frozen=True)
class CheckpointEntry:
    id: str
    selector: str
    run_name: str
    architecture: str
    architecture_label: str
    architecture_forced: bool
    checkpoint_index: int | None
    source_elo: int | None
    path: str


@dataclass(frozen=True)
class ArenaGame:
    game_index: int
    pair_index: int
    pair_game_index: int
    player_1: str
    player_2: str


def normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def parse_checkpoint_name(path: Path) -> tuple[int | None, int | None]:
    match = CHECKPOINT_RE.fullmatch(path.name)
    if not match:
        return None, None
    idx = int(match.group("idx"))
    elo = int(match.group("elo")) if match.group("elo") is not None else None
    return idx, elo


def infer_architecture(run_dir: Path, overrides: dict[str, str]) -> tuple[str, bool]:
    run_key = normalise_name(run_dir.name)
    for selector, architecture in overrides.items():
        if normalise_name(selector) in run_key:
            return architecture, True

    lowered = run_dir.name.lower()
    if "transformer" in lowered:
        return "transformer", False
    if "attention" in lowered:
        return "deep_sets", False
    if "deep" in lowered or "deepsets" in lowered:
        return "deep_sets", False
    return "deep_sets", False


def parse_architecture_overrides(values: Iterable[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Architecture override must be selector=architecture, got {value!r}")
        selector, architecture = value.split("=", 1)
        architecture = ARCHITECTURE_ALIASES.get(architecture.strip(), architecture.strip())
        if architecture not in ARCHITECTURE_LABELS:
            raise ValueError(f"Unknown architecture override {architecture!r}")
        overrides[selector.strip()] = architecture
    return overrides


def selector_matches(selector: str, run_dir: Path) -> bool:
    selector_key = normalise_name(selector)
    run_key = normalise_name(run_dir.name)
    return selector_key in run_key


def discover_checkpoints(
    root: Path,
    checkpoint_roots: list[str],
    selectors: list[str],
    architecture_overrides: dict[str, str],
) -> dict[str, list[CheckpointEntry]]:
    selected: dict[str, list[CheckpointEntry]] = {selector: [] for selector in selectors}

    selector_hits_by_checkpoint: dict[str, set[str]] = {}

    for checkpoint_root in checkpoint_roots:
        base = root / checkpoint_root
        if not base.exists():
            continue

        for run_dir in sorted(path for path in base.iterdir() if path.is_dir()):
            matched_selectors = [selector for selector in selectors if selector_matches(selector, run_dir)]
            if len(matched_selectors) > 1:
                raise RuntimeError(
                    f"Run directory {run_dir.name!r} matches multiple selectors {matched_selectors}; "
                    "use non-overlapping selectors for a self-safe arena."
                )
            if not matched_selectors:
                continue

            checkpoint_dir = run_dir / "checkpoints"
            if not checkpoint_dir.exists():
                continue

            architecture, architecture_forced = infer_architecture(run_dir, architecture_overrides)
            label = ARCHITECTURE_LABELS.get(architecture, architecture)
            for checkpoint_path in sorted(checkpoint_dir.glob("checkpoint_*.pt")):
                checkpoint_index, source_elo = parse_checkpoint_name(checkpoint_path)
                if source_elo is None:
                    continue
                rel_path = checkpoint_path.relative_to(root).as_posix()
                for selector in matched_selectors:
                    selector_hits_by_checkpoint.setdefault(rel_path, set()).add(selector)
                    if len(selector_hits_by_checkpoint[rel_path]) > 1:
                        raise RuntimeError(
                            f"Checkpoint {rel_path!r} matched multiple selectors "
                            f"{sorted(selector_hits_by_checkpoint[rel_path])}; use non-overlapping selectors."
                        )
                    selected[selector].append(
                        CheckpointEntry(
                            id=rel_path,
                            selector=selector,
                            run_name=run_dir.name,
                            architecture=architecture,
                            architecture_label=label,
                            architecture_forced=architecture_forced,
                            checkpoint_index=checkpoint_index,
                            source_elo=source_elo,
                            path=str(checkpoint_path),
                        )
                    )

    return selected


def top_checkpoints_by_selector(
    discovered: dict[str, list[CheckpointEntry]],
    top_n: int,
) -> list[CheckpointEntry]:
    chosen: list[CheckpointEntry] = []
    seen_ids: set[str] = set()

    for selector, checkpoints in discovered.items():
        elo_checkpoints = [item for item in checkpoints if item.source_elo is not None]
        ranked = sorted(
            elo_checkpoints,
            key=lambda item: (
                item.source_elo,
                item.checkpoint_index if item.checkpoint_index is not None else -1,
                item.id,
            ),
            reverse=True,
        )
        top = ranked[:top_n]
        if len(top) < top_n:
            raise RuntimeError(
                f"Selector {selector!r} only matched {len(top)} Elo-suffixed checkpoints; "
                f"requested top {top_n}."
            )
        for item in top:
            if item.id in seen_ids:
                raise RuntimeError(f"Checkpoint {item.id!r} was selected more than once; use non-overlapping selectors/roots.")
            chosen.append(item)
            seen_ids.add(item.id)

    expected_count = len(discovered) * top_n
    if len(chosen) != expected_count:
        raise RuntimeError(f"Selected {len(chosen)} checkpoints, expected {expected_count}.")

    return chosen


def build_round_robin_schedule(checkpoints: list[CheckpointEntry], k: int) -> list[ArenaGame]:
    games: list[ArenaGame] = []
    for pair_index, (left, right) in enumerate(combinations(sorted(checkpoints, key=lambda item: item.id), 2)):
        for pair_game_index in range(k):
            if (pair_index + pair_game_index) % 2 == 0:
                player_1, player_2 = left.id, right.id
            else:
                player_1, player_2 = right.id, left.id
            games.append(
                ArenaGame(
                    game_index=len(games),
                    pair_index=pair_index,
                    pair_game_index=pair_game_index,
                    player_1=player_1,
                    player_2=player_2,
                )
            )
    return games


def expected_score(rating_a: float, rating_b: float, scale: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def update_elos(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k_factor: float,
    scale: float,
) -> tuple[float, float]:
    expected_a = expected_score(rating_a, rating_b, scale)
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a
    return (
        rating_a + k_factor * (score_a - expected_a),
        rating_b + k_factor * (score_b - expected_b),
    )


def deterministic_simulated_score(player_1: CheckpointEntry, player_2: CheckpointEntry, game: ArenaGame) -> float:
    strength_1 = player_1.source_elo if player_1.source_elo is not None else 1200
    strength_2 = player_2.source_elo if player_2.source_elo is not None else 1200
    p1_win_prob = expected_score(strength_1, strength_2, 400.0)
    digest = hashlib.sha256(f"{game.game_index}:{player_1.id}:{player_2.id}".encode()).digest()
    roll = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    draw_margin = 0.03
    if abs(roll - p1_win_prob) <= draw_margin:
        return 0.5
    return 1.0 if roll < p1_win_prob else 0.0


def scalar_int(value) -> int:
    if hasattr(value, "detach"):
        return int(value.detach().cpu().item())
    return int(value)


def joined_actions(action_1, action_2, arena_width: int, arena_height: int) -> dict:
    def policy_position_to_xy(action):
        pos_idx = scalar_int(action["position"])
        return pos_idx % arena_width, pos_idx // arena_width

    def rotate_xy_180(x: int, y: int):
        return arena_width - 1 - x, arena_height - 1 - y

    p1_x, p1_y = policy_position_to_xy(action_1)
    p2_x, p2_y = rotate_xy_180(*policy_position_to_xy(action_2))
    return {
        "player_1_skip": scalar_int(action_1["skip"]),
        "player_1_card_idx": scalar_int(action_1["deck_idx"]),
        "player_1_card_position": (p1_x, p1_y),
        "player_2_skip": scalar_int(action_2["skip"]),
        "player_2_card_idx": scalar_int(action_2["deck_idx"]),
        "player_2_card_position": (p2_x, p2_y),
    }


def _shape(state_dict: dict, key: str) -> tuple[int, ...] | None:
    value = state_dict.get(key)
    if hasattr(value, "shape"):
        return tuple(int(dim) for dim in value.shape)
    return None


def _first_shape(state_dict: dict, keys: Iterable[str]) -> tuple[int, ...] | None:
    for key in keys:
        shape = _shape(state_dict, key)
        if shape is not None:
            return shape
    return None


def infer_state_dict_architecture(state_dict: dict) -> str | None:
    keys = set(state_dict.keys())
    if any("transformer_encoder" in key for key in keys):
        return "transformer"
    if "entity_attention.in_proj_weight" in keys:
        return "legacy_deep_sets_attention"
    if any("entity_attention" in key for key in keys):
        return "deep_sets"
    if any("entity_encoder" in key for key in keys):
        return "deep_sets"
    return None


def infer_policy_config(state_dict: dict, checkpoint: CheckpointEntry) -> dict:
    keys = set(state_dict.keys())
    inferred_architecture = infer_state_dict_architecture(state_dict)
    architecture = checkpoint.architecture if checkpoint.architecture_forced else (inferred_architecture or checkpoint.architecture)

    trunk_shape = _shape(state_dict, "critic_head.0.weight")
    trunk_out_ch = trunk_shape[1] if trunk_shape and len(trunk_shape) == 2 else 128

    pointer_decoder = "skip_token" in keys or "pointer_query.weight" in keys
    if "actor_skip_net.0.weight" in keys or "actor_deck_idx_net.0.weight" in keys:
        pointer_decoder = False

    config = {
        "architecture": architecture,
        "entity_encoder_mid_ch": 64,
        "entity_encoder_out_ch": 32,
        "trunk_out_ch": trunk_out_ch,
        "d_model": 64,
        "num_layers": 2,
        "activation_fn": "tanh",
        "disjoint_actor_critic": any(key.startswith("critic_") for key in keys) and any(key.startswith("actor_") for key in keys),
        "use_cnn_position_decoder": any(
            key.startswith("actor_position_net.") and len(getattr(value, "shape", ())) == 4
            for key, value in state_dict.items()
        ),
        "use_layer_init": True,
        "use_learned_temperature": any(key.startswith("log_temp") for key in keys),
        "append_deck_info_to_position_head_input": True,
        "use_attention_over_entities": any("entity_attention" in key for key in keys) or architecture == "legacy_deep_sets_attention",
        "use_pointer_decoder": pointer_decoder,
    }

    if architecture == "transformer":
        d_model_shape = _first_shape(
            state_dict,
            [
                "critic_transformer_encoder.entity_proj.weight",
                "actor_transformer_encoder.entity_proj.weight",
                "shared_transformer_encoder.entity_proj.weight",
            ],
        )
        if d_model_shape:
            config["d_model"] = d_model_shape[0]
        skip_shape = _shape(state_dict, "skip_token")
        if skip_shape and len(skip_shape) == 1:
            config["d_model"] = skip_shape[0]
        pointer_shape = _shape(state_dict, "pointer_query.weight")
        if pointer_shape and len(pointer_shape) == 2:
            config["d_model"] = pointer_shape[0]
        deck_shape = _shape(state_dict, "actor_deck_idx_net.0.weight")
        if deck_shape and len(deck_shape) == 2:
            config["d_model"] = deck_shape[1]

        layer_ids: set[int] = set()
        for key in keys:
            match = re.search(r"transformer_encoder\.layers\.(\d+)\.", key)
            if match:
                layer_ids.add(int(match.group(1)))
        if layer_ids:
            config["num_layers"] = max(layer_ids) + 1
    else:
        mid_shape = _first_shape(
            state_dict,
            [
                "actor_entity_encoder.0.weight",
                "critic_entity_encoder.0.weight",
                "shared_entity_encoder.0.weight",
            ],
        )
        if mid_shape:
            config["entity_encoder_mid_ch"] = mid_shape[0]

        out_shape = _first_shape(
            state_dict,
            [
                "actor_entity_encoder.6.weight",
                "critic_entity_encoder.6.weight",
                "shared_entity_encoder.6.weight",
            ],
        )
        if out_shape:
            config["entity_encoder_out_ch"] = out_shape[0]

    pos_linear_shape = _shape(state_dict, "actor_position_net.0.weight")
    pos_conv_shape = _shape(state_dict, "actor_position_net.1.weight")
    position_input_ch = None
    if pos_linear_shape and len(pos_linear_shape) == 2:
        position_input_ch = pos_linear_shape[1]
    elif pos_conv_shape and len(pos_conv_shape) == 4:
        position_input_ch = pos_conv_shape[0]

    if position_input_ch is not None:
        extra_ch = config["d_model"] if architecture == "transformer" else config["entity_encoder_out_ch"]
        config["append_deck_info_to_position_head_input"] = position_input_ch > trunk_out_ch
        if config["append_deck_info_to_position_head_input"] and position_input_ch != trunk_out_ch + extra_ch:
            # Preserve load compatibility for unusual checkpoints by trusting the observed head width.
            inferred_extra = max(0, position_input_ch - trunk_out_ch)
            if architecture == "transformer":
                config["d_model"] = inferred_extra or config["d_model"]
            elif architecture == "deep_sets":
                config["entity_encoder_out_ch"] = inferred_extra or config["entity_encoder_out_ch"]

    return config


class RealMatchRunner:
    def __init__(self, checkpoints: list[CheckpointEntry], max_steps_per_game: int | None):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

        import gymnasium as gym
        import pygame
        import torch as t

        import rl.env.cr_gym_env  # noqa: F401
        from rl.env.cr_flatten_norm_wrapper import CRFlattenNormWrapper
        from rl.networks import make_network
        from web_play.game_session import split_observations
        from web_play.legacy_networks import LegacyAttentionDeepSetsActorCritic

        self.torch = t
        self.make_network = make_network
        self.split_observations = split_observations
        self.legacy_network_cls = LegacyAttentionDeepSetsActorCritic
        self.max_steps_per_game = max_steps_per_game

        pygame.init()
        pygame.font.init()
        pygame.display.set_mode((1, 1))

        raw_env = gym.make("ClashRoyaleEnv-v0")
        self.env = CRFlattenNormWrapper(raw_env)
        self.arena = self.env.unwrapped.arena
        self.max_num_objects = self.arena.max_num_objects
        self.policies = {entry.id: self._load_policy(entry) for entry in checkpoints}

    def _invalid_position_mask(self):
        import numpy as np

        arena = self.arena
        scale = arena.tile_size
        tiled = np.where(arena.cell_occupancy == 1, 1, 0)[scale // 2 :: scale, scale // 2 :: scale]
        mask = tiled.astype(bool).T
        mask[arena.height // 2 :, :] = True
        return self.torch.tensor(mask).flatten()

    def _network_kwargs(self, config: dict) -> dict:
        return {
            "entity_encoder_in_ch": self.env.flat_card_space.shape[0],
            "entity_encoder_mid_ch": config["entity_encoder_mid_ch"],
            "entity_encoder_out_ch": config["entity_encoder_out_ch"],
            "trunk_extra_in_ch": 2,
            "trunk_mid_ch": 128,
            "trunk_out_ch": config["trunk_out_ch"],
            "d_model": config["d_model"],
            "num_layers": config["num_layers"],
            "activation_fn": config["activation_fn"],
            "disjoint_actor_critic": config["disjoint_actor_critic"],
            "use_cnn_position_decoder": config["use_cnn_position_decoder"],
            "use_layer_init": config["use_layer_init"],
            "use_learned_temperature": config["use_learned_temperature"],
            "append_deck_info_to_position_head_input": config["append_deck_info_to_position_head_input"],
            "use_attention_over_entities": config["use_attention_over_entities"],
            "use_pointer_decoder": config["use_pointer_decoder"],
            "num_cards_in_deck": self.env.unwrapped.NUM_CARDS_IN_HAND,
            "num_cards_in_hand": self.env.unwrapped.NUM_CARDS_IN_HAND,
            "max_num_cards": self.max_num_objects,
            "position_space_width": self.arena.width,
            "position_space_height": self.arena.height,
            "invalid_position_mask": self._invalid_position_mask(),
            "max_elixirs": 10,
            "deploy_cost_idx": self.env.flattened_card_space_indices["deploy_cost"][0],
        }

    def _load_policy(self, checkpoint: CheckpointEntry):
        state_dict = self.torch.load(checkpoint.path, map_location="cpu", weights_only=True)
        config = infer_policy_config(state_dict, checkpoint)
        kwargs = self._network_kwargs(config)
        if config["architecture"] == "legacy_deep_sets_attention":
            kwargs.pop("num_cards_in_deck", None)
            policy = self.legacy_network_cls(**kwargs)
        else:
            policy = self.make_network(config["architecture"], **kwargs)

        policy.load_state_dict(state_dict)
        policy.eval()
        return policy

    def _seed_everything(self, seed: int) -> None:
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)

    def play(self, game: ArenaGame, seed: int) -> tuple[float, dict]:
        self._seed_everything(seed)
        obs, _ = self.env.reset(seed=seed)
        self.arena = self.env.unwrapped.arena
        player_1_policy = self.policies[game.player_1]
        player_2_policy = self.policies[game.player_2]
        total_reward = [0.0, 0.0]
        info: dict = {}

        with self.torch.no_grad():
            for step in range(self.max_steps_per_game or 10**9):
                obs_1, obs_2 = self.split_observations(
                    obs,
                    self.env,
                    self.arena,
                    self.max_num_objects,
                )
                action_1, _, _, _ = player_1_policy.get_action_and_value(obs_1)
                action_2, _, _, _ = player_2_policy.get_action_and_value(obs_2)
                obs, reward, terminated, truncated, info = self.env.step(
                    joined_actions(action_1, action_2, self.arena.width, self.arena.height)
                )
                total_reward[0] += float(reward[0])
                total_reward[1] += float(reward[1])
                if terminated or truncated:
                    break
            else:
                info = {"episode": {"winner": None, "forced_max_steps": self.max_steps_per_game}}

        episode = info.get("episode", {})
        winner = episode.get("winner")
        if winner == 1:
            score = 1.0
        elif winner == 2:
            score = 0.0
        elif total_reward[0] > total_reward[1]:
            score = 1.0
        elif total_reward[0] < total_reward[1]:
            score = 0.0
        else:
            score = 0.5

        return score, {
            "winner": winner,
            "reward_p1": total_reward[0],
            "reward_p2": total_reward[1],
            "episode": episode,
        }


def blend_with_white(hex_color: str, amount: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = round(r + (255 - r) * amount)
    g = round(g + (255 - g) * amount)
    b = round(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def checkpoint_colors(checkpoints: list[CheckpointEntry]) -> dict[str, str]:
    by_arch: dict[str, list[CheckpointEntry]] = {}
    for checkpoint in sorted(checkpoints, key=lambda item: item.id):
        by_arch.setdefault(checkpoint.architecture, []).append(checkpoint)

    colors: dict[str, str] = {}
    for architecture, items in by_arch.items():
        base = ARCHITECTURE_COLORS.get(architecture, "#7f7f7f")
        for index, checkpoint in enumerate(items):
            amount = 0.0 if len(items) == 1 else 0.55 * index / max(1, len(items) - 1)
            colors[checkpoint.id] = blend_with_white(base, amount)
    return colors


def short_label(checkpoint: CheckpointEntry) -> str:
    idx = "?" if checkpoint.checkpoint_index is None else str(checkpoint.checkpoint_index)
    source_elo = "no-elo" if checkpoint.source_elo is None else str(checkpoint.source_elo)
    return f"{checkpoint.selector}:ckpt{idx}:{source_elo}"


def write_outputs(
    output_dir: Path,
    checkpoints: list[CheckpointEntry],
    games: list[dict],
    history_rows: list[dict],
    final_rows: list[dict],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "checkpoints": [asdict(item) for item in checkpoints],
                "num_games": len(games),
            },
            f,
            indent=2,
        )

    write_csv(output_dir / "games.csv", games)
    write_csv(output_dir / "elo_history.csv", history_rows)
    write_csv(output_dir / "final_elos.csv", final_rows)
    write_plot(output_dir / "elo_history.html", checkpoints, history_rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, checkpoints: list[CheckpointEntry], history_rows: list[dict]) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        write_svg_plot(path, checkpoints, history_rows)
        return

    entries = {checkpoint.id: checkpoint for checkpoint in checkpoints}
    colors = checkpoint_colors(checkpoints)
    fig = go.Figure()
    for checkpoint in sorted(checkpoints, key=lambda item: (item.architecture, item.id)):
        rows = [row for row in history_rows if row["checkpoint_id"] == checkpoint.id]
        fig.add_trace(
            go.Scatter(
                x=[row["game_index"] for row in rows],
                y=[row["elo"] for row in rows],
                mode="lines+markers",
                name=f"{short_label(entries[checkpoint.id])} ({checkpoint.architecture_label})",
                line={"color": colors[checkpoint.id], "width": 2.5},
                marker={"color": colors[checkpoint.id], "size": 6},
                hovertemplate="%{fullData.name}<br>game=%{x}<br>elo=%{y:.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Checkpoint Arena Elo History",
        xaxis_title="Game index",
        yaxis_title="Arena Elo",
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Checkpoint",
    )
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


def write_svg_plot(path: Path, checkpoints: list[CheckpointEntry], history_rows: list[dict]) -> None:
    width, height, margin = 1100, 650, 72
    colors = checkpoint_colors(checkpoints)
    xs = [float(row["game_index"]) for row in history_rows]
    ys = [float(row["elo"]) for row in history_rows]
    if not xs or not ys:
        path.write_text("<html><body><svg></svg></body></html>", encoding="utf-8")
        return

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0
    if math.isclose(min_y, max_y):
        min_y -= 1.0
        max_y += 1.0

    def xscale(x: float) -> float:
        return margin + (x - min_x) / (max_x - min_x) * (width - 2 * margin)

    def yscale(y: float) -> float:
        return height - margin - (y - min_y) / (max_y - min_y) * (height - 2 * margin)

    lines = [
        "<html><body>",
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#333'/>",
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='#333'/>",
        f"<text x='{width / 2}' y='36' text-anchor='middle' font-family='sans-serif' font-size='22'>Checkpoint Arena Elo History</text>",
    ]
    for checkpoint in sorted(checkpoints, key=lambda item: (item.architecture, item.id)):
        rows = [row for row in history_rows if row["checkpoint_id"] == checkpoint.id]
        points = " ".join(
            f"{xscale(float(row['game_index'])):.1f},{yscale(float(row['elo'])):.1f}"
            for row in rows
        )
        label = html.escape(short_label(checkpoint))
        color = colors[checkpoint.id]
        lines.append(f"<polyline fill='none' stroke='{color}' stroke-width='2.5' points='{points}'/>")
        if rows:
            last = rows[-1]
            lines.append(
                f"<text x='{xscale(float(last['game_index'])) + 6:.1f}' y='{yscale(float(last['elo'])):.1f}' "
                f"font-family='sans-serif' font-size='12' fill='{color}'>{label}</text>"
            )
    lines.extend(["</svg>", "</body></html>"])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_arena(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict], list[CheckpointEntry]]:
    root = Path(args.root).resolve()
    if len({normalise_name(selector) for selector in args.runs}) != len(args.runs):
        raise RuntimeError("Run selectors overlap after normalization; make --runs selectors unique.")

    architecture_overrides = parse_architecture_overrides(args.architecture_override)
    discovered = discover_checkpoints(root, args.checkpoint_roots, args.runs, architecture_overrides)
    checkpoints = top_checkpoints_by_selector(discovered, args.top_n)

    if len(checkpoints) < 2:
        raise RuntimeError("At least two checkpoints are required for an arena.")

    games = build_round_robin_schedule(checkpoints, args.k)
    expected_games = args.k * math.comb(len(checkpoints), 2)
    if len(games) != expected_games:
        raise AssertionError(f"Schedule has {len(games)} games, expected {expected_games}")

    checkpoint_by_id = {checkpoint.id: checkpoint for checkpoint in checkpoints}
    ratings = {checkpoint.id: float(args.initial_elo) for checkpoint in checkpoints}
    history_rows: list[dict] = []
    game_rows: list[dict] = []

    for checkpoint in checkpoints:
        history_rows.append(
            {
                "game_index": 0,
                "checkpoint_id": checkpoint.id,
                "label": short_label(checkpoint),
                "architecture": checkpoint.architecture,
                "elo": ratings[checkpoint.id],
            }
        )

    runner = None
    if args.mode == "real":
        runner = RealMatchRunner(checkpoints, args.max_steps_per_game)

    for game in games:
        player_1 = checkpoint_by_id[game.player_1]
        player_2 = checkpoint_by_id[game.player_2]
        before_p1 = ratings[player_1.id]
        before_p2 = ratings[player_2.id]

        if args.mode == "simulate":
            score_p1 = deterministic_simulated_score(player_1, player_2, game)
            detail = {"winner": 1 if score_p1 == 1.0 else 2 if score_p1 == 0.0 else None}
        else:
            assert runner is not None
            score_p1, detail = runner.play(game, seed=args.seed + game.game_index)

        after_p1, after_p2 = update_elos(
            before_p1,
            before_p2,
            score_p1,
            args.elo_k_factor,
            args.elo_scale,
        )
        ratings[player_1.id] = after_p1
        ratings[player_2.id] = after_p2

        game_rows.append(
            {
                "game_index": game.game_index,
                "pair_index": game.pair_index,
                "pair_game_index": game.pair_game_index,
                "player_1": player_1.id,
                "player_2": player_2.id,
                "score_player_1": score_p1,
                "player_1_elo_before": before_p1,
                "player_2_elo_before": before_p2,
                "player_1_elo_after": after_p1,
                "player_2_elo_after": after_p2,
                "winner": detail.get("winner"),
                "reward_p1": detail.get("reward_p1"),
                "reward_p2": detail.get("reward_p2"),
            }
        )

        for checkpoint in checkpoints:
            history_rows.append(
                {
                    "game_index": game.game_index + 1,
                    "checkpoint_id": checkpoint.id,
                    "label": short_label(checkpoint),
                    "architecture": checkpoint.architecture,
                    "elo": ratings[checkpoint.id],
                }
            )

    final_rows = [
        {
            "rank": rank,
            "checkpoint_id": checkpoint.id,
            "label": short_label(checkpoint),
            "selector": checkpoint.selector,
            "run_name": checkpoint.run_name,
            "architecture": checkpoint.architecture,
            "source_elo": checkpoint.source_elo,
            "arena_elo": ratings[checkpoint.id],
            "path": checkpoint.path,
        }
        for rank, checkpoint in enumerate(
            sorted(checkpoints, key=lambda item: ratings[item.id], reverse=True),
            start=1,
        )
    ]
    return game_rows, history_rows, final_rows, checkpoints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a deterministic checkpoint-vs-checkpoint Elo arena.")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--checkpoint-roots",
        nargs="+",
        default=["runs"],
        help="Directories under --root that contain run folders.",
    )
    parser.add_argument("--runs", nargs="+", default=["rerun30", "rerun31"], help="Run-name selectors.")
    parser.add_argument("--top-n", type=int, default=5, help="Top Elo checkpoints per run selector.")
    parser.add_argument("--k", type=int, default=3, help="Games per unordered checkpoint pair.")
    parser.add_argument("--mode", choices=["real", "simulate"], default="real")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-elo", type=float, default=1200.0)
    parser.add_argument("--elo-k-factor", type=float, default=32.0)
    parser.add_argument("--elo-scale", type=float, default=400.0)
    parser.add_argument("--max-steps-per-game", type=int, default=None)
    parser.add_argument(
        "--architecture-override",
        action="append",
        default=[],
        help="Override inferred architecture for matching runs, e.g. rerun30=legacy_deep_sets_attention.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write CSV/JSON/HTML outputs. Defaults to checkpoint_arena/runs/<timestamp>.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be >= 1")
    if args.k < 1:
        raise ValueError("--k must be >= 1")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir or f"checkpoint_arena/runs/{timestamp}").resolve()
    root = Path(args.root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    game_rows, history_rows, final_rows, checkpoints = run_arena(args)
    write_outputs(output_dir, checkpoints, game_rows, history_rows, final_rows, args)

    print(f"Wrote arena outputs to {output_dir}")
    print(f"Played {len(game_rows)} games across {len(checkpoints)} checkpoints.")
    print("Final Elo:")
    for row in final_rows:
        print(f"  {row['rank']:>2}. {row['arena_elo']:.1f}  {row['label']}  {row['architecture']}")


if __name__ == "__main__":
    main()
