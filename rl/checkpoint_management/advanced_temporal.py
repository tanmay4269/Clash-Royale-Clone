import os
from collections import deque
import numpy as np
import torch as t


class AdvancedTemporal_CheckpointManagement:
    def __init__(
        self, 
        checkpoint_dir, 
        
        # Loading
        loading_latest_ratio=0.5,
        loading_delta_window=0.2,

        # Storage
        min_games_before_checkpointing=100,
        score_queue_size=100,
        avg_score_threshold=0.55,
    ):
        self.checkpoint_dir = checkpoint_dir

        # Loading
        self.loading_latest_ratio = loading_latest_ratio
        self.loading_delta_window = loading_delta_window

        # Storage
        self.min_games_before_checkpointing = min_games_before_checkpointing
        self.avg_score_threshold = avg_score_threshold
        self._score_queue_maxlen = score_queue_size

        self.checkpoint_counter = 0
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.score_queue = deque(maxlen=score_queue_size)


    def load(self, net, current_elo=None, active_checkpoints=None):
        if self.checkpoint_counter == 0:
            return 1200, None

        checkpoint_min = int(self.checkpoint_counter * (1 - self.loading_delta_window))
        checkpoint_max = self.checkpoint_counter - 1

        if np.random.rand() < self.loading_latest_ratio:
            latest_idx = self.checkpoint_counter - 1
            if active_checkpoints is not None and latest_idx in active_checkpoints:
                allowed_indices = [idx for idx in range(self.checkpoint_counter) if idx not in active_checkpoints]
                checkpoint_sample = np.random.choice(allowed_indices) if allowed_indices else latest_idx
            else:
                checkpoint_sample = latest_idx
        else:
            if checkpoint_max <= checkpoint_min:
                checkpoint_sample = self.checkpoint_counter - 1
            else:
                allowed_indices = [idx for idx in range(checkpoint_min, checkpoint_max) if idx not in (active_checkpoints or [])]
                if not allowed_indices:
                    allowed_indices = list(range(checkpoint_min, checkpoint_max))
                checkpoint_sample = np.random.choice(allowed_indices)

        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{checkpoint_sample}.pt")
        net.load_state_dict(t.load(checkpoint_path, map_location=None if t.cuda.is_available() else 'cpu', weights_only=True))
        return 1200, checkpoint_sample


    def update(self, net, returns):
        score = 0
        if returns[0] > returns[1]:
            score = 1
        elif returns[0] == returns[1]:
            score = 0.5
        self.score_queue.append(score)

        if len(self.score_queue) < self.min_games_before_checkpointing:
            return

        if np.mean(self.score_queue) < self.avg_score_threshold:
            return

        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{self.checkpoint_counter}.pt")
        t.save(net.state_dict(), checkpoint_path)

        self.checkpoint_counter += 1
        self.score_queue.clear()


    def get_state(self):
        """Return a dict that fully captures this manager's internal state."""
        return {
            "checkpoint_counter": self.checkpoint_counter,
            "score_queue": list(self.score_queue),
        }


    def load_state(self, state: dict):
        """Restore internal state from a previously saved dict."""
        self.checkpoint_counter = state["checkpoint_counter"]
        self.score_queue = deque(state["score_queue"], maxlen=self._score_queue_maxlen)
