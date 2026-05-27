import os
from collections import deque
import numpy as np
import torch as t


class AdvancedEloBased_CheckpointManagement:
    def __init__(
        self, 
        checkpoint_dir, 
        elo_cfg,
        
        # Loading
        loading_latest_ratio=0.5,

        # Storage
        min_games_before_checkpointing=100,
        score_queue_size=100,
        avg_score_threshold=0.55,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.elo_cfg = elo_cfg

        # Loading
        self.loading_latest_ratio = loading_latest_ratio

        # Storage
        self.min_games_before_checkpointing = min_games_before_checkpointing
        self.avg_score_threshold = avg_score_threshold
        self._score_queue_maxlen = score_queue_size

        self.checkpoint_counter = 0
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.score_queue = deque(maxlen=score_queue_size)

        self.stored_by_idx = {}  # checkpoint_idx -> (elo, filename)
        self.stored_by_elo = {}  # elo -> [checkpoint_indices]


    def load(self, net, current_elo, active_checkpoints=None):
        current_elo = int(current_elo)
        
        if self.checkpoint_counter == 0:
            return self.elo_cfg.initial_rating, None

        allowed_indices = [idx for idx in range(self.checkpoint_counter) if idx not in (active_checkpoints or [])]
        if not allowed_indices:
            allowed_indices = list(range(self.checkpoint_counter))

        if np.random.rand() < self.loading_latest_ratio:
            checkpoint_sample = max(allowed_indices)
            opponent_elo, filename = self.stored_by_idx[checkpoint_sample]
            checkpoint_path = os.path.join(self.checkpoint_dir, filename)
            net.load_state_dict(t.load(checkpoint_path, map_location=next(net.parameters()).device, weights_only=True))
            return opponent_elo, checkpoint_sample

        allowed_by_elo = {}
        for idx in allowed_indices:
            elo = self.stored_by_idx[idx][0]
            if elo not in allowed_by_elo:
                allowed_by_elo[elo] = []
            allowed_by_elo[elo].append(idx)

        opponent_elos = list(allowed_by_elo.keys())
        E_As = []
        for opponent_elo in opponent_elos:
            E_As.append(
                1 / (1 + 10 ** ((opponent_elo - current_elo) / self.elo_cfg.scale))
            )

        probs = 0.5 - abs(0.5 - np.array(E_As))  # peak at 0.5, falls off to 0 on either extremes
        
        prob_sum = np.sum(probs)
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
            probs = np.ones_like(probs) / len(probs)
            
        chosen_elo = np.random.choice(opponent_elos, p=probs)
        checkpoint_indices = allowed_by_elo[chosen_elo]
        checkpoint_sample = np.random.choice(checkpoint_indices)

        opponent_elo, filename = self.stored_by_idx[checkpoint_sample]
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        net.load_state_dict(t.load(checkpoint_path, map_location=next(net.parameters()).device, weights_only=True))
        return opponent_elo, checkpoint_sample


    def update(self, net, score, current_elo):
        current_elo = int(current_elo)

        self.score_queue.append(score)

        if len(self.score_queue) < self.min_games_before_checkpointing:
            return

        if np.mean(self.score_queue) < self.avg_score_threshold:
            return

        filename = f"checkpoint_{self.checkpoint_counter}_{current_elo}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        t.save(net.state_dict(), checkpoint_path)

        self.stored_by_idx[self.checkpoint_counter] = (current_elo, filename)
        if current_elo not in self.stored_by_elo:
            self.stored_by_elo[current_elo] = []
        self.stored_by_elo[current_elo].append(self.checkpoint_counter)

        print(f"Stored checkpoint with ELO {current_elo}; checkpoint idx {self.checkpoint_counter}")

        self.checkpoint_counter += 1
        self.score_queue.clear()


    def update_checkpoint_elo(self, checkpoint_idx, opponent_score, current_elo):
        """
        Dynamically updates the ELO of a given checkpoint on disk and in memory.
        Uses os.rename to rename the checkpoint file on disk.
        """
        if checkpoint_idx not in self.stored_by_idx:
            return

        old_elo, old_filename = self.stored_by_idx[checkpoint_idx]
        
        # Calculate new ELO for the checkpoint
        E_B = 1 / (1 + 10 ** ((current_elo - old_elo) / self.elo_cfg.scale))
        new_elo = old_elo + self.elo_cfg.k_factor * (opponent_score - E_B)
        new_elo = int(round(new_elo))

        if new_elo != old_elo:
            new_filename = f"checkpoint_{checkpoint_idx}_{new_elo}.pt"
            old_path = os.path.join(self.checkpoint_dir, old_filename)
            new_path = os.path.join(self.checkpoint_dir, new_filename)
            
            try:
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                else:
                    print(f"[Warning] Checkpoint file {old_path} not found for renaming.")
            except Exception as e:
                print(f"[Error] Failed to rename checkpoint from {old_path} to {new_path}: {e}")
                return

            self.stored_by_idx[checkpoint_idx] = (new_elo, new_filename)
            
            if old_elo in self.stored_by_elo and checkpoint_idx in self.stored_by_elo[old_elo]:
                self.stored_by_elo[old_elo].remove(checkpoint_idx)
                if not self.stored_by_elo[old_elo]:
                    del self.stored_by_elo[old_elo]
                    
            if new_elo not in self.stored_by_elo:
                self.stored_by_elo[new_elo] = []
            self.stored_by_elo[new_elo].append(checkpoint_idx)
            
            print(f"Renamed checkpoint {checkpoint_idx} from ELO {old_elo} to {new_elo}")


    def get_state(self):
        """Return a dict that fully captures this manager's internal state."""
        return {
            "checkpoint_counter": self.checkpoint_counter,
            "score_queue": list(self.score_queue),
            "stored_by_idx": {k: list(v) for k, v in self.stored_by_idx.items()},
            "stored_by_elo": {k: list(v) for k, v in self.stored_by_elo.items()},
        }


    def load_state(self, state: dict):
        """Restore internal state from a previously saved dict."""
        self.checkpoint_counter = state["checkpoint_counter"]
        self.score_queue = deque(state["score_queue"], maxlen=self._score_queue_maxlen)
        
        stored_by_idx = {}
        for k, v in state["stored_by_idx"].items():
            idx = int(k)
            elo, val = v
            # Legacy state compatibility: extract just filename if absolute path is stored
            filename = os.path.basename(val)
            stored_by_idx[idx] = (int(elo), filename)
            
        self.stored_by_idx = stored_by_idx
        self.stored_by_elo = {
            int(k): list(v) for k, v in state["stored_by_elo"].items()
        }
