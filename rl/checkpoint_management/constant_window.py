import os
import numpy as np
import torch as t


class ConstantWindow_CheckpointManagement:
    def __init__(
        self, 
        checkpoint_dir, 
        window_size, 
        save_every_k_global_steps,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.window_size = window_size
        self.save_every_k_global_steps = save_every_k_global_steps

        self.checkpoint_counter = 0
        os.makedirs(self.checkpoint_dir, exist_ok=True)


    def load(self, net, current_elo=None, active_checkpoints=None):
        checkpoint_min = max(0, self.checkpoint_counter - self.window_size)
        checkpoint_max = self.checkpoint_counter - 1

        if checkpoint_max <= checkpoint_min:
            return 1200, None

        allowed_indices = [idx for idx in range(checkpoint_min, checkpoint_max) if idx not in (active_checkpoints or [])]
        if not allowed_indices:
            allowed_indices = list(range(checkpoint_min, checkpoint_max))

        checkpoint_sample = np.random.choice(allowed_indices)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{checkpoint_sample}.pt")
        net.load_state_dict(t.load(checkpoint_path, map_location=next(net.parameters()).device, weights_only=True))
        return 1200, checkpoint_sample
    

    def store(self, net, global_step):
        if global_step % self.save_every_k_global_steps != 0:
            return 

        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{self.checkpoint_counter}.pt")
        t.save(net.state_dict(), checkpoint_path)

        self.checkpoint_counter += 1
