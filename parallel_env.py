"""
Parallel environment manager for PPO training.

Spawns N worker processes, each running an independent ClashRoyaleEnv.
Workers communicate via multiprocessing.Pipe, which is the simplest and 
fastest IPC for this use case.

Usage:
    penv = ParallelEnvManager(num_envs=8, env_name="ClashRoyaleEnv-v0", env_kwargs={...})
    observations = penv.reset(seeds=[42, 43, ...])
    
    # Each step: send N actions, receive N results
    results = penv.step(actions_list)  # list of N action dicts
    # results = [(obs, reward, terminated, truncated), ...]
    
    penv.close()
"""

import multiprocessing as mp
import traceback
import gymnasium as gym
import numpy as np


def _worker_fn(pipe, env_name, env_kwargs, wrapper_cls):
    """
    Worker process: runs a single env instance and communicates via pipe.
    
    Protocol:
        ("reset", seed)       -> obs
        ("step", action)      -> (obs, reward, terminated, truncated)
        ("get_attr", name)    -> getattr(env, name): for arena access etc.
        ("close", None)       -> breaks loop
    
    On error: sends ("error", traceback_string) so the parent gets a clear message
    instead of a broken pipe.
    """
    env = gym.make(env_name, **env_kwargs)
    if wrapper_cls is not None:
        env = wrapper_cls(env)

    try:
        while True:
            cmd, data = pipe.recv()

            if cmd == "close":
                break

            try:
                if cmd == "reset":
                    obs, info = env.reset(seed=data)
                    pipe.send(obs)

                elif cmd == "step":
                    obs, reward, terminated, truncated, info = env.step(data)
                    pipe.send((obs, reward, terminated, truncated, info))

                elif cmd == "get_attr":
                    pipe.send(getattr(env, data))

                elif cmd == "get_unwrapped_attr":
                    pipe.send(getattr(env.unwrapped, data))

            except Exception:
                pipe.send(("error", traceback.format_exc()))

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def _check_result(result, worker_idx):
    """Raise if the worker sent back an error tuple."""
    if isinstance(result, tuple) and len(result) == 2 and result[0] == "error":
        raise RuntimeError(f"Worker {worker_idx} crashed:\n{result[1]}")
    return result


class ParallelEnvManager:
    """
    Manages N env workers running in separate processes.
    
    All communication is synchronous: you send commands to all workers,
    then recv from all workers. This keeps the code simple and predictable.
    """

    def __init__(self, num_envs, env_name, env_kwargs=None, wrapper_cls=None):
        self.num_envs = num_envs
        self.env_name = env_name
        self.env_kwargs = env_kwargs or {}
        self.wrapper_cls = wrapper_cls

        # One pipe pair per worker: (parent_end, child_end)
        self.parent_pipes = []
        self.workers = []

        for i in range(num_envs):
            parent_pipe, child_pipe = mp.Pipe()
            worker = mp.Process(
                target=_worker_fn,
                args=(child_pipe, env_name, self.env_kwargs, wrapper_cls),
                daemon=True,
            )
            worker.start()
            child_pipe.close()  # parent doesn't need the child end

            self.parent_pipes.append(parent_pipe)
            self.workers.append(worker)

    def reset(self, seeds=None):
        """
        Reset all envs. Returns list of N observations.
        
        seeds: list of N seeds, or None (uses default seeding)
        """
        if seeds is None:
            seeds = [None] * self.num_envs

        for pipe, seed in zip(self.parent_pipes, seeds):
            pipe.send(("reset", seed))

        return [_check_result(pipe.recv(), i) for i, pipe in enumerate(self.parent_pipes)]

    def reset_single(self, env_idx, seed=None):
        """Reset a single env and return its observation."""
        self.parent_pipes[env_idx].send(("reset", seed))
        return _check_result(self.parent_pipes[env_idx].recv(), env_idx)

    def step(self, actions):
        """
        Step all envs. Returns list of N tuples: (obs, reward, terminated, truncated, info).
        
        actions: list of N action dicts
        """
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(("step", action))

        return [_check_result(pipe.recv(), i) for i, pipe in enumerate(self.parent_pipes)]

    def step_single(self, env_idx, action):
        """Step a single env. Returns (obs, reward, terminated, truncated, info)."""
        self.parent_pipes[env_idx].send(("step", action))
        return _check_result(self.parent_pipes[env_idx].recv(), env_idx)

    def get_attr(self, name):
        """Get an attribute from all envs (e.g., 'arena')."""
        for pipe in self.parent_pipes:
            pipe.send(("get_attr", name))
        return [_check_result(pipe.recv(), i) for i, pipe in enumerate(self.parent_pipes)]

    def get_unwrapped_attr(self, name):
        """Get an attribute from all unwrapped envs."""
        for pipe in self.parent_pipes:
            pipe.send(("get_unwrapped_attr", name))
        return [_check_result(pipe.recv(), i) for i, pipe in enumerate(self.parent_pipes)]

    def close(self):
        """Shut down all workers."""
        for pipe in self.parent_pipes:
            try:
                pipe.send(("close", None))
            except BrokenPipeError:
                pass

        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()

    def __del__(self):
        self.close()
