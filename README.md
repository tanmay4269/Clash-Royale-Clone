# Clash Royale RL Playground

A custom tick-based Clash Royale-style simulator for reinforcement learning experiments. The core game engine is Python/Pygame, with PPO training code and a browser-playable web demo.

## Play

Live browser demo:

https://tanmay-4269-clash-royale-web-play.hf.space

The demo lets you play as blue against built-in bots or trained [run 30](docs/index.html#386aa1bf-7b79-806f-a831-df97cdbff596)/[run 31](docs/index.html#386aa1bf-7b79-80e3-812e-e9d67b54c853) checkpoints. It is hosted on Hugging Face Spaces, so the first load can be slow if the free Space has gone to sleep.

## Docs

The GitHub Pages article lives at:

```bash
docs/index.html
```

It is the main writeup for the simulator, PPO setup, training runs, and gameplay videos.

## Run The Browser Demo Locally

Install the web dependencies:

```bash
python3 -m pip install -r web_play/requirements-web.txt
```

Start the local server:

```bash
python3 -m web_play.server --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

## Run With Docker

Build the image:

```bash
docker build -t clash-royale-web-play .
```

Run it:

```bash
docker run --rm -p 7860:7860 clash-royale-web-play
```

Open:

```text
http://127.0.0.1:7860
```

## Package For Hugging Face Spaces

Create the upload folder:

```bash
python3 web_play/package_hf_space.py
```

The package is written to:

```bash
dist/hf-space
```

That folder contains the Docker Space files plus the served checkpoint folders.

## Run The Simulator Directly

Install the base project dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Run the local Pygame simulator:

```bash
python game.py
```

## Project Layout

- `game/`: tick-based simulator, entities, arena logic, rendering.
- `rl/`: PPO trainer, Gymnasium environment wrapper, networks, checkpoint logic.
- `web_play/`: browser-playable wrapper around the simulator.
- `docs/`: GitHub Pages article and article assets.
- `assets/`: card sprites used by the simulator and web demo.
