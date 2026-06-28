# Clash Royale Web Play

Local browser wrapper around the existing Python simulator.

## Local run

```bash
python3 -m pip install -r web_play/requirements-web.txt
python3 -m web_play.server --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`.

## Current opponent mapping

- `random`, `scripted`, `skip`: built-in bots.
- `run-30.2`: `legacy_deep_sets_attention`, shown as Deep Sets with entity attention.
- `run-31`: `transformer`, shown as Transformer.
- Best checkpoint means highest Elo suffix in the checkpoint filename.

Current discovered best checkpoints:

- `run-30.2`: `checkpoint_43_2001.pt`
- `run-31`: `checkpoint_110_1975.pt`

## Notes

- Blue is the human player, mapped to `player_side_2`.
- Red is the bot, mapped to `player_side_1`.
- The browser streams raw RGBA frames from a server-side Pygame surface, so checkpoint inference stays in Python for now.

## Deployment direction

## Docker run

```bash
docker build -t clash-royale-web-play .
docker run --rm -p 7860:7860 clash-royale-web-play
```

Open `http://127.0.0.1:7860`.

## Hugging Face Spaces deploy

Recommended first public host: Docker Space, CPU Basic.

1. Create a new Hugging Face Space with SDK set to Docker.
2. Copy `web_play/HF_SPACE_README.md` to the Space repo as `README.md`.
3. Push this repo's `Dockerfile`, `.dockerignore`, `game/`, `rl/`, `web_play/`, `assets/`, `pyproject.toml`, and the two served checkpoint folders under `saved-runs/`.
4. The container listens on `PORT`, defaulting to Hugging Face's `7860`.

Useful commands after `huggingface-cli login`:

```bash
huggingface-cli repo create clash-royale-web-play --type space --sdk docker
git remote add hf https://huggingface.co/spaces/<your-username>/clash-royale-web-play
git push hf HEAD:main
```

If the Space repo is separate, copy `web_play/HF_SPACE_README.md` to `README.md` before pushing.

## Production hardening

1. Add per-session match storage instead of one global active match.
2. Add a hard match timeout and idle cleanup.
3. Replace polling with WebSockets once local behavior is stable.
4. Keep only approved checkpoint files in the image.
5. Consider ONNX or TorchScript export if Python-side inference becomes the bottleneck.
