# MarioRL

Train a PPO agent to play Super Mario Bros using Stable-Baselines3 on stable-retro. This project targets Linux (Ubuntu 22.04 LTS on WSL is supported) and uses uv for a fast, reproducible Python environment. A single training entrypoint (main.py) is provided along with a separate runner (play.py) to play the full game with a trained policy.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" />
  <img src="https://img.shields.io/badge/RL-PPO-orange" />
  <img src="https://img.shields.io/badge/Framework-Stable--Baselines3-2ea44f" />
  <img src="https://img.shields.io/badge/Emulator-stable--retro-d73a49" />
  <img src="https://img.shields.io/badge/Backend-PyTorch-ee4c2c?logo=pytorch" />
  <img src="https://img.shields.io/badge/Env%20Manager-uv-7c3aed" />
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#requirements">Requirements</a> •
  <a href="#quickstart-windows--wsl--vs-code">Quickstart</a> •
  <a href="#rom-import">ROM Import</a> •
  <a href="#train">Train</a> •
  <a href="#play-the-full-game">Play</a> •
  <a href="#monitoring">Monitoring</a> •
  <a href="#tips">Tips</a> •
  <a href="#legal">Legal</a>
</p>

---

## Overview

- Stable-Baselines3 (PPO) with stable-retro
- Linux-first workflow (Ubuntu 22.04 LTS on WSL supported)
- uv-managed virtual environment
- ROM import instructions
- TensorBoard monitoring
- Runner script to play the full game using a trained policy

> Ubuntu 22.04 LTS includes Python 3.10 by default.

---

## Requirements

- Ubuntu 22.04 LTS (native or WSL)
- Python 3.10 (included with Ubuntu 22.04)
- uv (install below)
- FFmpeg
- A legally obtained Super Mario Bros NES ROM

---

## Quickstart (Windows • WSL • VS Code)

1) Install Ubuntu 22.04 (WSL):
```bash
wsl --install --distribution Ubuntu-22.04
```

> Open Ubuntu from the Start menu and continue in that terminal.

2) Clone and open:
```bash
git clone https://github.com/amugoodbad229/MarioRL.git
cd MarioRL
code .
```

3) System dependencies:
```bash
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install -y git curl zlib1g-dev libssl-dev ffmpeg python3-opengl
```

4) Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

5) Sync the environment (creates .venv and installs dependencies):
```bash
uv sync
```

6) Activate the environment:
```bash
source .venv/bin/activate
```

> After activation, all python and pip commands will use the project’s virtual environment.

---

## ROM Import

1) Place your ROM:
```bash
mkdir -p ROM
# Copy your legally obtained Super Mario Bros .nes file into ./ROM
```

2) Import into Retro’s database:
```bash
python3 -m retro.import ./ROM
```

> Ensure the game name (e.g., SuperMarioBros-Nes) matches what your scripts expect.

---

## Train

Run the training entrypoint:
```bash
python3 main.py
```

> Stop cleanly with Ctrl+C.  
> If you suspended with Ctrl+Z by mistake, bring it to the foreground with:
```bash
fg
```
> Then press Ctrl+C to terminate.

Optional: check CUDA availability for PyTorch:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## Play the Full Game

After training, run the separate runner to play:
```bash
python3 play.py --model runs/ppo_mario/final_model.zip \
  --game SuperMarioBros-Nes --state Level1-1 --episodes 3 --deterministic
```

> If you used observation preprocessing during training (e.g., grayscale/resize/framestack), mirror the same wrappers in play.py.  
> On WSL, rendering requires an X server on Windows. Headless training works without rendering.

---

## Monitoring

Launch TensorBoard:
```bash
python -m tensorboard.main --logdir tensorboard
```

> Open the printed URL in your browser to view training curves and metrics.

---

## Tips

- Ensure the environment is active:
```bash
source .venv/bin/activate
```

- Re-run sync if dependencies change:
```bash
uv sync
```

- Re-import ROMs if not detected:
```bash
python3 -m retro.import ./ROM
```

---

## Legal

ROMs are not included and must be obtained legally. Do not commit or distribute ROMs in this repository.
