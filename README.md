# MarioRL

Train a PPO agent to play Super Mario Bros using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) on [stable-retro](https://stable-retro.farama.org/index.html#). This project targets Linux (Ubuntu 22.04 LTS on WSL) and uses [`uv`](https://docs.astral.sh/uv/guides/projects/) for a fast, reproducible Python environment. 
> A single training entrypoint (main.py) is provided along with a separate runner (play.py) to play the full game with a trained policy.

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
  <a href="#important-links">Important Links</a> •
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


---

## Requirements

- Ubuntu 22.04 LTS (native or WSL)
- Python 3.10
- uv, FFmpeg, and OpenGL
- A legally obtained Super Mario Bros NES ROM

> [!NOTE]
> Ubuntu 22.04 LTS includes Python 3.10 by default.

---
## Important Links

- The compiled understandings can be found in this [PDF](https://jumpshare.com/share/rktyYJu2MsWSoJvxOwcG)
- All resources are available on [TLDRAW board](https://www.tldraw.com/f/T6oHe2VW4S5P4fRhE0Aqv?d=v2479.1132.1820.864.EPwSiQalDCLRnIXbqC-Kl)

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

4) Install `uv`:
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

> After activation, all python and uv commands will use the project’s virtual environment.

---

## ROM Import

1) Place your ROM:
```bash
mkdir -p ROM
# Copy your legally obtained Super Mario Bros.nes file into ./ROM
```

2) Import into Retro’s database:
```bash
python3 -m retro.import ./ROM
```

> Ensure the game name (e.g., SuperMarioBros-Nes) matches what your scripts expect.

---
## Quick Test: Random Agent

Run a quick environment sanity check:
```bash
python3 randomAgent.py
```
> [!IMPORTANT]
> Terminate the process with `Ctrl+C`.

*OPTIONAL: check CUDA availability for PyTorch*

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```
> [!WARNING]
> Set `device = "cuda"` as we are dealing with images
---

## Train

Run the training entrypoint:
```bash
python3 main.py
```
---

## Play the Full Game

*OPTIONAL: You can select the path file to your desired folder. For now, I have set it up for you.*

After training, run the separate runner to play:
```bash
python3 play.py
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
> [!IMPORTANT] 
> If `Imported 0 games` shows up, it could mean you downloaded a corrupted copy.
 
- Clean stop vs suspend:
```bash
# Clean stop
# Press Ctrl+C

# If you pressed Ctrl+Z by mistake:
# enter fg on the terminal to bring it to foreground, then press Ctrl+C
```
---

## Legal

ROMs are not included and must be obtained legally. Do not commit or distribute ROMs in this repository.
