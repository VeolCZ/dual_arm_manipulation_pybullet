# Useful commands

Run commands from the repository root. Copy the commands without putting quotes
around the whole command or block.

## Start the development shell

```bash
nix develop
```

Wait for the new shell prompt. Run the remaining commands inside that shell.
uv creates and manages `.venv`; manual activation is unnecessary.

Install the locked dependencies on first setup:

```bash
uv sync --locked
```

Leave the development shell with `exit`.

## Run the simulation with a window

```bash
uv run simulation.py --scenario=pile
```

GUI, GPU inference, GR_ConvNet, and one run are the defaults. If CUDA is
unavailable, inference falls back to CPU.

Other scenarios and multiple runs:

```bash
uv run simulation.py --scenario=packed
uv run simulation.py --scenario=isolated
uv run simulation.py --scenario=pile --runs=5
```

An isolated run goes through every object in the configured object list.

From a normal shell, run directly without entering an interactive Nix shell:

```bash
nix develop --command uv run simulation.py --scenario=pile
```

To measure the total runtime:

```bash
time nix develop --command uv run simulation.py --scenario=pile
```

## CPU and headless runs

Force CPU inference while keeping the PyBullet window:

```bash
uv run simulation.py --scenario=pile --device=CPU
```

Run without a window, keeping GPU inference:

```bash
uv run simulation.py --scenario=pile --gui=false --save-network-output=false
```

A short run that stops after one consecutive placement failure:

```bash
uv run simulation.py --scenario=pile --gui=false --attempts=1 --save-network-output=false
```

## Live camera previews

Use Matplotlib with the project's headless OpenCV package:

```bash
uv run simulation.py --scenario=pile --stream_cameras=true --stream_backend=matplotlib
```

Preview only the overhead camera at two frames per second:

```bash
uv run simulation.py --scenario=pile --stream_cameras=true --stream_backend=matplotlib --stream_which=external --stream_fps=2
```

## Record and export episodes

Record the default cameras into `episodes/`:

```bash
uv run simulation.py --scenario=pile --record_episodes=true
```

Verify saved recordings without opening windows:

```bash
uv run view_episodes.py episodes --verify
```

Replace `EPISODE_NAME` below with the folder printed by the recorder. Export
paths are relative to that episode folder:

```bash
uv run view_episodes.py episodes/EPISODE_NAME --save-video=episode.mp4
uv run view_episodes.py episodes/EPISODE_NAME --save-video=episode.gif
```

Interactive episode playback requires GUI-enabled OpenCV; export and
verification work with the installed headless OpenCV package.

## Tests and diagnostics

Run the headless test suite:

```bash
MPLBACKEND=Agg uv run --locked python -m unittest discover -s tests
```

Known blocker: `test_grasp_generator` imports the missing
`trained_models.GGCNN.models` module. The other 74 tests passed during validation.

Run the behavior checks without a GUI or pauses:

```bash
uv run test_behaviours.py --vis=false --sleep=0
```

Check that PyTorch can see the GPU (fails if CUDA is unavailable):

```bash
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
```

Validate the dependency lock and Nix shell definitions:

```bash
uv lock --check
nix flake check --all-systems --no-build
```

See all simulation options:

```bash
uv run simulation.py --help
```
