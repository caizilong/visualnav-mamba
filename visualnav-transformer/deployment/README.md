# Deployment Notes

This deployment setup now treats `nomad_mamba` as a first-class model entry.

## Model Weights

Place checkpoints under `deployment/model_weights/` with the filenames expected by
`deployment/config/models.yaml`.

- `nomad.pth`: original NoMaD checkpoint
- `nomad_mamba.pth`: NoMaD-Mamba checkpoint

The default NoMaD-Mamba deployment entry points to:

```text
deployment/model_weights/nomad_mamba.pth
```

## Reproducible Benchmark Config

Use `deployment/config/benchmark_nomad_mamba.yaml` to keep navigation and
exploration runs on a fixed parameter set.

It currently standardizes:

- `model: nomad_mamba`
- `waypoint: 2`
- `num_samples: 8`
- `guidance_min: 0.25`
- `guidance_max: 1.75`
- `guidance_power: 1.5`
- navigation defaults for `dir`, `goal_node`, `close_threshold`, and `radius`

## Navigation Commands

Run from `deployment/src/`:

```bash
python3 navigate.py --benchmark-config ../config/benchmark_nomad_mamba.yaml
```

Or with tmux launch:

```bash
./navigate.sh "--benchmark-config ../config/benchmark_nomad_mamba.yaml"
```

## Exploration Commands

Run from `deployment/src/`:

```bash
python3 explore.py --benchmark-config ../config/benchmark_nomad_mamba.yaml
```

Or with tmux launch:

```bash
./explore.sh "--benchmark-config ../config/benchmark_nomad_mamba.yaml"
```

## Manual Override

Command-line flags still take the same shape as before. The benchmark config is
only a convenient default bundle for repeatable evaluation.

Examples:

```bash
python3 navigate.py --model nomad_mamba --dir my_topomap --goal-node -1 --radius 6
python3 explore.py --model nomad_mamba --num-samples 16
```
