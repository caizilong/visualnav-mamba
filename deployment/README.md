# Deployment (NoMaD-Mamba)

This deployment folder now supports **NoMaD-Mamba only**.

## Required files

- Model entry: `deployment/config/models.yaml`
- Model config: `train/config/defaults.yaml`
- Weights: `deployment/model_weights/nomad_mamba.pth`

## Benchmark config

Use:

- `deployment/config/benchmark_nomad_mamba.yaml`

## Run navigation

From `deployment/src/`:

```bash
python3 navigate.py --benchmark-config ../config/benchmark_nomad_mamba.yaml
```

or:

```bash
./navigate.sh "--benchmark-config ../config/benchmark_nomad_mamba.yaml"
```

## Run exploration

From `deployment/src/`:

```bash
python3 explore.py --benchmark-config ../config/benchmark_nomad_mamba.yaml
```

or:

```bash
./explore.sh "--benchmark-config ../config/benchmark_nomad_mamba.yaml"
```
