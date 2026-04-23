# NoMaD-Mamba (Slimmed Repository)

This repository has been cleaned to keep only the code needed for **NoMaD-Mamba** training and deployment.

## Kept scope

- NoMaD container + NoMaD-Mamba vision encoder
- Diffusion-policy based training loop
- ROS deployment scripts for navigation/exploration
- Data processing and dataset loading pipeline

## Removed scope

- GNM
- ViNT
- NoMaD-ViNT / other legacy vision encoders

## Train

From `train/`:

```bash
python3 train.py -c ./config/nomad_mamba.yaml
```

or use:

```bash
./run_train.sh
```

## Deploy

See [deployment/README.md](deployment/README.md).

## Notes

- `train/train.py` only accepts `model_type: nomad` + `vision_encoder: nomad_mamba`.
- `deployment/src/navigate.py` and `deployment/src/explore.py` only load NoMaD-Mamba configs.
