# Model Weight Naming

Put deployment checkpoints in this folder.

Recommended filenames:

- `nomad.pth`: original NoMaD checkpoint
- `nomad_mamba.pth`: NoMaD-Mamba checkpoint used by `deployment/config/models.yaml`

If you change filenames, update `deployment/config/models.yaml` accordingly.
