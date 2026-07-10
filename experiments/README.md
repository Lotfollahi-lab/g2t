# Experiment recipes

One YAML per run — the **single source of truth** for how a run was configured,
versioned in the repo so any run can be reproduced by reading (or re-launching)
its file, with no config-hunting on disk.

## Launch / inspect
```bash
python scripts/launch_experiment.py --list                 # list recipes
python scripts/launch_experiment.py <name> --dry-run        # print the exact submit command
python scripts/launch_experiment.py <name>                  # submit it to LSF
```

## Recipe schema (`experiments/<name>.yaml`)
```yaml
description: free text — what this run is / why
method: scgg                     # scgg | luna | celery | novosparc
data_dir: /nfs/.../silver/mmc_luna
seed: 0
epochs: 2000
gpu_model: H200
queue: training-parallel         # optional
group: s10396                    # optional
wandb_run_name: scgg_...         # optional (defaults to the recipe name)
submit_script: /path/...         # optional (else $SCGG_SUBMIT or the built-in default)
extra_flags: ["--skip_training"] # optional — appended to the submit command verbatim
override:                        # Hydra model/train overrides (key: value)
  model.loss.sparse_local_distance.enabled: true
  model.position_feedback: "off"     # QUOTE "off"/"on"/"no" — YAML parses them as booleans otherwise
  train.lr: "8e-4"                   # quote scientific notation to keep it verbatim
```

The launcher flattens `override:` into the `--override "k=v k=v …"` string the
pipeline expects (booleans → `true`/`false`), and forwards the rest as submit
flags. To reproduce a run exactly: open its YAML, or run it with `--dry-run` to
see the command.

## Conventions
- Name the file for the run: `<dataset>_<method-variant>[_<seed>].yaml`.
- One run per file. For an A/B, make two files that differ only in the ablated key.
- When you find a good run on wandb/disk, capture it here immediately so it's never lost.
