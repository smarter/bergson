# DVC Pipeline for K-FAC Memorization Experiments

The bergson repository uses DVC to manage reproducible ML pipelines. The K-FAC memorization removal experiments from the paper "From Memorization to Reasoning in the Spectrum of Loss Curvature" are defined in the repository root.

## Pipeline Overview

The pipeline consists of two stages:

1. **collect_kfac_factors**: Collect K-FAC factors (A and G matrices) from the model
2. **eval_kfac_edit**: Apply K-FAC edit and evaluate memorization reduction

## Quick Start

### Run the full pipeline

```bash
# From repository root
uv run dvc repro
```

### Run individual stages

```bash
# Collect K-FAC factors only
uv run dvc repro collect_kfac_factors

# Run evaluation only (requires K-FAC factors)
uv run dvc repro eval_kfac_edit
```

### View pipeline DAG

```bash
uv run dvc dag
```

### Check pipeline status

```bash
uv run dvc status
```

## Configuration

Edit `params.yaml` to customize:

- **Model settings**: model size, layers to edit
- **K-FAC collection**: which projections (gate/up/down) to collect
- **Edit parameters**: keep_mass value (0-1) controls compression strength
  - Higher keep_mass (e.g., 0.8) = less compression, more memorization retained
  - Lower keep_mass (e.g., 0.6) = more compression, stronger memorization suppression

## Outputs

- **K-FAC factors**: Saved to `data/kfac_factors/olmo2-7b/`
  - Contains A (activation covariance) and G (gradient covariance) matrices
- **Evaluation metrics**: Saved to `metrics/kfac_eval_results.json`
  - Perplexity, memorization metrics, etc.

## Advanced Usage

### Modify parameters and rerun

```bash
# Edit params.yaml to change keep_mass
vim params.yaml

# Rerun affected stages
uv run dvc repro
```

### Experiment tracking

```bash
# Show metrics
uv run dvc metrics show

# Compare experiments
uv run dvc exp show
```

## Integration with Bergson

The goal is to replace the K-FAC collection (`data/collect_kfac_multilayer.py`) with bergson's EKFAC implementation. This pipeline will help validate that the replacement produces equivalent results.
