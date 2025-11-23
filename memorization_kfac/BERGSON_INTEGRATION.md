# Bergson Integration Status

## Overview

Successfully integrated bergson's EKFAC implementation into the memorization_kfac project as an alternative K-FAC collector.

## Implementation

### New Files

- **`memorization_kfac/data/collect_kfac_bergson.py`**: K-FAC collector using bergson's `CovarianceCollector`
  - Same CLI interface as `collect_kfac_multilayer.py`
  - Uses bergson's hook-based covariance collection
  - Outputs in bergson's native safetensors format

### DVC Pipeline

Added `collect_kfac_bergson` stage to `dvc.yaml` that runs the bergson-based collector alongside the original.

## Format Differences

### Original Collector Output

```
kfac_factors/olmo2_1b/kfac_factors_blk_14.pt:
{
  'blk14.gate': {'A': Tensor([2048, 2048]), 'G': Tensor([8192, 8192]), 'n_tokens': 3066},
  'blk14.up': {'A': Tensor([2048, 2048]), 'G': Tensor([8192, 8192]), 'n_tokens': 3066},
  'blk14.down': {'A': Tensor([8192, 8192]), 'G': Tensor([2048, 2048]), 'n_tokens': 3066}
}
```

- Single `.pt` file with all projections
- Matrices are **normalized** by `n_tokens` (number of sequence positions)
- Keys use format `blkN.{gate,up,down}`

### Bergson Collector Output

```
kfac_factors_bergson/
├── activation_covariance_sharded/shard_0.safetensors
├── gradient_covariance_sharded/shard_0.safetensors
└── metadata.json
```

- Separate safetensors files for A and G covariances
- Matrices are **raw sums** (not normalized)
- Keys use full module names: `layers.14.mlp.{gate,up,down}_proj`

## Key Implementation Details

### Module Naming

The model hierarchy uses different naming conventions:
- `model.layers.14.mlp.gate_proj` - Full path from model root
- `layers.14.mlp.gate_proj` - Path from `model.base_model`

The bergson collector uses the latter (base_model names) because that's where hooks are attached.

### Context Manager Pattern

`CovarianceCollector` is a context manager:

```python
with collector:
    # Forward and backward passes trigger hooks
    loss.backward()
# teardown() called automatically to save results
```

### Covariance Accumulation

Both collectors compute:
- `A = Σ(x^T @ x)` for activations
- `G = Σ(g^T @ g)` for gradients

But differ in normalization and sequence length handling.

## Next Steps

To use bergson's output with existing memorization_kfac evaluation:

1. **Option A: Format Converter**
   - Create script to convert bergson safetensors → original .pt format
   - Handle normalization and key name mapping

2. **Option B: Modify Evaluation**
   - Update `KFACTreatmentPairwise` to support both formats
   - Auto-detect format and load accordingly

3. **Option C: Use Bergson End-to-End**
   - Replace entire pipeline with bergson's EKFAC
   - Reimplement memorization removal using bergson's eigenvectors

## Testing

Verified that bergson collector:
- ✅ Discovers correct target modules
- ✅ Collects covariances (145MB activation, 265MB gradient)
- ✅ Saves in safetensors format
- ✅ Matches shapes of original collector
- ⚠️ Different normalization (raw sums vs. normalized)

## Usage

```bash
# Run bergson collector
uv run dvc exp run collect_kfac_bergson

# Run original collector (for comparison)
uv run dvc exp run collect_kfac_factors
```
