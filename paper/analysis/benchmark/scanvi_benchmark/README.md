# scANVI Benchmark Pipeline

Benchmark scANVI (single-cell ANnotation using Variational Inference) as a
classification-based cell-type annotation method, for comparison against
mLLMCelltype and popV in the published analysis.

## Overview

scANVI is a semi-supervised deep generative model that learns a shared latent
space between labelled reference and unlabelled query cells, then transfers
labels via a built-in classifier.  This pipeline uses the modern scvi-tools
API (v1.0+) with hyper-parameters matching the popV implementation.

## Directory structure

```
scanvi_benchmark/
  run_scanvi_benchmark.py      # Main training + prediction script
  submit_all.sh                # SLURM batch submission for arseven
  evaluate_scanvi_results.py   # Evaluation with ontology-aware matching
  README.md                    # This file
```

## Prerequisites

- Python 3.11 with scvi-tools, scanpy, and anndata installed
- NVIDIA GPU (A30 recommended, any CUDA-capable GPU works)
- Reference and query h5ad files in `$MLLMCELLTYPE_PAPER_ROOT/data/raw/`

Set `MLLMCELLTYPE_PAPER_ROOT` to the consolidated repository's `paper/`
directory. Optional cluster path overrides are documented in
`paper/cluster.env.example`.

## Running on arseven

### 1. Single dataset

```bash
export MLLMCELLTYPE_PAPER_ROOT=/path/to/mLLMCelltype/paper

python run_scanvi_benchmark.py \
    --dataset Thymus \
    --reference $MLLMCELLTYPE_PAPER_ROOT/data/raw/TS_Thymus_filtered.h5ad \
    --query $MLLMCELLTYPE_PAPER_ROOT/data/raw/Thymus.h5ad \
    --output_dir $MLLMCELLTYPE_PAPER_ROOT/results/benchmark/scanvi_benchmark/Thymus \
    --ref_label_key cell_type \
    --ref_batch_key donor \
    --save_model
```

### 2. All datasets via SLURM

```bash
# Review commands first
bash submit_all.sh --dry-run

# Submit jobs
bash submit_all.sh
```

SLURM configuration: partition=gpu, gres=gpu:a30:1, mem=64G, time=04:00:00,
cpus=8.

### 3. Evaluate results

```bash
python evaluate_scanvi_results.py \
    --dataset Thymus \
    --predictions $MLLMCELLTYPE_PAPER_ROOT/results/benchmark/scanvi_benchmark/Thymus/Thymus_scanvi_predictions.csv \
    --query $MLLMCELLTYPE_PAPER_ROOT/data/raw/Thymus.h5ad \
    --query_label_key cell_type \
    --output_dir $MLLMCELLTYPE_PAPER_ROOT/results/benchmark/scanvi_benchmark/Thymus
```

To include side-by-side comparison with LLM and popV, add:
```
    --llm_results /path/to/Thymus_results.csv \
    --popv_results /path/to/Thymus_popv_fast_results.csv
```

## Expected outputs

Per dataset, the pipeline produces:

| File | Description |
|------|-------------|
| `{dataset}_scanvi_predictions.csv` | Barcode-level predictions and probabilities |
| `{dataset}_scanvi_model/` | Saved scANVI model (if `--save_model`) |
| `{dataset}_scanvi_evaluation.csv` | Per-cell evaluation scores |
| `{dataset}_evaluation_summary.csv` | Summary accuracy table |

## Model parameters

Matching the popV scANVI configuration:

| Parameter | Value |
|-----------|-------|
| n_latent | 20 |
| n_layers | 3 |
| dropout_rate | 0.05 |
| gene_likelihood | nb |
| max_epochs (SCVI) | 20 |
| max_epochs (SCANVI) | 20 |
| batch_size | 512 |
| n_top_genes (HVG) | 4000 |

## Evaluation scoring

The evaluation uses the same cell-ontology-aware matching as the popV
comparison scripts:

- **1.0** -- Exact match, singular/plural variation, or known equivalent name
- **0.5** -- Hierarchical relationship (parent/child/sibling) or developmental
  stage relationship
- **0.0** -- No recognised match

The weighted accuracy is `mean(scores) * 100`.

## Notes

- The pipeline requires raw counts in the h5ad files (checks `.layers['counts']`,
  `.raw.X`, then falls back to `.X`).
- Gene sets are automatically harmonised to the intersection of reference and
  query before HVG selection.
- The unlabelled category for query cells is set to `"unlabeled"`.
