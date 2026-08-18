# mLLMCelltype paper workspace

This directory contains the research record for the mLLMCelltype article:

> Yang C, Zhang X, Chen J. Large language model consensus substantially improves the cell type annotation accuracy for scRNA-seq data. *Communications Biology*. 2026. https://doi.org/10.1038/s42003-026-10420-8

The article was published online on June 8, 2026. This workspace is retained for provenance and reproducibility; product documentation and installation instructions live in the repository root.

## Directory boundaries

- `analysis/`: benchmark and evaluation source code.
- `reproducibility/`: compact method entry points from the public reproduction repository; shared utilities use the canonical `scripts/` and `examples/` copies, while the original Git history remains preserved.
- `scripts/`: data preparation, ontology, marker, and visualization utilities.
- `manuscript/`: final manuscript sources, figures, supplementary data, and local publication records.
- `data/`: local input data. It is intentionally excluded from Git because the files are large or externally sourced.
- `results/`: local derived outputs. Final article figures also live under `manuscript/figures/`.
- `assets/` and `examples/`: paper-specific visual assets and small workflow examples.

The maintained software implementations are the top-level `R/` and `python/` packages. Historical copies formerly stored under `code/` were removed during repository consolidation to preserve a single source of truth.

## Reproducibility conventions

Run large analyses on the configured compute cluster rather than on a laptop. Python environments should be recreated with `uv`; R jobs on arseven use R 4.4.1 and the `~/R/4.4` library. Local secrets belong in `.env` and must never be committed.

Run local analysis commands from this `paper/` directory so relative data and result paths resolve consistently. Cluster scripts expose environment variables for site-specific scratch paths. Data availability and external service versions can still affect exact reruns. The published article, supplementary information, and archived final submission package remain the authoritative frozen record.
