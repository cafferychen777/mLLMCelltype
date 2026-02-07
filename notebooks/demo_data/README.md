# Demo Data for mLLMCelltype Tutorial

This directory contains pre-computed results for the mLLMCelltype tutorial notebook's demo mode.

## Purpose

These cached results allow users to experience the complete mLLMCelltype workflow without requiring API keys or incurring costs. The demo mode is automatically activated when:
1. No API keys are configured
2. The user selects the example PBMC dataset

## Files

### `cached_results.csv`
Simple CSV format with the final annotation results:
- **Cluster**: Cluster identifier (e.g., "Cluster_0")
- **Cell Type**: Consensus cell type annotation
- **Consensus Score**: Agreement level between models (0-1)
- **Entropy**: Shannon entropy measuring annotation uncertainty

### `cached_detailed_results.json`
Complete JSON structure matching the output of `interactive_consensus_annotation()`:
- **consensus**: Final consensus annotations for each cluster
- **consensus_proportion**: Agreement scores (0-1) for each annotation
- **entropy**: Shannon entropy values measuring uncertainty
- **model_annotations**: Individual predictions from each model
- **controversial_clusters**: Clusters requiring multi-round discussion

## Data Origin

These results were generated using the example PBMC marker genes with a multi-model consensus approach:
- **Models used**: GPT-4 Turbo, Claude Sonnet 4.5, Gemini 1.5 Pro
- **Dataset**: PBMC (Peripheral Blood Mononuclear Cells)
- **Species**: Human
- **Clusters**: 7 clusters (T cells, B cells, Monocytes, NK cells, etc.)

The marker genes used:
```python
{
    "Cluster_0": ["IL7R", "CD3D", "CD3E", "CD3G", "TRAC"],  # T cells
    "Cluster_1": ["CD79A", "MS4A1", "CD19", "BANK1"],       # B cells
    "Cluster_2": ["CD14", "LYZ", "S100A8", "S100A9"],       # Monocytes
    "Cluster_3": ["GNLY", "NKG7", "PRF1", "GZMB"],          # NK cells
    "Cluster_4": ["FCER1A", "CST3", "CLEC10A"],             # Dendritic cells
    "Cluster_5": ["FCGR3A", "MS4A7", "IFITM3"],             # CD16+ Monocytes
    "Cluster_6": ["PPBP", "PF4", "GP9"],                    # Platelets
}
```

## Transparency

**Important**: When demo mode is active, the notebook displays clear messages informing users that they are viewing pre-computed results, not live LLM predictions.

## Validation

The demo data has been validated to ensure:
1. All required fields are present
2. Data structures match the live annotation output
3. All downstream visualization and analysis cells work correctly
4. Consensus scores and entropy values are realistic
5. CSV and JSON formats are consistent

Run `python3 test_demo_mode.py` from the notebooks directory to validate the demo data integrity.

## Updating Demo Data

If you need to regenerate the demo data with fresh LLM predictions:

1. Configure your API keys
2. Run the notebook with the example PBMC data
3. Copy the outputs:
   - `mllmcelltype_results.csv` → `demo_data/cached_results.csv`
   - `mllmcelltype_detailed_results.json` → `demo_data/cached_detailed_results.json`
4. Run the validation test to ensure compatibility

## Academic Integrity

The demo mode implementation clearly labels cached vs. live results. Users are encouraged to run live annotations with their own API keys for production analyses.
