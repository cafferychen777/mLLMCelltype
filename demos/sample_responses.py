#!/usr/bin/env python3
"""
Sample responses and test data for langextract testing.

This module contains realistic LLM response samples based on the mLLMCelltype
project output formats, including cell type annotations, consensus results,
and discussion logs.
"""

from typing import Dict, List, Any

# Sample cell type annotation responses from different LLM providers
CELL_TYPE_RESPONSES = {
    "openai_gpt4_response": """
Based on the marker genes provided, here are the cell type annotations:

0: T cells (CD3+)
1: B cells 
2: Monocytes/Macrophages
3: NK cells
4: Dendritic cells
5: Neutrophils
6: Plasma cells
7: Regulatory T cells
8: Memory B cells
""",
    
    "claude_response": """
After analyzing the differential gene expression patterns, I can provide the following cell type annotations:

Cluster 0: CD8+ T cells - High expression of CD8A, CD8B, and cytotoxic markers
Cluster 1: B cells - Strong B cell markers including CD19, MS4A1 (CD20)
Cluster 2: Classical monocytes - CD14+, CD16- monocytes with inflammatory signature
Cluster 3: Natural Killer cells - GNLY, NKG7, KLRD1 expression
Cluster 4: Conventional dendritic cells - CD1C+, FCER1A+ dendritic cells
Cluster 5: Neutrophils - High S100A8, S100A9, neutrophil-specific genes
Cluster 6: Plasma cells - IGHG1, JCHAIN, immunoglobulin heavy chains
Cluster 7: CD4+ regulatory T cells - FOXP3+, IL2RA+ Tregs
Cluster 8: Memory B cells - CD27+, switched memory B cell signature
""",
    
    "gemini_response": """
Cell Type Annotation Results:

1. T cells (cytotoxic)
2. B cells (naive) 
3. Monocytes (classical)
4. NK cells
5. Dendritic cells (myeloid)
6. Neutrophils
7. Plasma cells
8. T cells (regulatory)
9. B cells (memory)
""",
    
    "mixed_format_response": """
Here are my predictions for the cell clusters:

**Cluster 0**: Cytotoxic T lymphocytes
The high expression of CD8A, CD8B, GZMB suggests these are CD8+ T cells.

**Cluster 1**: Naive B cells  
Strong expression of MS4A1, CD19, PAX5 indicates B cell lineage.

**Cluster 2**: Classical monocytes
CD14 high, CD16 low expression pattern typical of classical monocytes.

Cluster 3: Natural killer cells - GNLY and NKG7 are classic NK markers
Cluster 4: Dendritic cells - CD1C and FCER1A positive
Cluster 5: Neutrophils - S100A8/A9 high
Cluster 6: Plasma cells - High immunoglobulin expression
Cluster 7: Regulatory T cells - FOXP3 positive
Cluster 8: Memory B cells - CD27 positive, class-switched
""",
    
    "inconsistent_format_response": """
Looking at the gene expression patterns:

For cluster #0: These appear to be CD8+ T cells
Cluster 1 - B cells based on CD19 expression  
2: Monocytes (CD14+)
NK cells (cluster 3)
4. cDCs - conventional dendritic cells
5) Neutrophils with high S100 genes
Plasma cells in cluster 6
Cluster 7: Tregs (regulatory T cells)
8 = Memory B cells
"""
}

# Sample consensus check responses
CONSENSUS_RESPONSES = {
    "consensus_reached": """
Consensus Analysis Results:

After reviewing the predictions from all models:
- GPT-4: "CD8+ T cells"  
- Claude: "Cytotoxic T lymphocytes"
- Gemini: "T cells (cytotoxic)"

CONSENSUS REACHED: YES
Majority Prediction: CD8+ T cells
Consensus Proportion: 1.0 (100% agreement on T cell identity)
Entropy: 0.0 (perfect agreement)
Confidence Level: High

All models agree these are cytotoxic T cells, with slight variations in terminology.
""",
    
    "consensus_not_reached": """
Consensus Analysis:

Model predictions for this cluster:
- GPT-4: "Dendritic cells"
- Claude: "Monocytes/Macrophages" 
- Gemini: "Classical monocytes"
- DeepSeek: "Myeloid dendritic cells"

CONSENSUS REACHED: NO
Majority Prediction: Monocytes (2/4 models)
Consensus Proportion: 0.5 (50% agreement)
Entropy: 1.39 (high uncertainty)
Confidence Level: Low

There is disagreement between myeloid cell types. Discussion needed to resolve.
""",
    
    "partial_consensus": """
Consensus Check Results:

Predictions received:
1. "B cells" (GPT-4o)
2. "Memory B cells" (Claude-3.5)  
3. "B cells (activated)" (Gemini)
4. "Plasma cell precursors" (DeepSeek)

Analysis:
- All models identify B cell lineage
- Disagreement on activation/differentiation state

Consensus Reached: PARTIAL
Majority Prediction: B cells
Consensus Proportion: 0.75
Entropy: 0.81
Requires further discussion for specific subtype.
"""
}

# Sample discussion round responses
DISCUSSION_RESPONSES = {
    "round_1": """
Discussion Round 1 - Cluster 5 Analysis

Models participating: claude-3-5-sonnet, gpt-4o, gemini-1.5-pro

Initial predictions showed disagreement:
- Claude: "Dendritic cells"
- GPT-4: "Monocytes" 
- Gemini: "Macrophages"

**Claude's argument:**
The high expression of CD1C and FCER1A strongly suggests conventional dendritic cells (cDC2). 
These markers are specific to the cDC2 subset and not typically expressed by monocytes or macrophages.

**GPT-4's counter-argument:**
While CD1C is present, the expression of CD14 and absence of CD16 points toward classical monocytes.
The FCER1A could be due to activation state rather than cell type identity.

**Gemini's perspective:**  
The gene expression pattern shows markers of both monocytes and tissue-resident macrophages.
High CD68 and low CD16 suggest these might be alternatively activated macrophages.

Consensus not reached. Proceeding to round 2.
""",
    
    "round_2_final": """
Discussion Round 2 - Final Resolution

After reviewing additional evidence:

**Claude (revised):** 
Upon closer examination of the full gene expression profile, I now see significant CD14 expression 
alongside the dendritic cell markers. This suggests a mixed population or transitioning cells.

**GPT-4 (maintains):**
The CD14+/CD16- profile with moderate HLA-DR expression is consistent with classical monocytes.
The dendritic cell markers might reflect cells in transition.

**Gemini (updated):**
Given the tissue context (peripheral blood), these are likely circulating monocytes rather than 
tissue macrophages. The dendritic markers could indicate pre-DC commitment.

**Final Consensus:** Classical Monocytes
**Reasoning:** Predominant CD14+/CD16- expression pattern in peripheral blood context
**Confidence:** Medium - some markers suggest transitional state
**Agreement Level:** 3/3 models now agree on monocyte identity
""",
    
    "complex_discussion": """
Multi-Round Discussion Summary - Controversial Cluster 12

=== ROUND 1 ===
Participants: claude-sonnet-4, gpt-4o, gemini-2.0-flash, qwen-max

Initial disagreement:
- Claude: "Plasma cells" (based on IGHG1, JCHAIN)
- GPT-4: "Activated B cells" (CD27+, high MKI67)
- Gemini: "Plasmablasts" (intermediate markers)
- Qwen: "Memory B cells" (CD27+, but low Ig secretion)

=== ROUND 2 ===
After presenting additional context about tissue origin (lymph node):

Claude (revised): "Actually, the proliferation markers suggest these might be plasmablasts"
GPT-4 (maintains): "The Ki-67 expression supports activated B cells in germinal center"
Gemini (supports): "Plasmablasts make sense - intermediate between B cells and plasma cells"
Qwen (updated): "Could be short-lived plasma cells based on tissue context"

=== ROUND 3 - CONSENSUS ===
Final agreement reached:

**CONSENSUS: Plasmablasts**

Reasoning:
- High immunoglobulin production (plasma cell feature)
- Retained proliferation markers (B cell feature)  
- Lymph node tissue context supports germinal center reaction
- All models now agree on transitional B cell -> plasma cell state

Confidence: High (4/4 models agree)
Final entropy: 0.0 (perfect consensus)
"""
}

# Sample error and edge cases
ERROR_RESPONSES = {
    "empty_response": "",
    
    "partial_response": """
Based on the markers provided, I can identify:
0: T cells
1: B cells  
2: Monocytes
""",  # Missing clusters
    
    "malformed_response": """
Cluster analysis results:
- Some kind of immune cells in cluster 0
- Cluster 1 has interesting pattern but unclear
- Need more data for cluster 2
- Cluster 3: ???
- Various cell types present
""",
    
    "conflicting_response": """
0: T cells... wait, actually these look more like NK cells
1: Definitely B cells, no question
2: Could be monocytes or dendritic cells, hard to tell
3: T cells again? Or maybe different T cell subset
4: Plasma cells (high confidence)
5: Not sure - need to review
""",
    
    "technical_error_response": """
ERROR: Model timeout during processing
Partial results available:
Cluster 0: CD8+ T cells
Connection lost...
Unable to complete analysis for remaining clusters.
"""
}

# Test case configurations
TEST_CASES = [
    {
        "name": "Standard_GPT4_Annotation",
        "text": CELL_TYPE_RESPONSES["openai_gpt4_response"],
        "expected_clusters": 9,
        "format": "simple_colon_format",
        "difficulty": "easy"
    },
    {
        "name": "Detailed_Claude_Annotation", 
        "text": CELL_TYPE_RESPONSES["claude_response"],
        "expected_clusters": 9,
        "format": "detailed_description",
        "difficulty": "medium"
    },
    {
        "name": "Mixed_Format_Response",
        "text": CELL_TYPE_RESPONSES["mixed_format_response"],
        "expected_clusters": 9,
        "format": "mixed",
        "difficulty": "hard"
    },
    {
        "name": "Inconsistent_Format",
        "text": CELL_TYPE_RESPONSES["inconsistent_format_response"],
        "expected_clusters": 9,
        "format": "inconsistent",
        "difficulty": "very_hard"
    },
    {
        "name": "Consensus_Reached",
        "text": CONSENSUS_RESPONSES["consensus_reached"],
        "expected_fields": ["consensus_reached", "majority_prediction", "consensus_proportion"],
        "format": "consensus_analysis",
        "difficulty": "medium"
    },
    {
        "name": "Complex_Discussion",
        "text": DISCUSSION_RESPONSES["complex_discussion"],
        "expected_fields": ["round", "participants", "consensus"],
        "format": "discussion_log",
        "difficulty": "very_hard"
    }
]

# Schemas for different types of extractions
EXTRACTION_SCHEMAS = {
    "cell_type_annotation": {
        "type": "object",
        "properties": {
            "annotations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cluster_id": {"type": "string"},
                        "cell_type": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                        "markers_mentioned": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["cluster_id", "cell_type"]
                }
            },
            "total_clusters": {"type": "integer"},
            "tissue_context": {"type": "string"}
        },
        "required": ["annotations"]
    },
    
    "consensus_result": {
        "type": "object", 
        "properties": {
            "consensus_reached": {"type": "boolean"},
            "majority_prediction": {"type": "string"},
            "consensus_proportion": {"type": "number", "minimum": 0, "maximum": 1},
            "entropy": {"type": "number", "minimum": 0},
            "participating_models": {
                "type": "array",
                "items": {"type": "string"}
            },
            "individual_predictions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "prediction": {"type": "string"}
                    },
                    "required": ["model", "prediction"]
                }
            }
        },
        "required": ["consensus_reached", "majority_prediction"]
    },
    
    "discussion_round": {
        "type": "object",
        "properties": {
            "round_number": {"type": "integer"},
            "cluster_id": {"type": "string"},
            "participating_models": {
                "type": "array",
                "items": {"type": "string"}
            },
            "model_responses": {
                "type": "array",
                "items": {
                    "type": "object", 
                    "properties": {
                        "model": {"type": "string"},
                        "argument": {"type": "string"},
                        "cell_type_prediction": {"type": "string"},
                        "confidence": {"type": "string"}
                    },
                    "required": ["model", "argument"]
                }
            },
            "consensus_achieved": {"type": "boolean"},
            "final_prediction": {"type": "string"}
        },
        "required": ["round_number", "model_responses"]
    }
}

# Evaluation metrics for comparison
EVALUATION_METRICS = {
    "accuracy": {
        "description": "How accurately the extraction identifies the correct information",
        "weight": 0.4
    },
    "completeness": {
        "description": "How completely all expected information is extracted",
        "weight": 0.3
    },
    "consistency": {
        "description": "How consistent the extraction format is across samples",
        "weight": 0.2
    },
    "performance": {
        "description": "Speed and efficiency of the extraction process",
        "weight": 0.1
    }
}


def get_sample_by_type(response_type: str) -> Dict[str, str]:
    """Get sample responses by type."""
    samples = {
        "cell_type": CELL_TYPE_RESPONSES,
        "consensus": CONSENSUS_RESPONSES,
        "discussion": DISCUSSION_RESPONSES,
        "error": ERROR_RESPONSES
    }
    return samples.get(response_type, {})


def get_test_cases_by_difficulty(difficulty: str) -> List[Dict[str, Any]]:
    """Filter test cases by difficulty level."""
    return [case for case in TEST_CASES if case.get("difficulty") == difficulty]


def get_schema_by_type(schema_type: str) -> Dict[str, Any]:
    """Get extraction schema by type."""
    return EXTRACTION_SCHEMAS.get(schema_type, {})


if __name__ == "__main__":
    # Example usage
    print("Sample responses loaded successfully!")
    print(f"Available cell type responses: {len(CELL_TYPE_RESPONSES)}")
    print(f"Available consensus responses: {len(CONSENSUS_RESPONSES)}")  
    print(f"Available discussion responses: {len(DISCUSSION_RESPONSES)}")
    print(f"Total test cases: {len(TEST_CASES)}")
    print(f"Available schemas: {list(EXTRACTION_SCHEMAS.keys())}")