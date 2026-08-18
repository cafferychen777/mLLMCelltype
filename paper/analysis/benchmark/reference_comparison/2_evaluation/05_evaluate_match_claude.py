import anthropic
import pandas as pd
import os
from pathlib import Path
import json
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(".env"))

# Define paths
RESULTS_DIR = Path("results/benchmark/reference_comparison/2_evaluation")
OUTPUT_DIR = RESULTS_DIR

# Initialize Anthropic client
client = anthropic.Anthropic()

# System prompt
SYSTEM_PROMPT = """You are a cell type annotation expert. You will evaluate the match between reference cell type names and consensus annotations based on the following criteria:

Full match (1.0): Identical terminology and Cell Ontology (CL) terms
Partial match (0.5): Shared hierarchical relationships or aligned broad categories
Mismatch (0.0): Annotations with no shared CL ancestry or divergent classifications

Please output only the score (1.0, 0.5, or 0.0) without any explanation."""

def evaluate_match_batch(reference_names, consensus_names):
    """Evaluate a batch of matches using Claude"""
    scores = []
    for ref, cons in zip(reference_names, consensus_names):
        try:
            prompt = f"Please evaluate the match between these cell type annotations:\n\nReference: {ref}\nConsensus: {cons}\n\nOutput only a single number (1.0, 0.5, or 0.0) based on these criteria:\n- 1.0 for identical terminology and Cell Ontology (CL) terms\n- 0.5 for shared hierarchical relationships or aligned broad categories\n- 0.0 for annotations with no shared CL ancestry or divergent classifications"

            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract the score from response
            score_text = response.content[0].text.strip()
            try:
                score = float(score_text)
                if score not in [0.0, 0.5, 1.0]:
                    print(f"Invalid score {score} for {ref} vs {cons}, defaulting to 0.0")
                    score = 0.0
            except ValueError:
                print(f"Could not parse score '{score_text}' for {ref} vs {cons}, defaulting to 0.0")
                score = 0.0

            scores.append(score)

        except Exception as e:
            print(f"Error evaluating {ref} vs {cons}: {e}")
            scores.append(0.0)

    return scores

def process_csv_file(file_path):
    """Process a single CSV file and add match scores"""
    try:
        # Check if output file already exists
        output_path = OUTPUT_DIR / f"{file_path.stem}_with_match.csv"
        if output_path.exists():
            print(f"Skipping {file_path.name}: Match file already exists at {output_path}")
            return

        df = pd.read_csv(file_path)

        # Verify required columns exist
        if 'reference_name' not in df.columns or 'final_consensus' not in df.columns:
            print(f"Skipping {file_path.name} as it's missing required columns")
            return

        # Process in batches of 10
        batch_size = 10
        all_scores = []

        for i in tqdm(range(0, len(df), batch_size), desc=f"Processing {file_path.name}"):
            batch_df = df.iloc[i:i+batch_size]
            scores = evaluate_match_batch(
                batch_df['reference_name'].tolist(),
                batch_df['final_consensus'].tolist()
            )
            all_scores.extend(scores)

        # Add scores to dataframe
        df['match_score'] = all_scores

        # Calculate average score
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

        # Create a new row for the total score
        total_row = pd.DataFrame([{
            'cluster_id': 'Total',
            'reference_name': f'Average Score ({len(all_scores)} entries)',
            'final_consensus': '',
            'match_score': avg_score,
            **{col: '' for col in df.columns if col not in ['cluster_id', 'reference_name', 'final_consensus', 'match_score']}
        }])

        # Concatenate the original dataframe with the total row
        df = pd.concat([df, total_row], ignore_index=True)

        # Save results
        output_path = OUTPUT_DIR / f"{file_path.stem}_with_match.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path} with average score: {avg_score:.3f}")
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")

def main():
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate cell type annotation matches using Claude')
    parser.add_argument('files', nargs='+', help='CSV files to process (e.g., tissue_results.csv)')

    args = parser.parse_args()

    # Process specified CSV files
    for file_name in args.files:
        if not file_name.endswith('_results.csv'):
            file_name = f"{file_name}_results.csv"

        file_path = RESULTS_DIR / file_name
        if not file_path.exists():
            print(f"Error: File {file_path} does not exist")
            continue

        if "_with_match" in file_path.name:
            print(f"Skipping {file_path.name} as it already contains match scores")
            continue

        process_csv_file(file_path)

if __name__ == "__main__":
    main()
