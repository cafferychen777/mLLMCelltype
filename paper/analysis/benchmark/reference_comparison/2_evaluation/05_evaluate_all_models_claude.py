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

# Define model columns to evaluate
MODEL_COLUMNS = [
    'initial_claude-3-5-sonnet-latest',
    'initial_claude-3-5-haiku-latest',
    'initial_gemini-1.5-pro',
    'initial_gemini-2.0-flash-exp',
    'initial_qwen-max-2025-01-25',
    'initial_gpt-4o',
    'final_consensus'
]

# System prompt
SYSTEM_PROMPT = """You are a cell type annotation expert. You will evaluate the match between reference cell type names and consensus annotations based on the following criteria:

Full match (1.0): Identical terminology and Cell Ontology (CL) terms
Partial match (0.5): Shared hierarchical relationships or aligned broad categories
Mismatch (0.0): Annotations with no shared CL ancestry or divergent classifications

Please output only the score (1.0, 0.5, or 0.0) without any explanation."""

def evaluate_match_batch(reference_names, prediction_names):
    """Evaluate a batch of matches using Claude"""
    scores = []
    for ref, pred in zip(reference_names, prediction_names):
        try:
            prompt = f"Please evaluate the match between these cell type annotations:\n\nReference: {ref}\nPrediction: {pred}\n\nOutput only a single number (1.0, 0.5, or 0.0) based on these criteria:\n- 1.0 for identical terminology and Cell Ontology (CL) terms\n- 0.5 for shared hierarchical relationships or aligned broad categories\n- 0.0 for annotations with no shared CL ancestry or divergent classifications"

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
                    print(f"Invalid score {score} for {ref} vs {pred}, defaulting to 0.0")
                    score = 0.0
            except ValueError:
                print(f"Could not parse score '{score_text}' for {ref} vs {pred}, defaulting to 0.0")
                score = 0.0

            scores.append(score)

        except Exception as e:
            print(f"Error evaluating {ref} vs {pred}: {e}")
            scores.append(0.0)

    return scores

def process_csv_file(file_path):
    """Process a single CSV file and add match scores for all models"""
    try:
        # Check if output file already exists
        output_path = OUTPUT_DIR / f"{file_path.stem}_all_models_match.csv"
        if output_path.exists():
            print(f"Skipping {file_path.name}: Match file already exists at {output_path}")
            return

        df = pd.read_csv(file_path)

        # Verify required columns exist
        required_columns = ['reference_name'] + MODEL_COLUMNS
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Skipping {file_path.name}: Missing columns: {missing_columns}")
            return

        # Create a dictionary to store results for each model
        model_scores = {model: [] for model in MODEL_COLUMNS}

        # Process in batches of 10 for each model
        batch_size = 10

        for model in MODEL_COLUMNS:
            print(f"\nProcessing {model} for {file_path.name}")
            all_scores = []

            for i in tqdm(range(0, len(df), batch_size), desc=f"Processing {model}"):
                batch_df = df.iloc[i:i+batch_size]
                scores = evaluate_match_batch(
                    batch_df['reference_name'].tolist(),
                    batch_df[model].tolist()
                )
                all_scores.extend(scores)

            # Add scores to the dictionary
            model_scores[model] = all_scores

            # Calculate and print average score for this model
            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
            print(f"Average score for {model}: {avg_score:.3f}")

        # Add all scores to dataframe
        for model in MODEL_COLUMNS:
            df[f'{model}_match_score'] = model_scores[model]

        # Create summary rows for each model
        summary_rows = []
        for model in MODEL_COLUMNS:
            scores = model_scores[model]
            avg_score = sum(scores) / len(scores) if scores else 0
            full_matches = sum(1 for s in scores if s == 1.0)
            partial_matches = sum(1 for s in scores if s == 0.5)
            mismatches = sum(1 for s in scores if s == 0.0)

            summary_row = {
                'cluster_id': 'Summary',
                'reference_name': f'{model} Average',
                model: f'Score: {avg_score:.3f} (Full: {full_matches}, Partial: {partial_matches}, Miss: {mismatches})',
                f'{model}_match_score': avg_score
            }
            summary_rows.append(summary_row)

        # Add summary rows to dataframe
        summary_df = pd.DataFrame(summary_rows)
        df = pd.concat([df, summary_df], ignore_index=True)

        # Save results
        df.to_csv(output_path, index=False)
        print(f"\nSaved results to {output_path}")

    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")

def main():
    # Only process files that end with _results.csv and don't have _with_match in their name
    for file_path in RESULTS_DIR.glob("*_results.csv"):
        if "_with_match" not in file_path.name and "_all_models" not in file_path.name:
            process_csv_file(file_path)

if __name__ == "__main__":
    main()
