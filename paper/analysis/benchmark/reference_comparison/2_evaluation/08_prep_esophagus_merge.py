import pandas as pd
from pathlib import Path

# Define paths
RESULTS_DIR = Path("results/benchmark/reference_comparison/2_evaluation")

def merge_esophagus_data():
    # Read both esophagus datasets
    mucosa_df = pd.read_csv(RESULTS_DIR / "GTEx_esophagus_mucosa_results_with_match.csv")
    muscularis_df = pd.read_csv(RESULTS_DIR / "GTEx_esophagus_muscularis_results_with_match.csv")

    # Get the counts based on match_score
    mucosa_full_matches = len(mucosa_df[mucosa_df['match_score'] == 1.0])
    mucosa_partial_matches = len(mucosa_df[(mucosa_df['match_score'] > 0) & (mucosa_df['match_score'] < 1)])
    muscularis_full_matches = len(muscularis_df[muscularis_df['match_score'] == 1.0])
    muscularis_partial_matches = len(muscularis_df[(muscularis_df['match_score'] > 0) & (muscularis_df['match_score'] < 1)])

    # Calculate total matches
    total_full_matches = mucosa_full_matches + muscularis_full_matches
    total_partial_matches = mucosa_partial_matches + muscularis_partial_matches

    print(f"Mucosa: {mucosa_full_matches} full matches, {mucosa_partial_matches} partial matches")
    print(f"Muscularis: {muscularis_full_matches} full matches, {muscularis_partial_matches} partial matches")
    print(f"Total: {total_full_matches} full matches, {total_partial_matches} partial matches")

    # Combine the dataframes
    combined_df = pd.concat([mucosa_df, muscularis_df], ignore_index=True)

    # Save to new file
    output_path = RESULTS_DIR / "GTEx_esophagus_results_with_match.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\nCombined esophagus data saved to {output_path}")

    # Print suggestion for updating visualization script
    print("\nSuggested addition to visualization script:")
    print('"GTEx_esophagus_results_with_match.csv": ("Esophagus(Literature)(GTEx)", ' \
          f'{total_full_matches}, {total_partial_matches}),\n')

if __name__ == "__main__":
    merge_esophagus_data()
