import pandas as pd
import os
import glob
from difflib import SequenceMatcher

def string_similarity(a, b):
    """Calculate string similarity between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def process_final_consensus(row, file_name):
    """Process final_consensus column and compare with reference_name"""
    final_consensus = row['final_consensus']
    reference_name = row['reference_name']

    # If there's no slash in the final_consensus, return as is
    if '/' not in final_consensus:
        return final_consensus

    # Split the term and compare each part with reference_name
    terms = [term.strip() for term in final_consensus.split('/')]
    similarities = [string_similarity(term, reference_name) for term in terms]

    print('\nIn {}:'.format(file_name))
    print("  Found split term: '{}'".format(final_consensus))
    print("  Reference name: '{}'".format(reference_name))
    print("  Comparing similarities:")
    for term, similarity in zip(terms, similarities):
        print("    - '{}' -> similarity: {:.3f}".format(term, similarity))

    chosen_term = terms[similarities.index(max(similarities))]
    print("  → Choosing '{}' (highest similarity)".format(chosen_term))

    return chosen_term

def process_csv_files():
    # Get all results CSV files
    csv_files = glob.glob('results/benchmark/reference_comparison/2_evaluation/*results.csv')
    print("Found {} CSV files to process".format(len(csv_files)))

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        try:
            print('\nProcessing {}...'.format(file_name))

            # Read the CSV file
            df = pd.read_csv(file_path)
            print("  Loaded {} rows".format(len(df)))

            # Count how many terms contain '/'
            split_terms = df['final_consensus'].str.contains('/', na=False).sum()
            print("  Found {} terms containing '/'".format(split_terms))

            if split_terms > 0:
                # Process final_consensus column
                df['final_consensus'] = df.apply(lambda row: process_final_consensus(row, file_name), axis=1)

                # Save the processed file
                output_path = file_path.replace('.csv', '_processed.csv')
                df.to_csv(output_path, index=False)
                print("  ✓ Saved processed file as {}".format(os.path.basename(output_path)))
            else:
                print("  → No terms to process in this file, skipping")

        except Exception as e:
            print("  ✗ Error processing {}: {}".format(file_name, str(e)))

if __name__ == "__main__":
    print("Starting cell type term processing...")
    process_csv_files()
    print("\nProcessing complete!")
