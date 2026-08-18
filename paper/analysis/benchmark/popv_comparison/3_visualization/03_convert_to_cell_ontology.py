import pandas as pd
import os
import sys

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

def load_hlca_cl_mapping():
    """Load the HLCA to Cell Ontology mapping"""
    mapping_file = os.path.join(
        project_root,
        "external/reference_data/cell_type_mapping_to_cellontology.csv"
    )

    if not os.path.exists(mapping_file):
        # Try alternative path
        mapping_file = os.path.join(
            project_root,
            "external/reference_data/cell_type_mapping_to_cellontology.csv"
        )

    print(f"Looking for mapping file at: {mapping_file}")

    # Read the mapping file
    mapping_df = pd.read_csv(mapping_file, header=0)

    # Create two mapping dictionaries
    name_mapping = {}  # HLCA name to Uberon term
    id_mapping = {}    # HLCA name to CL ID

    for _, row in mapping_df.iterrows():
        hlca_name = row['HLCA']
        uberon_term = row.iloc[1]  # Second column (Uberon term)
        cl_id = row.iloc[2]        # Third column (CL ID)

        if pd.notna(uberon_term):
            name_mapping[hlca_name] = uberon_term
        if pd.notna(cl_id):
            id_mapping[hlca_name] = cl_id.replace('_', ':')  # Convert CL_0010003 to CL:0010003

    return name_mapping, id_mapping

def main():
    # Load HLCA to Cell Ontology mapping
    print("Loading Cell Ontology mapping...")
    name_mapping, id_mapping = load_hlca_cl_mapping()

    # Get all unique HLCA annotations
    hlca_annotations = list(name_mapping.keys())

    # Create mapping report
    mapping_report = pd.DataFrame([
        {
            'HLCA_annotation': ann,
            'Cell_Ontology_term': name_mapping.get(ann, 'Unknown'),
            'Cell_Ontology_ID': id_mapping.get(ann, 'Unknown'),
        }
        for ann in hlca_annotations
    ])

    # Save results
    print("Saving results...")
    output_dir = os.path.join(project_root, "results/benchmark/popv_comparison/3_visualization")
    os.makedirs(output_dir, exist_ok=True)

    # Save the mapping report
    report_file = os.path.join(output_dir, "cl_mapping_report.csv")
    mapping_report.to_csv(report_file, index=False)

    # Print mapping statistics
    total_annotations = len(hlca_annotations)
    mapped_with_term = sum(1 for ann in hlca_annotations if ann in name_mapping)
    mapped_with_id = sum(1 for ann in hlca_annotations if ann in id_mapping)

    print(f"\nMapping statistics:")
    print(f"Total HLCA annotations: {total_annotations}")
    print(f"Mapped to Cell Ontology terms: {mapped_with_term} ({mapped_with_term/total_annotations*100:.1f}%)")
    print(f"Mapped to Cell Ontology IDs: {mapped_with_id} ({mapped_with_id/total_annotations*100:.1f}%)")

    # Print some examples
    print("\nExample mappings:")
    print(mapping_report.head())

    # Print terms without CL IDs
    no_cl_id = [ann for ann in hlca_annotations if ann not in id_mapping]
    if no_cl_id:
        print("\nTerms without Cell Ontology IDs:")
        print("\n".join(no_cl_id))

if __name__ == "__main__":
    main()
