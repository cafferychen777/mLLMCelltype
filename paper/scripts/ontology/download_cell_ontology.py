#!/usr/bin/env python3
"""
Download and Process Cell Ontology (CL) for Cell Type Annotation

This script downloads the Cell Ontology OBO file and creates useful mappings
for cell type annotation workflows.

Reference: Adapted from YosefLab/popv-reproducibility
https://github.com/YosefLab/popv-reproducibility

Cell Ontology: http://obofoundry.org/ontology/cl.html

Usage:
    python download_cell_ontology.py [--output-dir ./ontology]
"""

import argparse
import os

import pandas as pd
import requests

# Cell Ontology download URL
CL_OBO_URL = "http://purl.obolibrary.org/obo/cl.obo"


def download_cell_ontology(output_dir):
    """
    Download the Cell Ontology OBO file.

    Parameters
    ----------
    output_dir : str
        Directory to save the OBO file

    Returns
    -------
    str
        Path to downloaded OBO file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "cl.obo")

    if os.path.exists(output_file):
        print(f"Cell Ontology already exists: {output_file}")
        return output_file

    print(f"Downloading Cell Ontology from: {CL_OBO_URL}")
    response = requests.get(CL_OBO_URL, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    written = 0

    with open(output_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            written += len(chunk)
            f.write(chunk)
            if total_size > 0:
                progress = int((written / total_size) * 100)
                print(f"\rProgress: {progress}%", end="")

    print(f"\nDownloaded: {output_file}")
    return output_file


def parse_cell_ontology(obo_file):
    """
    Parse Cell Ontology OBO file and create mappings.

    Parameters
    ----------
    obo_file : str
        Path to cl.obo file

    Returns
    -------
    name2id : dict
        Cell type name to ontology ID mapping
    id2name : dict
        Ontology ID to cell type name mapping
    hierarchy : list
        List of [child_id, parent_id] relationships
    """
    try:
        import obonet
    except ImportError:
        raise ImportError(
            "obonet is required for parsing OBO files. Install with: pip install obonet"
        )

    print(f"Parsing Cell Ontology: {obo_file}")
    graph = obonet.read_obo(obo_file)

    # Create ID to name mapping (only CL terms)
    id2name = {}
    for id_, data in graph.nodes(data=True):
        if "CL:" in id_ and data.get("name"):
            id2name[id_] = data["name"]

    # Create name to ID mapping
    name2id = {v: k for k, v in id2name.items()}

    print(f"Found {len(id2name)} cell type terms")

    # Extract hierarchical relationships (is_a)
    hierarchy = []
    for edge in graph.edges(data=True):
        child, parent = edge[0], edge[1]
        # Only keep CL to CL relationships
        if "CL:" in child and "CL:" in parent:
            hierarchy.append([child, parent])

    print(f"Found {len(hierarchy)} hierarchical relationships")

    return name2id, id2name, hierarchy


def save_mappings(name2id, id2name, hierarchy, output_dir):
    """
    Save ontology mappings to CSV files.

    Parameters
    ----------
    name2id : dict
        Cell type name to ontology ID mapping
    id2name : dict
        Ontology ID to cell type name mapping
    hierarchy : list
        List of [child_id, parent_id] relationships
    output_dir : str
        Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save name to ID mapping
    name2id_file = os.path.join(output_dir, "celltype_to_ontology_id.csv")
    pd.DataFrame(
        [{"cell_type": name, "ontology_id": id_} for name, id_ in name2id.items()]
    ).to_csv(name2id_file, index=False)
    print(f"Saved: {name2id_file}")

    # Save ID to name mapping
    id2name_file = os.path.join(output_dir, "ontology_id_to_celltype.csv")
    pd.DataFrame(
        [{"ontology_id": id_, "cell_type": name} for id_, name in id2name.items()]
    ).to_csv(id2name_file, index=False)
    print(f"Saved: {id2name_file}")

    # Save hierarchy
    hierarchy_file = os.path.join(output_dir, "cl.ontology")
    pd.DataFrame(hierarchy, columns=["child", "parent"]).to_csv(
        hierarchy_file, sep="\t", header=False, index=False
    )
    print(f"Saved: {hierarchy_file}")


def lookup_cell_type(query, name2id, id2name):
    """
    Look up a cell type in the ontology.

    Parameters
    ----------
    query : str
        Cell type name or ontology ID to look up
    name2id : dict
        Name to ID mapping
    id2name : dict
        ID to name mapping

    Returns
    -------
    dict or None
        Dictionary with 'name' and 'id' keys, or None if not found
    """
    # Check if query is an ID
    if query in id2name:
        return {"id": query, "name": id2name[query]}

    # Check if query is a name
    if query in name2id:
        return {"id": name2id[query], "name": query}

    # Try case-insensitive search
    query_lower = query.lower()
    for name, id_ in name2id.items():
        if name.lower() == query_lower:
            return {"id": id_, "name": name}

    # Try partial match
    matches = []
    for name, id_ in name2id.items():
        if query_lower in name.lower():
            matches.append({"id": id_, "name": name})

    if matches:
        print(f"Found {len(matches)} partial matches for '{query}':")
        for m in matches[:10]:
            print(f"  - {m['name']} ({m['id']})")
        return matches[0] if len(matches) == 1 else None

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download and process Cell Ontology for cell type annotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download and process Cell Ontology
    python download_cell_ontology.py --output-dir ./ontology

    # Look up a specific cell type
    python download_cell_ontology.py --output-dir ./ontology --lookup "T cell"

Reference:
    Cell Ontology: http://obofoundry.org/ontology/cl.html
    OBO Foundry: http://obofoundry.org/
        """,
    )
    parser.add_argument(
        "--output-dir",
        default="./ontology",
        help="Output directory for ontology files (default: ./ontology)",
    )
    parser.add_argument(
        "--lookup", default=None, help="Look up a specific cell type (optional)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if OBO file already exists",
    )

    args = parser.parse_args()

    # Download OBO file
    obo_file = os.path.join(args.output_dir, "cl.obo")
    if not args.skip_download or not os.path.exists(obo_file):
        obo_file = download_cell_ontology(args.output_dir)

    # Parse and save mappings
    name2id, id2name, hierarchy = parse_cell_ontology(obo_file)
    save_mappings(name2id, id2name, hierarchy, args.output_dir)

    # Look up cell type if requested
    if args.lookup:
        print(f"\nLooking up: '{args.lookup}'")
        result = lookup_cell_type(args.lookup, name2id, id2name)
        if result:
            print(f"Found: {result['name']} ({result['id']})")
        else:
            print("Not found")

    print("\nDone!")


if __name__ == "__main__":
    main()
