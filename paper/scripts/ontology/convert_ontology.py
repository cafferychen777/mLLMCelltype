import os
import json
import urllib.request
import pronto

# Input and output paths
obo_file = "external/reference_data/cl.obo"
output_dir = "data/thymus/ontology"

# Download Cell Ontology if not present
if not os.path.exists(obo_file):
    os.makedirs(os.path.dirname(obo_file), exist_ok=True)
    print("Downloading Cell Ontology (cl.obo)...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/obophenotype/cell-ontology/master/cl.obo",
        obo_file,
    )

# Load the ontology
ont = pronto.Ontology(obo_file)

# Create the graph structure
graph = {
    "graphs": [{
        "nodes": [],
        "edges": []
    }]
}

# Add nodes
for term in ont.terms():
    if term.id.startswith("CL:"):
        node = {
            "id": term.id,
            "type": "CLASS",
            "lbl": term.name,
            "meta": {
                "definition": {"val": term.definition or ""},
                "comments": [str(comment) for comment in term.comments]
            }
        }
        graph["graphs"][0]["nodes"].append(node)

# Add edges
for term in ont.terms():
    if term.id.startswith("CL:"):
        for parent in term.superclasses(distance=1):
            if parent.id.startswith("CL:"):
                edge = {
                    "sub": term.id,
                    "pred": "is_a",
                    "obj": parent.id
                }
                graph["graphs"][0]["edges"].append(edge)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save the JSON file
output_file = os.path.join(output_dir, "cl_popv.json")
with open(output_file, "w") as f:
    json.dump(graph, f, indent=4)

print(f"Created {output_file}")
