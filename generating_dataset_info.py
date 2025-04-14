import json
import os
import pandas as pd
import numpy as np

# Helper functions to get property units and descriptions
def get_property_unit(prop):
    """Return the unit for a given property."""
    units = {
        "band_gap": "eV",
        "formation_energy_per_atom": "eV/atom",
        "energy_above_hull": "eV/atom",
        # Add more properties as needed
    }
    return units.get(prop, "")

def get_property_description(prop):
    """Return description for a given property."""
    descriptions = {
        "band_gap": "Electronic band gap calculated using DFT-PBE",
        "formation_energy_per_atom": "Formation energy per atom",
        "energy_above_hull": "Energy above convex hull",
        # Add more descriptions as needed
    }
    return descriptions.get(prop, "")

# Load your QMOF data
# Assuming you have qmof.json loaded as a JSON file
with open('qmof.json', 'r') as f:
    qmof_data = json.load(f)

# Or if you're using the CSV
# qmof_df = pd.read_csv('qmof.csv')

# Determine the target property you want to use
# For example, if you want to predict band gaps:
target_property = "band_gap"  # Change this to your desired property

# Calculate dataset splits (80% train, 10% val, 10% test)
total_count = len(qmof_data)
train_count = int(0.8 * total_count)
val_count = int(0.1 * total_count)
test_count = total_count - train_count - val_count

# Create the dataset_info.json content
dataset_info = {
    "dataset_name": "qmof",
    "authors": ["QMOF Database Contributors"],
    "references": ["https://github.com/arosen93/QMOF"],
    "data_types": ["dft_3d"],
    "target_property": target_property,
    "n_total": total_count,
    "n_train": train_count,
    "n_val": val_count,
    "n_test": test_count,
    "property_unit": get_property_unit(target_property),  # Function to determine unit
    "property_description": get_property_description(target_property),  # Function to get description
    "data_available": [target_property],
    "structure_format": "cif"
}

# Write the dataset_info.json file
with open('dataset_info.json', 'w') as f:
    json.dump(dataset_info, f, indent=2)

print(f"Created dataset_info.json with {total_count} structures")
print(f"Train: {train_count}, Validation: {val_count}, Test: {test_count}")