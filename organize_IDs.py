import os
import shutil
import pandas as pd
import json
import zipfile
from pathlib import Path


# Define the function before it's used
def find_matching_structure(directory, structure_id):
    """
    Find a structure file that matches the given ID.
    This function needs to be customized based on how your files are named.
    """
    # Direct match
    direct_match = os.path.join(directory, f"{structure_id}.cif")
    if os.path.exists(direct_match):
        return direct_match

    # Search for files with the ID in the filename
    for filename in os.listdir(directory):
        if filename.endswith('.cif') and structure_id in filename:
            return os.path.join(directory, filename)

    # If no match is found
    return None


# Define paths
base_dir = "/Users/shizhenli/Documents/Aixelo/PotNet"  # Your PotNet directory
id_prop_path = os.path.join(base_dir, "id_prop.csv")
structures_zip_path = None  # Path to your relaxed_structures.zip if you have it
structures_dir_path = "/Users/shizhenli/Documents/Aixelo/PotNet_on_QMOF/relaxed_structures"  # Path to your extracted structure files

# Create a structures directory if it doesn't exist
structures_dir = os.path.join(base_dir, "structures")
if not os.path.exists(structures_dir):
    os.makedirs(structures_dir)

# Read the id_prop.csv to get structure IDs
df = pd.read_csv(id_prop_path)
structure_ids = df['id'].tolist()

print(f"Found {len(structure_ids)} structure IDs in id_prop.csv")

# Option 1: Extract structures from a zip file
if structures_zip_path and os.path.exists(structures_zip_path):
    print(f"Extracting structures from {structures_zip_path}...")
    with zipfile.ZipFile(structures_zip_path, 'r') as zip_ref:
        # Extract to a temporary directory
        temp_dir = os.path.join(base_dir, "temp_structures")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        zip_ref.extractall(temp_dir)

        # Move and rename files to match IDs
        files_processed = 0
        for structure_id in structure_ids:
            # This assumes your zip contains files that can be mapped to your IDs
            # You might need to adjust the filename patterns based on how your files are named
            source_path = find_matching_structure(temp_dir, structure_id)
            if source_path:
                dest_path = os.path.join(structures_dir, f"{structure_id}.cif")
                shutil.copy2(source_path, dest_path)
                files_processed += 1

        print(f"Processed {files_processed} out of {len(structure_ids)} structures")

        # Clean up
        shutil.rmtree(temp_dir)

# Option 2: Copy from an existing directory of structures
elif structures_dir_path and os.path.exists(structures_dir_path):
    print(f"Copying structures from {structures_dir_path}...")
    files_processed = 0
    for structure_id in structure_ids:
        # This assumes your directory contains files that can be mapped to your IDs
        source_path = find_matching_structure(structures_dir_path, structure_id)
        if source_path:
            dest_path = os.path.join(structures_dir, f"{structure_id}.cif")
            shutil.copy2(source_path, dest_path)
            files_processed += 1

    print(f"Processed {files_processed} out of {len(structure_ids)} structures")

# Option 3: Create a script to extract structures from qmof_structure_data.json
else:
    qmof_structure_json = os.path.join(base_dir, "qmof_structure_data.json")
    if os.path.exists(qmof_structure_json):
        print(f"Extracting structures from {qmof_structure_json}...")

        # This part would need pymatgen to convert structure objects to CIF files
        try:
            import pymatgen as pmg
            from pymatgen.io.cif import CifWriter

            with open(qmof_structure_json, 'r') as f:
                structure_data = json.load(f)

            files_processed = 0
            for structure_id in structure_ids:
                if structure_id in structure_data:
                    # Convert pymatgen structure dict to Structure object
                    structure_dict = structure_data[structure_id]
                    structure = pmg.core.Structure.from_dict(structure_dict)

                    # Write to CIF
                    cif_path = os.path.join(structures_dir, f"{structure_id}.cif")
                    CifWriter(structure).write_file(cif_path)
                    files_processed += 1

            print(f"Processed {files_processed} out of {len(structure_ids)} structures")

        except ImportError:
            print("Pymatgen is required to convert structure data to CIF files.")
            print("Please install it with: pip install pymatgen")
    else:
        print("ERROR: Could not find structure files.")
        print("Please specify the path to either:")
        print("1. relaxed_structures.zip")
        print("2. A directory containing extracted structure files")
        print("3. qmof_structure_data.json")

print("\nNext steps:")
print("1. Check if the structure files were properly organized in the 'structures' directory")
print("2. Ensure your potnet.yaml has:")
print("   dataset: dft_3d")
print("   target: target")
print("3. Run training with:")
print(f"   python main.py --config configs/potnet.yaml --output_dir ./results --data_root {base_dir}")