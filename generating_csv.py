import pandas as pd
import numpy as np

# Read your current id_prop.csv file with headers
df = pd.read_csv('id_prop.csv')

# Rename the 'band_gap' column to 'target' as expected by PotNet
if 'band_gap' in df.columns:
    df = df.rename(columns={'band_gap': 'target'})

# Set random seed for reproducibility
np.random.seed(42)

# Assign train/val/test splits (80% train, 10% val, 10% test)
splits = np.random.choice(
    ['train', 'val', 'test'],
    size=len(df),
    p=[0.8, 0.1, 0.1]
)

# Add the split column to your dataframe
df['split'] = splits

# Save the properly formatted file
df.to_csv('id_prop.csv', index=False)

# Display statistics
print(f"Total structures: {len(df)}")
print(f"Train: {sum(df['split'] == 'train')}")
print(f"Validation: {sum(df['split'] == 'val')}")
print(f"Test: {sum(df['split'] == 'test')}")