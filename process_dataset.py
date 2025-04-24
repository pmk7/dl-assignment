import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Define the root directory of your dataset
DATASET_ROOT = Path("face_age")

# Function to convert numeric age to age group
def get_age_group(age):
    age = int(age)
    if age <= 12:
        return 'child'
    elif age <= 19:
        return 'teen'
    elif age <= 30:
        return 'youth'
    elif age <= 45:
        return 'mid'
    elif age <= 60:
        return 'mature'
    else:
        return 'older'

# Class label mapping
label_to_index = {
    'child': 0,
    'teen': 1,
    'youth': 2,
    'mid': 3,
    'mature': 4,
    'older': 5
}

# Collect image data
data = []
for folder in DATASET_ROOT.iterdir():
    if folder.is_dir() and folder.name.isdigit():
        age = int(folder.name)
        for img_file in folder.iterdir():
            data.append({
                "filepath": str(img_file),
                "age": age,
                "age_group": get_age_group(age)
            })

df = pd.DataFrame(data)

# Remove infants under age 4
df = df[df['age'] >= 4].reset_index(drop=True)

# Split into train/val/test
train_val_df, test_df = train_test_split(df, test_size=0.10, random_state=50, stratify=df['age_group'])
train_df, val_df = train_test_split(train_val_df, test_size=0.10, random_state=50, stratify=train_val_df['age_group'])

# Add labels to all three splits
for split_df in [train_df, val_df, test_df]:
    split_df['label'] = split_df['age_group'].map(label_to_index)

# Save outputs
output_dir = Path("processed_csvs")
output_dir.mkdir(exist_ok=True)

# train_df.to_csv(output_dir / "train_filtered_female.csv", index=False)
# val_df.to_csv(output_dir / "val_filtered_female.csv", index=False)
# test_df.to_csv(output_dir / "test_filtered.csv", index=False)

# print("\nSaved all filtered CSVs with label column:")
# print(train_df[['age_group', 'label']].drop_duplicates().sort_values(by='label'))

# Optional: Create Part 2 blocks
non_test_df = pd.concat([train_df, val_df]).reset_index(drop=True)
block1_df, block2_df = train_test_split(non_test_df, test_size=0.5, random_state=42, shuffle=True)

block1_df.to_csv(output_dir / "block1_autoencoder.csv", index=False)
block2_df.to_csv(output_dir / "block2_classification.csv", index=False)

print("\nBlock 1 & 2 CSVs created for Part 2")

# csv_dir = Path("processed_csvs")

# for csv_file in csv_dir.glob("*.csv"):
#     df = pd.read_csv(csv_file)
#     df['filepath'] = df['filepath'].apply(lambda x: '/'.join(Path(x).parts[-3:]) if 'face_age' in x else x)
#     df.to_csv(csv_file, index=False)
#     print(f"Cleaned: {csv_file}")