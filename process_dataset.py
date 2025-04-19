import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Define the root directory of your dataset
DATASET_ROOT = Path("face_age")

df = pd.read_csv('processed_csvs/test.csv')
print(df.columns)
print(df.head())

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

# Collect data
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

# Remove babies under age 4
df = df[df['age'] >= 4].reset_index(drop=True)

# Shuffle and stratified split
train_val_df, test_df = train_test_split(df, test_size=0.10, random_state=50, stratify=df['age_group'])
train_df, val_df = train_test_split(train_val_df, test_size=0.10, random_state=50, stratify=train_val_df['age_group'])

# Save outputs with new filenames
output_dir = Path("processed_csvs")
output_dir.mkdir(exist_ok=True)

train_df.to_csv(output_dir / "train_filtered.csv", index=False)
val_df.to_csv(output_dir / "val_filtered.csv", index=False)
test_df.to_csv(output_dir / "test_filtered.csv", index=False)

print("Filtered CSVs created:")
print("- train_filtered.csv")
print("- val_filtered.csv")
print("- test_filtered.csv")