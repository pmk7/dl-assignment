import pandas as pd
from deepface import DeepFace
from tqdm import tqdm
import os

# Load the CSV to filter
input_csv = "processed_csvs/train_filtered.csv"
output_csv = "processed_csvs/train_filtered_female.csv"

df = pd.read_csv(input_csv)
print(f"Loaded {len(df)} entries from {input_csv}")

female_rows = []
print(female_rows)

for _, row in tqdm(df.iterrows(), total=len(df)):
    image_path = row["filepath"]
    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=["gender"],
            enforce_detection=False,
        )
        print(result)

        dominant_gender = result[0]["dominant_gender"]
        print(dominant_gender)

        if dominant_gender == "Woman":
            row["gender"] = dominant_gender
            female_rows.append(row)

    except Exception as e:
        print(f"Skipping {image_path} due to error: {e}")

# Save new filtered DataFrame
female_df = pd.DataFrame(female_rows)
print(f"Filtered to {len(female_df)} female entries.")
female_df.to_csv(output_csv, index=False)
print(f"Saved to {output_csv}")