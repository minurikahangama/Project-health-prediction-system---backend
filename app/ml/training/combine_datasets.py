import pandas as pd
from collections import Counter

dfs = []

# Load SEntiMoji dataset
try:
    df1 = pd.read_csv('app/ml/training/dataset_sentimoji.csv')
    df1 = df1[df1['label'].notna()]
    df1['label'] = df1['label'].astype(int)
    dfs.append(df1)
    print(f"Loaded SEntiMoji: {len(df1)} rows")
except Exception as e:
    print(f"Could not load SEntiMoji: {e}")

if not dfs:
    print("ERROR: No datasets found")
    exit()

# Combine
combined = pd.concat(dfs, ignore_index=True)
combined = combined[['text','label']].dropna()
combined['text'] = combined['text'].astype(str).str.strip()
combined = combined.drop_duplicates(subset='text')
combined = combined[combined['text'].str.len().between(15, 300)]

print(f"\nTotal before balancing: {len(combined)}")
print(f"Label counts: {Counter(combined['label'].tolist())}")

# Balance to 150 per class
target = 150
balanced = combined.groupby('label').apply(
    lambda x: x.sample(min(len(x), target), random_state=42)
).reset_index(drop=True)

balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nAfter balancing: {len(balanced)} rows")
print(f"Label counts: {Counter(balanced['label'].tolist())}")

# Save
balanced.to_csv('app/ml/training/phps_training_final.csv', index=False)
print("\nSaved: app/ml/training/phps_training_final.csv")
print("This file is ready to upload to Google Colab for RoBERTa training")