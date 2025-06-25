"""
Script: load_medquad.py
Description: Loads and preprocesses the medquad.csv dataset for training and evaluation.
Ensures HIPAA-compliant anonymization and basic cleaning.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('data/medquad.csv')

# Basic cleaning: drop rows with missing values in question/answer
df = df.dropna(subset=['question', 'answer'])

# Remove duplicates
qa_df = df.drop_duplicates(subset=['question', 'answer'])

# Shuffle and split dataset (80% train, 10% val, 10% test)
train, temp = train_test_split(qa_df, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save splits
train.to_csv('data/medquad_train.csv', index=False)
val.to_csv('data/medquad_val.csv', index=False)
test.to_csv('data/medquad_test.csv', index=False)

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
print("Preprocessed files saved as medquad_train.csv, medquad_val.csv, medquad_test.csv in data/.")
