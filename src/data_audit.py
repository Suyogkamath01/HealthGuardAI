import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/diabetic_data.csv")

print("=" * 50)
print("DATASET SHAPE")
print("=" * 50)
print(df.shape)

print("\n")

print("=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())

print("\n")

print("=" * 50)
print("COLUMN INFO")
print("=" * 50)
print(df.info())

print("\n")

print("=" * 50)
print("READMISSION COUNTS")
print("=" * 50)
print(df["readmitted"].value_counts())