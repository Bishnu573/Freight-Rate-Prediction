import pandas as pd

# Load the training data
df = pd.read_csv("data/train_test.csv")

print("=" * 60)
print("FIRST 5 ROWS")
print(df.head())

print("\n" + "=" * 60)
print("DATASET SHAPE")
print(df.shape)

print("\n" + "=" * 60)
print("COLUMN NAMES")
print(df.columns.tolist())

print("\n" + "=" * 60)
print("DATA TYPES")
print(df.dtypes)

print("\n" + "=" * 60)
print("MISSING VALUES")
print(df.isnull().sum())

print("\n" + "=" * 60)
print("SUMMARY")
print(df.describe(include="all"))