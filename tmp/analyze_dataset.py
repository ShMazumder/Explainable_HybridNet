import pandas as pd
import numpy as np

path = r"C:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\datasets\CICIOT23\train\train.csv"

# Load a small chunk
df = pd.read_csv(path, nrows=100)
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
print("\nUnique Labels (Sample):", df.iloc[:, -1].unique())
print("\nInfo:")
print(df.info())
