import numpy as np
import pandas as pd
from pathlib import Path

# Mock check of the new logic
base_dir = Path(r"C:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\datasets\CICIOT23")

for split in ['train', 'test', 'validation']:
    path = base_dir / split / f"{split}.csv"
    if path.exists():
        df = pd.read_csv(path, nrows=100)
        print(f"Split {split:10} | Rows: {len(df):4} | Columns: {len(df.columns):2} | Last row label: {df.iloc[-1, -1]}")
    else:
        print(f"Split {split:10} | NOT FOUND")

print("\nLogic check: All categorical labels should be mapped to consistent codes across splits.")
