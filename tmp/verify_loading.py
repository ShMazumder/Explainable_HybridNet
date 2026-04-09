import numpy as np
import pandas as pd
from pathlib import Path

# Mock the DatasetManager logic to verify it works with the real file
class Verifier:
    def __init__(self, n_samples=5000):
        self.n_samples = n_samples
        self.base_dir = Path(r"C:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\datasets")
        self.local_cic_path = self.base_dir / 'CICIOT23/train/train.csv'

    def verify_load(self, mode='binary'):
        print(f"Checking path: {self.local_cic_path}")
        if not self.local_cic_path.exists():
            print("Path does not exist!")
            return
        
        print(f"Loading {self.n_samples} samples (mode={mode})...")
        df = pd.read_csv(self.local_cic_path, nrows=self.n_samples)
        
        X = df.select_dtypes(include=[np.number]).values
        if X.shape[1] > 64: X = X[:, :64]
        if X.shape[1] < 64: X = np.pad(X, ((0, 0), (0, 64-X.shape[1])), 'constant')
        
        label_col = df.columns[-1]
        if mode == 'binary':
            y = (~df[label_col].astype(str).str.contains('Benign', case=False)).astype(int).values
        else:
            y = df[label_col].astype('category').cat.codes.values
            
        print(f"X shape: {X.shape}")
        print(f"y unique values: {np.unique(y)}")
        print(f"y sample: {y[:10]}")
        print("Verification successful!")

v = Verifier(n_samples=100)
v.verify_load(mode='binary')
print("-" * 20)
v.verify_load(mode='multiclass')
