import nbformat as nbf
import os

notebook_path = r"C:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\HybridNet_Implementation.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Define the new code for the DatasetManager cell (index 4 in the previous view_file output)
new_code = """import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DatasetManager:
    def __init__(self, name, mode, n_samples=5000):
        self.name = name
        self.mode = mode
        self.n_samples = n_samples
        self.base_dir = Path('./datasets')
        self.base_dir.mkdir(exist_ok=True)
        self.local_cic_path = self.base_dir / 'CICIOT23/train/train.csv'

    def load_data(self):
        if self.name == 'Simulation': return self._load_synthetic()
        elif self.name == 'CIC-IoT-2023': 
            path = self._check_download_cic()
            return self._process_cic_iot(path)
        elif self.name == 'Edge-IIoTset': 
            path = self._check_download_edge()
            return self._process_edge_iiot(path)
        else: raise ValueError(f'Unknown dataset')

    def _load_synthetic(self):
        X = np.random.randn(self.n_samples, 64)
        if self.mode == 'binary': y = np.random.randint(0, 2, self.n_samples)
        elif self.mode == 'multiclass': y = np.random.randint(0, 8, self.n_samples)
        else: y = np.random.rand(self.n_samples, 2)
        return X, y

    def _check_download_cic(self):
        # Priority 1: Check for locally added full dataset
        if self.local_cic_path.exists():
            print(f"Using local full dataset: {self.local_cic_path}")
            return self.local_cic_path
        
        # Priority 2: Fallback to sample (if available) or download
        url = 'https://raw.githubusercontent.com/mohamedamineferrag/CICIoT2023/main/sample/sampled_data.csv'
        path = self.base_dir / 'cic_sample.csv'
        if not path.exists():
            try:
                print("Downloading CIC-IoT-2023 sample...")
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"Download failed: {e}. Please ensure datasets/CICIOT23 exists.")
        return path

    def _check_download_edge(self):
        url = 'https://raw.githubusercontent.com/mohamedamineferrag/Edge-IIoTset/main/Sample_Dataset/DNN-EdgeIIoT-dataset.csv'
        path = self.base_dir / 'edge_sample.csv'
        if not path.exists(): urllib.request.urlretrieve(url, path)
        return path

    def _process_cic_iot(self, path):
        # Use nrows to optimize loading for large files
        df = pd.read_csv(path, nrows=self.n_samples)
        
        # Extract numeric features and pad to 64
        X = df.select_dtypes(include=[np.number]).values
        if X.shape[1] > 64: X = X[:, :64]
        if X.shape[1] < 64: X = np.pad(X, ((0, 0), (0, 64-X.shape[1])), 'constant')
        
        # Robust label mapping
        label_col = df.columns[-1]
        if self.mode == 'binary':
            # Handle both 'Benign' and 'BenignTraffic' variations
            y = (~df[label_col].astype(str).str.contains('Benign', case=False)).astype(int).values
        else:
            y = df[label_col].astype('category').cat.codes.values
        return X, y

    def _process_edge_iiot(self, path):
        df = pd.read_csv(path, nrows=self.n_samples)
        X = df.select_dtypes(include=[np.number]).values[:, :64]
        if X.shape[1] < 64: X = np.pad(X, ((0, 0), (0, 64-X.shape[1])), 'constant')
        if self.mode == 'binary': y = (df.iloc[:, -1] != 'Normal').astype(int).values
        else: y = df.iloc[:, -1].astype('category').cat.codes.values
        return X, y"""

# Update the relevant cell
nb.cells[4].source = new_code

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook updated successfully.")
