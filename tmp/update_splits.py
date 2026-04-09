import json

notebook_path = r"C:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\HybridNet_Implementation.ipynb"

# --- 1. Update DatasetManager (Cell 4) ---
# Modified to handle splits, consistent labeling, and safe scaling.
dataset_manager_code = """import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class DatasetManager:
    def __init__(self, name, mode, n_samples=5000):
        self.name = name
        self.mode = mode
        self.n_samples = n_samples
        self.base_dir = Path('./datasets')
        self.base_dir.mkdir(exist_ok=True)
        self.scaler = MinMaxScaler()
        self.label_names = []
        # Pre-defined folder mapping for CICIOT23
        self.ciciot_dir = self.base_dir / 'CICIOT23'
        
    def load_split(self, split='train'):
        \"\"\"Loads a specific dataset split (train/test/validation).\"\"\"
        if self.name == 'Simulation':
            return self._load_synthetic()
        elif self.name == 'CIC-IoT-2023':
            path = self.ciciot_dir / split / f"{split}.csv"
            if not path.exists():
                print(f"Warning: Split {split} path {path} not found. Falling back to default.")
                return self._process_cic_iot(self._check_download_cic(), fit_scaler=(split=='train'))
            return self._process_cic_iot(path, fit_scaler=(split=='train'))
        elif self.name == 'Edge-IIoTset':
            path = self._check_download_edge()
            return self._process_edge_iiot(path)
        else:
            raise ValueError(f"Unknown dataset: {self.name}")

    def _process_cic_iot(self, path, fit_scaler=False):
        df = pd.read_csv(path, nrows=self.n_samples)
        
        # Extract numeric features and scale
        X = df.select_dtypes(include=[np.number]).values
        if fit_scaler:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
            
        # Pad to 64
        if X.shape[1] > 64: X = X[:, :64]
        if X.shape[1] < 64: X = np.pad(X, ((0, 0), (0, 64-X.shape[1])), 'constant')
        
        # Robust label mapping
        label_col = df.columns[-1]
        if self.mode == 'binary':
            y = (~df[label_col].astype(str).str.contains('Benign', case=False)).astype(int).values
        else:
            # For multiclass, we need consistent codes. 
            # We'll use pandas categorical with a fixed set of categories if it's the first time.
            y_series = df[label_col].astype('category')
            if not self.label_names or fit_scaler: # Re-initialize labels for training
                self.label_names = sorted(df[label_col].unique().tolist())
            
            # Ensure consistent mapping to index
            mapping = {name: i for i, name in enumerate(self.label_names)}
            y = df[label_col].map(lambda x: mapping.get(x, -1)).values
            
        return X, y

    def _load_synthetic(self):
        X = np.random.randn(self.n_samples, 64)
        if self.mode == 'binary': y = np.random.randint(0, 2, self.n_samples)
        elif self.mode == 'multiclass': y = np.random.randint(0, 8, self.n_samples)
        else: y = np.random.rand(self.n_samples, 2)
        return X, y

    def _check_download_cic(self):
        url = 'https://raw.githubusercontent.com/mohamedamineferrag/CICIoT2023/main/sample/sampled_data.csv'
        path = self.base_dir / 'cic_sample.csv'
        if not path.exists():
            try: urllib.request.urlretrieve(url, path)
            except: pass
        return path

    def _process_edge_iiot(self, path):
        df = pd.read_csv(path, nrows=self.n_samples)
        X = self.scaler.fit_transform(df.select_dtypes(include=[np.number]).values)[:, :64]
        if X.shape[1] < 64: X = np.pad(X, ((0, 0), (0, 64-X.shape[1])), 'constant')
        if self.mode == 'binary': y = (df.iloc[:, -1] != 'Normal').astype(int).values
        else:
            y_series = df.iloc[:, -1].astype('category')
            self.label_names = y_series.cat.categories.tolist()
            y = y_series.cat.codes.values
        return X, y
"""

# --- 2. Update Loading Cell (Cell 6) ---
loading_code = """# Load Splits via Manager
manager = DatasetManager(DATASET_TYPE, TASK_MODE, SAMPLE_SIZE)

# Efficiently load separate splits
print("Loading Training Split...")
X_train_raw, y_train_raw = manager.load_split('train')

print("Loading Testing Split...")
X_test_raw, y_test_raw = manager.load_split('test')

print("Loading Validation Split...")
X_val_raw, y_val_raw = manager.load_split('validation')

# Store names for XAI
if TASK_MODE == 'multiclass':
    print(f"Dataset contains {len(manager.label_names)} labels.")
"""

# --- 3. Update Training Cell (Cell 13) ---
# Removed train_test_split and updated loader creation.
training_code = """# --- Build Model ---
# Dynamically determine the number of classes for the current task
if TASK_MODE == 'regression':
    num_classes = 2 # Regression output (Magnitude, Probability)
else:
    num_classes = len(manager.label_names) if manager.label_names else len(np.unique(y_train_raw))
    print(f'Configured for {num_classes} classes.')

model = HybridNet(num_classes=num_classes, mode=TASK_MODE).to(device)

def train_model(model, train_loader, test_loader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() if TASK_MODE == 'regression' else nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.reshape(-1, 1, 8, 8))
            
            target_loss = targets.long() if TASK_MODE != 'regression' else targets
            loss = criterion(outputs, target_loss)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            if TASK_MODE != 'regression':
                preds = torch.argmax(outputs, dim=1)
                train_correct += (preds == targets).sum().item()
                train_total += targets.size(0)
        
        # --- Validation ---
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs.reshape(-1, 1, 8, 8))
                
                target_loss = targets.long() if TASK_MODE != 'regression' else targets
                loss = criterion(outputs, target_loss)
                test_loss += loss.item()
                
                if TASK_MODE != 'regression':
                    preds = torch.argmax(outputs, dim=1)
                    test_correct += (preds == targets).sum().item()
                    test_total += targets.size(0)
        
        # Record Metrics
        epoch_metrics = {
            'train_loss': train_loss / len(train_loader),
            'test_loss': test_loss / len(test_loader),
            'train_acc': (train_correct / train_total * 100) if TASK_MODE != 'regression' else 0,
            'test_acc': (test_correct / test_total * 100) if TASK_MODE != 'regression' else 100
        }
        for k, v in epoch_metrics.items(): history[k].append(v)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_metrics['train_loss']:.4f} | " + 
              f"Test Acc: {epoch_metrics['test_acc']:.2f}%")
        
    return history

# --- Prepare High-Efficiency DataLoaders ---
tr_ds = TensorDataset(torch.FloatTensor(X_train_raw), torch.FloatTensor(y_train_raw))
ts_ds = TensorDataset(torch.FloatTensor(X_test_raw), torch.FloatTensor(y_test_raw))

train_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(ts_ds, batch_size=64)

history = train_model(model, train_loader, test_loader, epochs=3)
"""

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Apply changes
nb['cells'][4]['source'] = [line + '\n' for line in dataset_manager_code.split('\n')]
nb['cells'][6]['source'] = [line + '\n' for line in loading_code.split('\n')]
nb['cells'][13]['source'] = [line + '\n' for line in training_code.split('\n')]

# Clean trailing newlines
for idx in [4, 6, 13]:
    if nb['cells'][idx]['source'][-1] == '\n':
        nb['cells'][idx]['source'].pop()

# Save notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated to use pre-defined dataset splits.")
