import json

notebook_path = r"C:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\HybridNet_Implementation.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Cell 3 (Global Config)
nb['cells'][3]['source'] = [
    "# --- GLOBAL CONFIGURATION ---\n",
    "DATASET_TYPE = 'CIC-IoT-2023'  # 'Simulation', 'CIC-IoT-2023', 'Edge-IIoTset'\n",
    "TASK_MODE = 'multiclass'    # 'binary', 'multiclass', 'regression'\n",
    "# Split-wise sample sizes: Use an integer for a subset, or None to load the FULL split.\n",
    "SAMPLE_SIZES = {\n",
    "    'train': 200000, \n",
    "    'test': None, \n",
    "    'validation': None\n",
    "}\n",
    "# ----------------------------\n",
    "print(f'Configured for {DATASET_TYPE} with {TASK_MODE}')"
]

# 2. Update Cell 4 (DatasetManager)
# Modify to handle n_samples override
new_manager_source = [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import DataLoader, TensorDataset\n",
    "import urllib.request\n",
    "from pathlib import Path\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "class DatasetManager:\n",
    "    def __init__(self, name, mode, n_samples=5000):\n",
    "        self.name = name\n",
    "        self.mode = mode\n",
    "        self.n_samples = n_samples\n",
    "        self.base_dir = Path('./datasets')\n",
    "        self.base_dir.mkdir(exist_ok=True)\n",
    "        self.scaler = MinMaxScaler()\n",
    "        self.label_names = []\n",
    "        self.ciciot_dir = self.base_dir / 'CICIOT23'\n",
    "        \n",
    "    def load_split(self, split='train', n_samples_override=None):\n",
    "        \"\"\"Loads a specific dataset split (train/test/validation).\"\"\"\n",
    "        # Use override if provided, else use default\n",
    "        n_samples = n_samples_override if n_samples_override is not None else self.n_samples\n",
    "        \n",
    "        if self.name == 'Simulation':\n",
    "            return self._load_synthetic(n_samples)\n",
    "        elif self.name == 'CIC-IoT-2023':\n",
    "            path = self.ciciot_dir / split / f\"{split}.csv\"\n",
    "            if not path.exists():\n",
    "                print(f\"Warning: Split {split} path {path} not found. Falling back to default.\")\n",
    "                return self._process_cic_iot(self._check_download_cic(), n_samples, fit_scaler=(split=='train'))\n",
    "            return self._process_cic_iot(path, n_samples, fit_scaler=(split=='train'))\n",
    "        elif self.name == 'Edge-IIoTset':\n",
    "            path = self._check_download_edge()\n",
    "            return self._process_edge_iiot(path, n_samples)\n",
    "        else:\n",
    "            raise ValueError(f\"Unknown dataset: {self.name}\")\n",
    "\n",
    "    def _process_cic_iot(self, path, n_samples, fit_scaler=False):\n",
    "        # pd.read_csv handles nrows=None by reading the full file\n",
    "        df = pd.read_csv(path, nrows=n_samples)\n",
    "        \n",
    "        X = df.select_dtypes(include=[np.number]).values\n",
    "        if fit_scaler: X = self.scaler.fit_transform(X)\n",
    "        else: X = self.scaler.transform(X)\n",
    "            \n",
    "        if X.shape[1] > 64: X = X[:, :64]\n",
    "        if X.shape[1] < 64: X = np.pad(X, ((0, 0), (0, 64-X.shape[1])), 'constant')\n",
    "        \n",
    "        label_col = df.columns[-1]\n",
    "        if self.mode == 'binary':\n",
    "            y = (~df[label_col].astype(str).str.contains('Benign', case=False)).astype(int).values\n",
    "        else:\n",
    "            if not self.label_names or fit_scaler:\n",
    "                self.label_names = sorted(df[label_col].unique().tolist())\n",
    "            mapping = {name: i for i, name in enumerate(self.label_names)}\n",
    "            y = df[label_col].map(lambda x: mapping.get(x, -1)).values\n",
    "        return X, y\n",
    "\n",
    "    def _load_synthetic(self, n_samples):\n",
    "        X = np.random.randn(n_samples, 64)\n",
    "        if self.mode == 'binary': y = np.random.randint(0, 2, n_samples)\n",
    "        elif self.mode == 'multiclass': y = np.random.randint(0, 8, n_samples)\n",
    "        else: y = np.random.rand(n_samples, 2)\n",
    "        return X, y\n",
    "\n",
    "    def _check_download_cic(self):\n",
    "        url = 'https://raw.githubusercontent.com/mohamedamineferrag/CICIoT2023/main/sample/sampled_data.csv'\n",
    "        path = self.base_dir / 'cic_sample.csv'\n",
    "        if not path.exists():\n",
    "            try: urllib.request.urlretrieve(url, path)\n",
    "            except: pass\n",
    "        return path\n",
    "\n",
    "    def _process_edge_iiot(self, path, n_samples):\n",
    "        df = pd.read_csv(path, nrows=n_samples)\n",
    "        X = self.scaler.fit_transform(df.select_dtypes(include=[np.number]).values)[:, :64]\n",
    "        if X.shape[1] < 64: X = np.pad(X, ((0, 0), (0, 64-X.shape[1])), 'constant')\n",
    "        if self.mode == 'binary': y = (df.iloc[:, -1] != 'Normal').astype(int).values\n",
    "        else:\n",
    "            y_series = df.iloc[:, -1].astype('category')\n",
    "            self.label_names = y_series.cat.categories.tolist()\n",
    "            y = y_series.cat.codes.values\n",
    "        return X, y\n"
]
nb['cells'][4]['source'] = new_manager_source

# 3. Update Cell 6 (Loading Logic)
nb['cells'][6]['source'] = [
    "# Load Splits via Manager with Split-wise Config\n",
    "manager = DatasetManager(DATASET_TYPE, TASK_MODE)\n",
    "\n",
    "print(f\"Loading Training Split (Size: {SAMPLE_SIZES['train']})...\")\n",
    "X_train_raw, y_train_raw = manager.load_split('train', n_samples_override=SAMPLE_SIZES['train'])\n",
    "\n",
    "print(f\"Loading Testing Split (Size: {SAMPLE_SIZES['test']})...\")\n",
    "X_test_raw, y_test_raw = manager.load_split('test', n_samples_override=SAMPLE_SIZES['test'])\n",
    "\n",
    "print(f\"Loading Validation Split (Size: {SAMPLE_SIZES['validation']})...\")\n",
    "X_val_raw, y_val_raw = manager.load_split('validation', n_samples_override=SAMPLE_SIZES['validation'])\n",
    "\n",
    "if TASK_MODE == 'multiclass':\n",
    "    print(f\"Dataset contains {len(manager.label_names)} labels.\")\n"
]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Split-wise SAMPLE_SIZES configuration implemented successfully.")
