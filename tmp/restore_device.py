import json

notebook_path = r"C:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\HybridNet_Implementation.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Add device definition to Cell 4
# Placing it alongside primary imports.
source = nb['cells'][4]['source']
new_source = [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import DataLoader, TensorDataset\n",
    "# Device configuration\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f'Using device: {device}')\n",
    "\n"
] + [line for line in source if "import torch" not in line and "from torch.utils.data" not in line]

nb['cells'][4]['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Device configuration restored to Cell 4.")
