import json
import os

notebook_path = r'c:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\HybridNet_Implementation.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The imports cell is the 5th cell (index 4)
# Let's find the cell that contains 'import urllib.request' to be sure
target_cell_index = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'import urllib.request' in source:
            target_cell_index = i
            break

if target_cell_index != -1:
    new_source = [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "from torch.utils.data import DataLoader, TensorDataset\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "from sklearn.metrics import confusion_matrix\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import urllib.request\n",
        "from pathlib import Path\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import time\n",
        "\n",
        "# Device configuration\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "print(f\"Using device: {device}\")\n"
    ]
    
    # Keep the rest of the original content in that cell if any
    original_source = nb['cells'][target_cell_index]['source']
    # Filter out the old imports that we've consolidated
    remaining_source = [line for line in original_source if 'import ' not in line and 'from ' not in line]
    
    nb['cells'][target_cell_index]['source'] = new_source + remaining_source
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully updated cell {target_cell_index} in {notebook_path}")
else:
    print("Could not find the target cell with 'import urllib.request'")
