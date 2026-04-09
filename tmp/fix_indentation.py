import json
import os

notebook_path = r'c:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\HybridNet_Implementation.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix Indentation in Class Distribution cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'if hasattr(manager, "label_names")' in source:
            # Re-write the source with clean indentation
            cell['source'] = [
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import numpy as np\n",
                "\n",
                "if TASK_MODE != 'regression':\n",
                "    plt.figure(figsize=(16, 6))\n",
                "    counts = np.bincount(y_raw.astype(int))\n",
                "    \n",
                "    if hasattr(manager, 'label_names') and len(manager.label_names) > 0:\n",
                "        sns.barplot(x=manager.label_names, y=counts, palette='viridis')\n",
                "    else:\n",
                "        sns.barplot(x=np.arange(len(counts)), y=counts, palette='viridis')\n",
                "        \n",
                "    plt.title(f'Class Distribution: {DATASET_TYPE} ({TASK_MODE})')\n",
                "    plt.xlabel('Attack Type')\n",
                "    plt.ylabel('Number of Samples')\n",
                "    plt.xticks(rotation=90)\n",
                "    plt.grid(axis='y', linestyle='--', alpha=0.7)\n",
                "    plt.show()\n",
                "\n",
                "    print(f\"Total Samples: {len(y_raw)}\")\n",
                "    for i, count in enumerate(counts):\n",
                "        if count > 0:\n",
                "            name = manager.label_names[i] if hasattr(manager, 'label_names') and i < len(manager.label_names) else f'Class {i}'\n",
                "            print(f\"  {name:30s}: {count:5d} samples ({count/len(y_raw)*100:6.2f}%)\")\n",
                "else:\n",
                "    plt.figure(figsize=(10, 4))\n",
                "    plt.hist(y_raw, bins=50, color='skyblue', edgecolor='black')\n",
                "    plt.title(f'Regression Target Distribution: {DATASET_TYPE}')\n",
                "    plt.xlabel('Target Value')\n",
                "    plt.ylabel('Frequency')\n",
                "    plt.show()\n"
            ]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Fixed indentation in {notebook_path}")
