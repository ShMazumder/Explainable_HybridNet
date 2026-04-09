import json
import os

notebook_path = r'c:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\HybridNet_Implementation.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the training cell index
target_index = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if '# --- Build Model ---' in source:
            target_index = i
            break

if target_index != -1:
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "## 3. Class Distribution Analysis\n",
            "# Visualizing the balance of Attack vs Benign classes in the current sample\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import numpy as np\n",
            "\n",
            "if TASK_MODE != 'regression':\n",
            "    plt.figure(figsize=(14, 6))\n",
            "    # Show distribution of y_raw\n",
            "    counts = np.bincount(y_raw.astype(int))\n",
            "    sns.barplot(x=np.arange(len(counts)), y=counts, palette='viridis')\n",
            "    plt.title(f'Class Distribution: {DATASET_TYPE} ({TASK_MODE})')\n",
            "    plt.xlabel('Class Index')\n",
            "    plt.ylabel('Number of Samples')\n",
            "    plt.xticks(rotation=45)\n",
            "    plt.grid(axis='y', linestyle='--', alpha=0.7)\n",
            "    plt.show()\n",
            "\n",
            "    # Print percentages\n",
            "    print(f\"Total Samples: {len(y_raw)}\")\n",
            "    for i, count in enumerate(counts):\n",
            "        if count > 0:\n",
            "            print(f\"  Class {i:2d}: {count:5d} samples ({count/len(y_raw)*100:5.2f}%)\")\n",
            "else:\n",
            "    plt.figure(figsize=(10, 4))\n",
            "    plt.hist(y_raw, bins=50, color='skyblue', edgecolor='black')\n",
            "    plt.title(f'Regression Target Distribution: {DATASET_TYPE}')\n",
            "    plt.xlabel('Target Value')\n",
            "    plt.ylabel('Frequency')\n",
            "    plt.show()\n"
        ]
    }
    
    # Insert before the target_index
    nb['cells'].insert(target_index, new_cell)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully inserted distribution cell at index {target_index} in {notebook_path}")
else:
    print("Could not find the training cell with '# --- Build Model ---'")
