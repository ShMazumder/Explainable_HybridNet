import json
import os

notebook_path = r'c:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\HybridNet_Implementation.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Identify and remove the misplaced cell
misplaced_index = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if '## 3. Class Distribution Analysis' in source:
            misplaced_index = i
            break

if misplaced_index != -1:
    dist_cell = nb['cells'].pop(misplaced_index)
    # Clean up the header inside the code cell if we're making a separate markdown cell
    dist_cell['source'] = [line for line in dist_cell['source'] if '## 3.' not in line]
    print(f"Removed misplaced cell at index {misplaced_index}")
else:
    print("Could not find the misplaced cell.")
    dist_cell = None

# 2. Find the correct insertion point (after X_raw, y_raw loading)
insertion_index = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'X_raw, y_raw =' in source:
            insertion_index = i + 1
            break

if insertion_index != -1 and dist_cell:
    # Insert a Markdown header
    header_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1.1 Class Distribution Analysis\n",
            "Before proceeding to the model architecture, we check the distribution of the loaded data sample to identify potential class imbalances."
        ]
    }
    
    nb['cells'].insert(insertion_index, header_cell)
    nb['cells'].insert(insertion_index + 1, dist_cell)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully relocated distribution analysis to index {insertion_index}")
else:
    print("Could not find insertion point or misplaced cell context.")
