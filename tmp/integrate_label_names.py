import json
import os

notebook_path = r'c:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\HybridNet_Implementation.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update DatasetManager
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'class DatasetManager:' in source:
            # Add self.label_names to __init__
            if 'self.label_names = []' not in source:
                cell['source'] = [line.replace('self.scaler = MinMaxScaler()\n', 
                                             'self.scaler = MinMaxScaler()\n        self.label_names = []\n') 
                                 for line in cell['source']]
            
            # Update _process_cic_iot to capture categories
            cell['source'] = [line.replace('y = df[label_col].astype(\'category\').cat.codes.values', 
                                         'y_series = df[label_col].astype(\'category\')\n        self.label_names = y_series.cat.categories.tolist()\n        y = y_series.cat.codes.values') 
                             for line in cell['source']]
            
            # Update _process_edge_iiot to capture categories
            cell['source'] = [line.replace('y = df.iloc[:, -1].astype(\'category\').cat.codes.values', 
                                         'y_series = df.iloc[:, -1].astype(\'category\')\n        self.label_names = y_series.cat.categories.tolist()\n        y = y_series.cat.codes.values') 
                             for line in cell['source']]

# 2. Update Class Distribution Analysis cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if '### 1.1 Class Distribution Analysis' in "".join(cell.get('source', [])) or 'plt.title(f\'Class Distribution:' in source:
             # Update barplot x-ticks and print statements
             new_source = []
             for line in cell['source']:
                 if 'sns.barplot(x=np.arange(len(counts))' in line:
                     # Use label names if available
                     new_source.append('    if hasattr(manager, "label_names") and len(manager.label_names) > 0:\n')
                     new_source.append('        sns.barplot(x=manager.label_names, y=counts, palette=\'viridis\')\n')
                     new_source.append('    else:\n')
                     new_source.append('        ' + line)
                 elif 'print(f"  Class {i:2d}:' in line:
                     new_source.append('        name = manager.label_names[i] if hasattr(manager, "label_names") and i < len(manager.label_names) else f"Class {i}"\n')
                     new_source.append('        print(f"  {name:25s}: {count:5d} samples ({count/len(y_raw)*100:5.2f}%)")\n')
                 else:
                     new_source.append(line)
             cell['source'] = new_source

# 3. Update Confusion Matrix cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'sns.heatmap(cm, annot=True' in source:
            new_source = []
            for line in cell['source']:
                if 'sns.heatmap(cm, annot=True' in line:
                    new_source.append('labels = manager.label_names if hasattr(manager, "label_names") else None\n')
                    new_source.append('sns.heatmap(cm, annot=True, fmt=\'d\', cmap=\'Blues\', xticklabels=labels, yticklabels=labels)\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source

# 4. Inject manager instance to make labels accessible
# We need to ensure that the distribution cell can see 'manager'
# Currently it's loaded as X_raw, y_raw = DatasetManager(...).load_data()
# Let's change it to:
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'X_raw, y_raw = DatasetManager' in source:
            cell['source'] = [line.replace('X_raw, y_raw = DatasetManager(DATASET_TYPE, TASK_MODE, SAMPLE_SIZE).load_data()', 
                                         'manager = DatasetManager(DATASET_TYPE, TASK_MODE, SAMPLE_SIZE)\nX_raw, y_raw = manager.load_data()') 
                             for line in cell['source']]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Successfully integrated label names in {notebook_path}")
