import json
import os

notebook_path = r'c:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\HybridNet_Implementation.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update DatasetManager (Cell 4/5)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'class DatasetManager:' in source:
            # Add self.scaler to __init__
            if 'self.scaler = MinMaxScaler()' not in source:
                cell['source'] = [line.replace('self.local_cic_path = self.base_dir / \'CICIOT23/train/train.csv\'\n', 
                                             'self.local_cic_path = self.base_dir / \'CICIOT23/train/train.csv\'\n        self.scaler = MinMaxScaler()\n') 
                                 for line in cell['source']]
            
            # Update _process_cic_iot to use scaler
            # We want to replace X = df.select_dtypes(include=[np.number]).values
            # with X = self.scaler.fit_transform(df.select_dtypes(include=[np.number]).values)
            cell['source'] = [line.replace('X = df.select_dtypes(include=[np.number]).values', 
                                         'X = self.scaler.fit_transform(df.select_dtypes(include=[np.number]).values)') 
                             for line in cell['source']]
            
            # Update _process_edge_iiot as well
            cell['source'] = [line.replace('X = df.select_dtypes(include=[np.number]).values[:, :64]', 
                                         'X = self.scaler.fit_transform(df.select_dtypes(include=[np.number]).values)[:, :64]') 
                             for line in cell['source']]

# 2. Update explain_task (search for the cell containing it)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'def explain_task(model, loader):' in source:
            # Remove annot=True
            cell['source'] = [line.replace('annot=True', 'annot=False') for line in cell['source']]
            # Update attribution heatmap params
            cell['source'] = [line.replace('cmap=\'RdBu\'', 'cmap=\'seismic\', center=0') for line in cell['source']]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Successfully enhanced {notebook_path}")
