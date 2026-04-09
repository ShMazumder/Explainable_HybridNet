import json

notebook_path = r"C:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\HybridNet_Implementation.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Cell 3 (Global Config)
# Set to None as requested by the user
nb['cells'][3]['source'] = [
    "# --- GLOBAL CONFIGURATION ---\n",
    "DATASET_TYPE = 'CIC-IoT-2023'  # 'Simulation', 'CIC-IoT-2023', 'Edge-IIoTset'\n",
    "TASK_MODE = 'multiclass'    # 'binary', 'multiclass', 'regression'\n",
    "# Split-wise sample sizes: Use an integer for a subset, or None to load the FULL split.\n",
    "SAMPLE_SIZES = {\n",
    "    'train': None, \n",
    "    'test': None, \n",
    "    'validation': None\n",
    "}\n",
    "# ----------------------------\n",
    "print(f'Configured for {DATASET_TYPE} with {TASK_MODE}')"
]

# 2. Fix _load_synthetic in Cell 4
# Ensure it doesn't crash when n_samples is None
manager_source = nb['cells'][4]['source']
new_manager_source = []
for line in manager_source:
    if "X = np.random.randn(n_samples, 64)" in line:
        new_manager_source.append("        # Handle None case for synthetic generation\n")
        new_manager_source.append("        cnt = n_samples if n_samples is not None else 1000\n")
        new_manager_source.append("        X = np.random.randn(cnt, 64)\n")
    elif "if self.mode == 'binary': y = np.random.randint(0, 2, n_samples)" in line:
        new_manager_source.append("        if self.mode == 'binary': y = np.random.randint(0, 2, cnt)\n")
    elif "elif self.mode == 'multiclass': y = np.random.randint(0, 8, n_samples)" in line:
        new_manager_source.append("        elif self.mode == 'multiclass': y = np.random.randint(0, 8, cnt)\n")
    elif "else: y = np.random.rand(n_samples, 2)" in line:
        new_manager_source.append("        else: y = np.random.rand(cnt, 2)\n")
    else:
        new_manager_source.append(line)
nb['cells'][4]['source'] = new_manager_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Global configuration updated to load all data by default.")
