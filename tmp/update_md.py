import json

notebook_path = r"C:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\HybridNet_Implementation.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update cell index 5 (Markdown before simulator)
nb['cells'][5]['source'] = [
    "## 1. Data Loading & Reshaping\n",
    "Initially designed with a simulator for multi-GB datasets, this framework now supports the full **CIC-IoT-2023** dataset. The `DatasetManager` automatically detects local data in `datasets/CICIOT23` and uses optimized `nrows` loading to maintain memory efficiency while providing real-world validation."
]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Markdown cell updated.")
