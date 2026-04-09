import json

notebook_path = r"C:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\HybridNet_Implementation.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Correcting Cell Types and Content

# 1. Cell 5 should be Markdown
nb['cells'][5]['cell_type'] = 'markdown'
nb['cells'][5]['source'] = [
    "## 1. Data Loading & Reshaping\n",
    "Initially designed with a simulator for multi-GB datasets, this framework now supports the full **CIC-IoT-2023** dataset. The `DatasetManager` automatically detects local data in `datasets/CICIOT23` and uses optimized `nrows` loading to maintain memory efficiency while providing real-world validation."
]

# 2. Cell 6 should be Code with the correct splitting logic
nb['cells'][6]['cell_type'] = 'code'
nb['cells'][6]['source'] = [
    "# Load Splits via Manager\n",
    "manager = DatasetManager(DATASET_TYPE, TASK_MODE, SAMPLE_SIZE)\n",
    "\n",
    "# Efficiently load separate splits\n",
    "print(\"Loading Training Split...\")\n",
    "X_train_raw, y_train_raw = manager.load_split('train')\n",
    "\n",
    "print(\"Loading Testing Split...\")\n",
    "X_test_raw, y_test_raw = manager.load_split('test')\n",
    "\n",
    "print(\"Loading Validation Split...\")\n",
    "X_val_raw, y_val_raw = manager.load_split('validation')\n",
    "\n",
    "# Store names for XAI\n",
    "if TASK_MODE == 'multiclass':\n",
    "    print(f\"Dataset contains {len(manager.label_names)} labels.\")\n"
]

# 3. Cell 7 is redundant/wrong, we'll keep it as a markdown spacer or remove it.
# Actually, the user's error was in "Cell In[6]", which might map to this index.
# Let's remove the redundancy by shifting or just emptying the problematic cell.
# Better to just delete it to avoid confusion.
nb['cells'].pop(7)

# 4. Update the "Class Distribution Analysis" (formerly Cell 9, now Cell 8)
# It needs to use y_train_raw instead of a non-existent y_raw.
nb['cells'][8]['source'] = [line.replace('y_raw', 'y_train_raw') for line in nb['cells'][8]['source']]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook structure fixed. Redundant load_data() cell removed.")
