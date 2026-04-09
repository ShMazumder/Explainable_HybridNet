import json

notebook_path = r"C:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\HybridNet_Implementation.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix the logic in DatasetManager.load_split (Cell 4)
# We want to allow None to propagate as "Full Load"
manager_source = nb['cells'][4]['source']
new_manager_source = []
for line in manager_source:
    if "n_samples = n_samples_override if n_samples_override is not None else self.n_samples" in line:
        # If the argument is omitted entirely (using original default n_samples_override=None), 
        # but the intention was to use self.n_samples, we need a better way.
        # Actually, in the current calling code, we ALWAYS pass SAMPLE_SIZES[split].
        # So we should just use what's passed.
        new_manager_source.append("        # If an override is provided (even if None), use it. Else use default.\n")
        # I'll change the method signature in the next step or use a check.
        # But for now, let's fix the specific line to favor the override's value for None.
        new_manager_source.append("        n_samples = n_samples_override\n")
    else:
        new_manager_source.append(line)
nb['cells'][4]['source'] = new_manager_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("DatasetManager fixed to correctly propagate 'None' as a full-load signal.")
