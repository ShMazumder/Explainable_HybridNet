import json

notebook_path = r"C:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\HybridNet_Implementation.ipynb"

new_plot_code = """import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_distribution(y, title, labels=None):
    counts = np.bincount(y.astype(int))
    # Correct for cases where labels list might be longer than present counts
    if labels:
        actual_labels = labels[:len(counts)]
        sns.barplot(x=actual_labels, y=counts, palette='viridis')
    else:
        sns.barplot(x=np.arange(len(counts)), y=counts, palette='viridis')
    plt.title(title)
    plt.xlabel('Attack Type')
    plt.ylabel('Samples')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

if TASK_MODE != 'regression':
    plt.figure(figsize=(18, 5))
    
    # Train
    plt.subplot(1, 3, 1)
    plot_distribution(y_train_raw, 'Train Distribution', manager.label_names)
    
    # Test
    plt.subplot(1, 3, 2)
    plot_distribution(y_test_raw, 'Test Distribution', manager.label_names)
    
    # Validation
    plt.subplot(1, 3, 3)
    plot_distribution(y_val_raw, 'Validation Distribution', manager.label_names)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Summary Statistics:")
    print(f"  Train samples:      {len(y_train_raw)}")
    print(f"  Test samples:       {len(y_test_raw)}")
    print(f"  Validation samples: {len(y_val_raw)}")
else:
    plt.figure(figsize=(15, 4))
    datasets = [(y_train_raw, 'Train', 'skyblue'), (y_test_raw, 'Test', 'salmon'), (y_val_raw, 'Val', 'lightgreen')]
    
    for i, (data, title, color) in enumerate(datasets):
        plt.subplot(1, 3, i+1)
        plt.hist(data, bins=50, color=color, edgecolor='black')
        plt.title(f'{title} Distribution')
        plt.xlabel('Target Value')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()"""

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'][8]['source'] = [line + '\n' for line in new_plot_code.split('\n')]
if nb['cells'][8]['source'][-1] == '\n':
    nb['cells'][8]['source'].pop()

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Class Distribution Analysis cell updated to include all splits.")
