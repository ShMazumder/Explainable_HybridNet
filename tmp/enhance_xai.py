import json

notebook_path = r"C:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\HybridNet_Implementation.ipynb"

# Full list of features extracted from the CICIOT23 dataset
feature_names = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate', 
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number', 
    'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'urg_count', 
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 
    'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 
    'Tot size', 'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
]
# Pad to 64
feature_names += [f'Padding_{i}' for i in range(64 - len(feature_names))]

new_code = f"""# Updated XAI Interpretation Module with Dynamic Feature Mapping
from captum.attr import IntegratedGradients
import numpy as np

# Feature names mapping for the 8x8 grid
FEATURE_LIST = {feature_names}

def explain_task(model, loader, manager):
    \"\"\"
    Enhanced XAI explanation function:
    1. Dynamically selects the predicted class as the explanation target.
    2. Maps grid cells to real-world network feature names.
    3. Displays human-readable labels for the clarified prediction.
    \"\"\"
    model.eval()
    inputs, labels = next(iter(loader))
    
    # Select the first sample from the batch
    input_sample = inputs[0:1].reshape(1, 1, 8, 8).to(device).requires_grad_()
    
    # 1. Get Prediction
    with torch.no_grad():
        outputs = model(input_sample)
        pred_idx = torch.argmax(outputs, dim=1).item()
        
    # Get human-readable label
    pred_label = manager.label_names[pred_idx] if hasattr(manager, 'label_names') and manager.label_names else str(pred_idx)
    true_label = manager.label_names[int(labels[0])] if hasattr(manager, 'label_names') and manager.label_names else str(int(labels[0]))
    
    # 2. Compute Integrated Gradients
    ig = IntegratedGradients(model)
    attr = ig.attribute(input_sample, target=pred_idx)
    
    # 3. Visualization
    plt.figure(figsize=(14, 6))
    
    # Plot Input Features
    plt.subplot(1, 2, 1)
    sns.heatmap(input_sample.detach().cpu().numpy()[0,0], annot=False, cmap='viridis')
    plt.title(f'Normalized Input Features\\n(True: {{true_label}}, Pred: {{pred_label}})')
    
    # Plot Attribution
    plt.subplot(1, 2, 2)
    attr_np = attr.detach().cpu().numpy()[0,0]
    sns.heatmap(attr_np, annot=False, cmap='RdBu', center=0)
    plt.title(f'Feature Attribution (Target: {{pred_label}})')
    
    plt.tight_layout()
    plt.show()
    
    # 4. Top Feature Insights
    attr_flat = attr_np.flatten()
    top_indices = np.argsort(np.abs(attr_flat))[-5:][::-1]
    
    print(f"\\n--- XAI Insight for [{{pred_label}}] ---")
    print(f"Top 5 Influential Features:")
    for idx in top_indices:
        feat_name = FEATURE_LIST[idx]
        val = attr_flat[idx]
        impact = "Increased" if val > 0 else "Decreased"
        print(f"- {{feat_name:<20}}: {{val:.4f}} ({{impact}} probability)")

# Run explanation
explain_task(model, test_loader, manager)"""

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'][15]['source'] = [line + '\n' for line in new_code.split('\n')]
# Fix the trailing newline
if nb['cells'][15]['source'][-1] == '\n':
    nb['cells'][15]['source'].pop()

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("XAI module enhanced successfully.")
