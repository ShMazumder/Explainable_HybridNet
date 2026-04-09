import json

notebook_path = r"C:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\HybridNet_Implementation.ipynb"

# Full list of features for XAI
feature_names = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate', 
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number', 
    'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'urg_count', 
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 
    'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 
    'Tot size', 'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
]
feature_names += [f'Padding_{i}' for i in range(64 - len(feature_names))]

# 1. Update Cell 13 (Training with Detailed Metrics)
new_training_code = """# --- Build Model ---
# Dynamically determine the number of classes for the current task
if TASK_MODE == 'regression':
    num_classes = 2 # Regression output (Magnitude, Probability)
else:
    num_classes = len(np.unique(y_raw))
    print(f'Detected {num_classes} classes for training.')

model = HybridNet(num_classes=num_classes, mode=TASK_MODE).to(device)

def train_model(model, train_loader, test_loader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() if TASK_MODE == 'regression' else nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.reshape(-1, 1, 8, 8))
            
            target_loss = targets.long() if TASK_MODE != 'regression' else targets
            loss = criterion(outputs, target_loss)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            if TASK_MODE != 'regression':
                preds = torch.argmax(outputs, dim=1)
                train_correct += (preds == targets).sum().item()
                train_total += targets.size(0)
        
        # --- Validation Phase ---
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs.reshape(-1, 1, 8, 8))
                
                target_loss = targets.long() if TASK_MODE != 'regression' else targets
                loss = criterion(outputs, target_loss)
                test_loss += loss.item()
                
                if TASK_MODE != 'regression':
                    preds = torch.argmax(outputs, dim=1)
                    test_correct += (preds == targets).sum().item()
                    test_total += targets.size(0)
        
        # Record Metrics
        metrics = {
            'train_loss': train_loss/len(train_loader),
            'test_loss': test_loss/len(test_loader),
            'train_acc': (train_correct/train_total)*100 if TASK_MODE != 'regression' else 0,
            'test_acc': (test_correct/test_total)*100 if TASK_MODE != 'regression' else 0
        }
        
        for k, v in metrics.items(): history[k].append(v)
        
        # Detailed Reporting
        msg = f"Epoch {epoch+1}/{epochs} | "
        msg += f"Train Loss: {metrics['train_loss']:.4f} | "
        msg += f"Test Loss: {metrics['test_loss']:.4f}"
        if TASK_MODE != 'regression':
            msg += f" | Train Acc: {metrics['train_acc']:.2f}% | Test Acc: {metrics['test_acc']:.2f}%"
        print(msg)
        
    return history

# Prep Data Splits
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
tr_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
ts_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
train_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(ts_ds, batch_size=64)

history = train_model(model, train_loader, test_loader, epochs=3)"""

# 2. Update Cell 15 (XAI with Descriptive Narrative)
new_xai_code = f"""# Updated XAI Interpretation Module with Descriptive Narrative
from captum.attr import IntegratedGradients
import numpy as np

# Feature definitions for descriptive interpretation
FEATURE_INFO = {{
    'flow_duration': 'Duration of the network flow',
    'Rate': 'Frequency of packets per second',
    'syn_flag_number': 'Presence of TCP Synchronize flags (Handshake start)',
    'ack_flag_number': 'Presence of TCP Acknowledgment flags',
    'rst_flag_number': 'Presence of TCP Reset flags (Connection termination)',
    'fin_flag_number': 'Presence of TCP Finish flags',
    'IAT': 'Inter-Arrival Time between packets',
    'Tot size': 'Total size of payloads in the flow',
    'Magnitue': 'Signal magnitude indicator',
    'Radius': 'Flow dispersion metric'
}}

FEATURE_LIST = {feature_names}

def explain_task(model, loader, manager):
    model.eval()
    inputs, labels = next(iter(loader))
    input_sample = inputs[0:1].reshape(1, 1, 8, 8).to(device).requires_grad_()
    
    # 1. Prediction
    with torch.no_grad():
        outputs = model(input_sample)
        pred_idx = torch.argmax(outputs, dim=1).item()
    
    pred_label = manager.label_names[pred_idx] if hasattr(manager, 'label_names') and manager.label_names else str(pred_idx)
    true_label = manager.label_names[int(labels[0])] if hasattr(manager, 'label_names') and manager.label_names else str(int(labels[0]))
    
    # 2. Integrated Gradients
    ig = IntegratedGradients(model)
    attr = ig.attribute(input_sample, target=pred_idx)
    attr_np = attr.detach().cpu().numpy()[0,0]
    
    # 3. Visualization
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(input_sample.detach().cpu().numpy()[0,0], annot=False, cmap='viridis')
    plt.title(f'Inputs (True: {{true_label}}, Pred: {{pred_label}})')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(attr_np, annot=False, cmap='RdBu', center=0)
    plt.title(f'Feature Attribution (Target: {{pred_label}})')
    plt.tight_layout()
    plt.show()
    
    # 4. Descriptive Narrative Generation
    attr_flat = attr_np.flatten()
    top_indices = np.argsort(np.abs(attr_flat))[-5:][::-1]
    
    print(f"\\n--- [ Decision Summary ] ---")
    print(f"Model identified this flow as: **{{pred_label}}**")
    print(f"The decision was primarily driven by the following factors:\\n")
    
    for idx in top_indices:
        feat = FEATURE_LIST[idx]
        val = attr_flat[idx]
        desc = FEATURE_INFO.get(feat, "General network attribute")
        impact = "positively" if val > 0 else "negatively"
        strength = "significant" if abs(val) > 0.1 else "subtle"
        
        print(f"- **{{feat}}** ({{desc}}):")
        print(f"  This feature {{impact}} influenced the prediction with a {{strength}} weight ({{val:.4f}}).")
    
    print(f"\\n--- [ Conclusion ] ---")
    if "Benign" in pred_label:
        print("The model observed balanced flow patterns and standard handshake behaviors, typical of normal traffic.")
    else:
        print(f"The identification of [{{pred_label}}] is consistent with observations of {{FEATURE_LIST[top_indices[0]]}} patterns, ")
        print("commonly associated with resource exhaustion or protocol exploitation in modern networks.")

# Run explanation
explain_task(model, test_loader, manager)"""

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'][13]['source'] = [line + '\n' for line in new_training_code.split('\n')]
nb['cells'][15]['source'] = [line + '\n' for line in new_xai_code.split('\n')]

# Clean up trailing newlines
for idx in [13, 15]:
    if nb['cells'][idx]['source'][-1] == '\n':
        nb['cells'][idx]['source'].pop()

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook enhanced with detailed metrics and XAI narrative.")
