import json

notebook_path = r"C:\Users\Dell\Downloads\Explainable_HybridNet__A_Lightweight_CNN_Transformer_Framework_for_Intelligent_DDoS_Attack_Detection_and_Interpretation_in_Modern_Networks\HybridNet_Implementation.ipynb"

# Dedicated CICIOT23 Label List (34 Classes)
# Discovered from the dataset structure to ensure consistent mapping even with sampling.
CICIOT23_LABELS = [
    'Backdoor_Malware', 'BenignTraffic', 'BrowserHijacking', 'CommandInjection', 
    'DDoS-ACK_Fragmentation', 'DDoS-HTTP_Flood', 'DDoS-ICMP_Flood', 
    'DDoS-ICMP_Fragmentation', 'DDoS-PSHACK_Flood', 'DDoS-RSTFINFlood', 
    'DDoS-SYN_Flood', 'DDoS-SlowLoris', 'DDoS-SynonymousIP_Flood', 
    'DDoS-TCP_Flood', 'DDoS-UDP_Flood', 'DDoS-UDP_Fragmentation', 
    'DNS_Spoofing', 'DictionaryBruteForce', 'DoS-HTTP_Flood', 'DoS-SYN_Flood', 
    'DoS-TCP_Flood', 'DoS-UDP_Flood', 'MITM-ArpSpoofing', 'Mirai-greeth_flood', 
    'Mirai-greip_flood', 'Mirai-udpplain', 'Recon-HostDiscovery', 'Recon-OSScan', 
    'Recon-PingSweep', 'Recon-PortScan', 'SqlInjection', 'Uploading_Attack', 
    'VulnerabilityScan', 'XSS'
]

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update DatasetManager (Cell 4)
# Improved to use a deterministic label set for CICIOT23 to avoid mapping errors during sampling.
manager_lines = nb['cells'][4]['source']
new_manager_source = []
for line in manager_lines:
    if "self.label_names = []" in line:
        new_manager_source.append(f"        self.label_names = {CICIOT23_LABELS}\n")
    elif "self.label_names = sorted(df[label_col].unique().tolist())" in line:
        # Avoid overriding the master list during loading
        new_manager_source.append("                pass # Keep master label list\n")
    else:
        new_manager_source.append(line)
nb['cells'][4]['source'] = new_manager_source

# 2. Update Plotting Function (Cell 8)
# Added robustness against negative indices (e.g. unknown labels)
plot_lines = nb['cells'][8]['source']
new_plot_source = []
for line in plot_lines:
    if "counts = np.bincount(y.astype(int))" in line:
        # Filter out negative values if they exist
        new_plot_source.append("    y_clean = y[y >= 0]\n")
        new_plot_source.append("    counts = np.bincount(y_clean.astype(int))\n")
    else:
        new_plot_source.append(line)
nb['cells'][8]['source'] = new_plot_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Label mapping fixed and plotting robustness added.")
