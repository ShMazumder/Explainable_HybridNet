# Dataset Strategy: Dual Focus (IoT/Edge & Cloud/SDN)

To satisfy the requirement of covering both **IoT/Edge** and **Cloud/SDN** modern networks, this research will utilize a multi-dataset approach. This ensures robust generalization and real-world applicability.

## 1. Selected Datasets

| Environment | Dataset | Why this dataset? | Who else used it? |
| :--- | :--- | :--- | :--- |
| **IoT/Edge** | **CIC-IoT-2023** | Specifically designed for IoT devices (Smart home, industry). High variety of DDoS. | Neto et al. (2023), Al-Zahrani (2024) |
| **IoT/Edge** | **Edge-IIoTset** | Realistic Industrial IoT traffic. Includes sensor data + network features. | Kamal (2024), Shohan (2025) |
| **Cloud/SDN** | **CIC-DDoS2019** | The benchmark for reflection-based and exploitation-based DDoS attacks. | Wang (2024), Liu (2025) |
| **Cloud/SDN** | **CSE-CIC-IDS2018**| Large-scale network monitoring traffic from AWS infrastructure. | Zhang (2025), Alabdulatif (2025) |

---

## 2. Dataset Descriptions & Statistics (Example - CIC-DDoS2019)
- **Description:** Contains 80+ network features extracted from PCAP files using CICFlowMeter. 
- **Attack Types:** NTP, DNS, LDAP, MSSQL, NetBIOS, SNMP, SSDP, UDP, UDP-Lag, WebDDoS, TFTP, and Syn.
- **Benign Traffic:** Captures normal user behavior in a corporate network.
- **Usage History:** It is the most cited dataset for hybrid CNN-Transformer research because its high feature dimensionality rewards spatial extraction (CNN).

---

## 3. Preparation Guidelines (Step-by-Step)

### Step 1: Data Cleaning
- **Drop Identity Features:** Remove `Flow ID`, `Source IP`, `Destination IP`, and `Timestamp` to prevent the model from "memorizing" specific network nodes.
- **Handle Missing/Infinite Values:** Replace `Infinity` with the max value of the feature column and fill `NaN` with zeros or the median.

### Step 2: Feature Selection & Engineering
- **Recursive Feature Elimination (RFE):** Reduce the 80+ features to the most informative top 25-64 features.
- **Reshaping (For CNN):** 
    - Pad the selected features to reach a square number (e.g., 64 features).
    - Reshape into an $8 \times 8$ grayscale "image" grid. This allows the CNN to extract relationships between correlated features that may be adjacent in the grid.

### Step 3: Class Balancing
- **The Problem:** DDoS datasets often have millions of attack packets but few "Benign" packets.
- **Solution:** 
    - **Undersampling:** Reduce the majority class (Attack) to match the minority class.
    - **SMOTE:** Synthetically generate "Benign" cases if undersampling loses too much information.

### Step 4: Normalization
- **Min-Max Scaling:** Scale all values to range $[0, 1]$.
- **Why:** Transformers are sensitive to feature scales; large raw packet counts would otherwise dominate the "Attention" weights.

---

## 4. Combined Dataset Strategy (Advanced)
If you wish to create a "Unified Model" across IoT and Cloud:
1.  **Common Feature Mapping:** Identify the intersection of features present in both CIC-IoT-2023 and CIC-DDoS2019 (e.g., Duration, Packets/s, Avg Packet Size).
2.  **Domain Adaptation:** Use a small "Environment Index" feature (0 for IoT, 1 for Cloud) to help the model adjust its internal thresholds based on the network type.
