# Result & Analysis Planning

This document provides a template and structure for presenting the findings of the **Explainable HybridNet** research. Standardizing these metrics and visualizations ensures the study is scientifically rigorous and publication-ready.

## 1. Performance Metrics
We will evaluate the model across the following dimensions:

| Metric | Definition | Importance for DDoS |
| :--- | :--- | :--- |
| **Accuracy** | $(TP+TN)/(TP+TN+FP+FN)$ | Overall correctness. |
| **Precision** | $TP/(TP+FP)$ | Minimizing false alarms (False Positives). |
| **Recall** | $TP/(TP+FN)$ | Ensuring no attack goes uncaught (False Negatives). |
| **F1-Score** | $2 \cdot \frac{Prec \cdot Rec}{Prec + Rec}$ | Harmonic mean; best for imbalanced datasets. |
| **MCC** | Matthews Correlation Coeff. | Robust for large class imbalances. |
| **Inference Latency** | Time per packet ($ms$) | Critical for real-time edge deployment. |
| **Model Size** | Total Parameters / File Size | Confirms the "Lightweight" requirement. |

---

## 2. Visualization Templates

### A. Detection Performance
- **Confusion Matrix:** Heatmap showing classification across all 12+ attack types in CIC-DDoS2019.
- **ROC-AUC & Precision-Recall Curves:** To demonstrate model robustness across different decision thresholds.
- **Ablation Analysis Chart:** A bar chart comparing:
    1.  Baseline CNN
    2.  Baseline Transformer
    3.  **HybridNet (Ours)**

### B. Explainability (XAI) Visuals
- **SHAP Summary Plot:** A beeswarm plot showing which features (e.g., `Destination Port`, `Flow Duration`) contribute most to the "DDoS" classification.
- **Attention Heatmaps:** A grid showing how the Transformer's attention is distributed across the feature matrix. 
    - *Example:* "High attention localized on Packet Length features during Volumetric Attacks."
- **Local Explanation:** A bridge plot for a single instance showing why it was flagged (e.g., "Feature X pushed it towards DDoS, Feature Y pushed it towards Benign").

### C. Resource Efficiency
- **Metric vs. Complexity Trade-off:** A scatter plot with `Accuracy` on the Y-axis and `Parameter Count` or `Latency` on the X-axis. Our model should occupy the top-left quadrant (High accuracy, Low complexity).

---

## 3. Comparative Analysis Structure

We will compare **HybridNet** against state-of-the-art models identified in the Literature Review:

| Model | Architecture | Dataset | Accuracy | Latency ($ms$) | Explainable? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sathaporn (2025)** | CNN-RNN + Attn | Cloud | 95.0% | High | No |
| **Zhang (2025)** | CNN-BiLSTM-Trans| CIC-2018 | 94.0% | High | Partial |
| **HybridNet (Ours)** | **CNN-Transformer** | **Multi-Env**| **Target: >95%**| **Low** | **Yes (SHAP/Attn)** |

---

## 4. Discussion & Interpretation Plan
- **Robustness across Environments:** Discuss how the model performed on IoT data (CIC-IoT-2023) vs. Cloud data (CIC-DDoS2019).
- **Interpretability Insight:** Analyze whether the features identified by SHAP align with known networking domain knowledge (e.g., Reflection attacks should show high entropy in source IPs/Ports).
- **Scalability:** Discuss the impact of INT8 quantization on accuracy vs. speed.
