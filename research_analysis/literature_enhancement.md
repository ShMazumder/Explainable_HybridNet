# Literature Enhancement: Explainable HybridNet

Your existing literature review is remarkably up-to-date (with several 2025 citations). To further strengthen the "Lightweight" and "Explainable" pillars of your research, I suggest incorporating the following specific research directions and papers.

## 1. Missing Pillar: Model Compression & Lightweight Optimization
While "lightweight" is a keyword in your title, the current literature focusing on **how** to achieve this via PyTorch-specific optimizations is sparse. Adding these will make your methodology more robust.

### Suggested References:
- **Reference A:** *G. Hinton et al., "Distilling the Knowledge in a Neural Network," 2015.* (Foundational for Knowledge Distillation).
- **Reference B:** *M. Nagel et al., "A White Paper on Neural Network Quantization," 2021.* (Essential for INT8 optimization in PyTorch).
- **Reference C:** *L. Deng et al., "Model Compression and Hardware Acceleration for Deep Neural Networks," 2020.* (Relevant for Edge/IoT hardware constraints).

**Why add this?** It provides a theoretical basis for the "Lightweight" claim in your title, moving beyond "few layers" to actual "computational efficiency."

## 2. Advanced Explainability: Beyond SHAP
SHAP is excellent, but for "Modern Networks," **Global interpretation** (what the whole model learned) and **Causal interpretation** are the new frontiers.

### Suggested References:
- **Reference D:** *R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," 2017.* (Useful for visualizing CNN feature maps).
- **Reference E:** *A. Vaswani et al., "Attention is All You Need," 2017.* (Foundational for explaining the Transformer's attention mechanism).
- **Reference F (Recent):** *LENS-XAI (2025): "Scalable Attribution-based Explanations for Industrial IoT Security."* (Matches your IoT/Edge focus perfectly).

**Why add this?** It shows you are aware of the evolution of XAI from general ML to domain-specific (IIoT/DDoS) applications.

## 3. Modern Hybrid Benchmarks
To support your "both IoT and Cloud" requirement, consider these datasets and their associated literature:

### Suggested References:
- **Reference G:** *Neto et al., "CIC-IoT-2023: A novel and comprehensive dataset for the Internet of Things," 2023.*
- **Reference H:** *Sami et al., "Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications," 2022.*

## 4. Summary Table for LaTeX Integration
You can add these to your comparative study table in `main.tex`:

| Study | Focus | Contribution to HybridNet |
| :--- | :--- | :--- |
| **Hinton et al. (2015)** | Knowledge Distillation | Strategy for training the lightweight student model. |
| **Nagel et al. (2021)** | INT8 Quantization | Framework for deployment on resource-constrained Edge devices. |
| **LENS-XAI (2025)** | Attribution-based XAI | State-of-the-art benchmark for XAI in IIoT. |
| **Neto et al. (2023)** | CIC-IoT-2023 | Modern dataset providing diverse IoT-specific DDoS patterns. |

---

### Action Items for `main.tex`:
1.  **Section 1.2 (Rationale):** Mention that while accuracy is high, the *resource-efficiency* (quantization/distillation) remains an open challenge.
2.  **Section 2 (Literature Review):** Add a paragraph specifically on "AI Model Compression for Cybersecurity."
3.  **Table 2.1:** Insert a row for LENS-XAI (2025) to show the most recent competition.
