# Experiment & XAI Plan (PyTorch Architecture)

This plan details the implementation and evaluation of **HybridNet** using the **PyTorch** framework, optimized for future-proof and lightweight performance.

## 1. Implementation Roadmap (PyTorch)

### Architecture Design
```python
import torch
import torch.nn as nn

class HybridNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(HybridNet, self).__init__()
        # 1. CNN Module: Local Spatial Patterns
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # 2. Transformer Encoder: Global Context
        self.pos_encoding = nn.Parameter(torch.zeros(1, 64, input_dim // 4))
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 3. Classifier
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(2).permute(0, 2, 1) # Prep for Transformer
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.mean(dim=1) # Global Average Pooling
        return self.fc(x)
```

### Lightweight Optimization (TorchScript & Quantization)
- **Quantization:** We will use **Post-Training Quantization (PTQ)** to convert the FP32 model to INT8.
  ```python
  model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
  quantized_model = torch.quantization.convert(model_prepared)
  ```
- **Pruning:** Remove weights with low contribution identified via SHAP.

---

## 2. XAI Implementation Strategy

We will use the **Captum** library, the official PyTorch XAI toolkit.

| Method | Purpose | Implementation |
| :--- | :--- | :--- |
| **SHAP (KernelSHAP)** | Local Feature Attribution | Explain specific instances of flagged DDoS traffic. |
| **Integrated Gradients** | Feature Baseline Comparison | Compare attack flow against a "Benign" baseline. |
| **Attention Rollout** | Visualizing Context | Extract and average attention weights from the `transformer` module. |

---

## 3. Error Analysis Plan
To ensure the model isn't hallucinatory or biased, we will perform a deep dive into misclassifications:

1.  **Confusion Matrix Breakdown:** Analyze which attack types (e.g., DNS vs NTP) are most frequently confused.
2.  **Boundary Analysis:** Use SHAP to identify "borderline" samples (flows with low confidence) and examine their raw network features.
3.  **Outlier Detection:** Verify if zero-day attacks or novel variants are consistently missed, suggesting a need for better data augmentation.

---

## 4. Hyperparameter Tuning Space
| Parameter | Range | Reasoning |
| :--- | :--- | :--- |
| **Learning Rate** | $1e-4$ to $1e-2$ | Vital for Transformer stability. |
| **Batch Size** | 32, 64, 128 | Balance between memory efficiency and gradient stability. |
| **Attention Heads**| 4, 8 | Higher heads capture more complex relations but increase latency. |
| **CNN Kernels** | 16, 32, 64 | Controls the granularity of local feature extraction. |

## 5. Deployment Simulation
- **Edge Deployment:** Benchmark the `quantized_model` on a CPU-only environment (Raspberry Pi or Edge Simulator) to measure actual inference time ($ms$).
- **Cloud Deployment:** Measure throughput (flows per second) on a multi-GPU setup.
