# Model Calibration Study — Temperature Scaling on CIFAR-10

> **Can we trust a model's confidence score?**  
> This project investigates neural network calibration and applies temperature scaling to fix overconfidence — without retraining.

---

## What is Calibration?

A model that outputs "95% confidence" should be correct **95% of the time** when it says that. Most modern neural networks are not — they tend to be overconfident, outputting high confidence even when wrong.

**Calibration** measures how well a model's confidence scores match its actual accuracy.

---

## What is Temperature Scaling?

Temperature scaling is the simplest post-hoc calibration fix. It adds a single scalar `T` to the model's output:

```
p = softmax(logits / T)
```

- `T > 1` → softens the distribution → less confident
- `T = 1` → no change (original model)
- `T < 1` → sharpens the distribution → more confident

We find the best `T` by minimising the **Negative Log-Likelihood** on a held-out validation set.

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| **ECE** (Expected Calibration Error) | Average gap between confidence and accuracy across bins. Lower is better. |
| **MCE** (Maximum Calibration Error) | Worst-case gap across all confidence bins. |

---

## Project Structure

```
model-calibration-study/
│
├── main.py                        # Entry point — runs the full study
│
├── models/
│   ├── __init__.py
│   └── model_loader.py            # ResNet-50 adapted for CIFAR-10
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py             # CIFAR-10 download + train/val/test split
│   ├── calibration_metrics.py     # ECE, MCE, reliability data computation
│   ├── temperature_scaling.py     # TemperatureScaler class
│   └── plotting.py                # Reliability diagrams, histograms, bar charts
│
├── results/                       # Auto-created — plots saved here
├── data/                          # Auto-created — CIFAR-10 downloaded here
│
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Clone and set up
```bash
git clone https://github.com/YOUR_USERNAME/model-calibration-study.git
cd model-calibration-study

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the study
```bash
python main.py
```

This will:
1. Download CIFAR-10 (~170 MB, automatic)
2. Load a pretrained ResNet-50
3. Collect logits on val + test sets
4. Compute ECE and MCE **before** calibration
5. Fit temperature scaling on the validation set
6. Compute ECE and MCE **after** calibration
7. Save plots to `results/`
8. Print a summary table

---

## Results

After running, you will find three plots in `results/`:

| File | Description |
|------|-------------|
| `reliability_diagram.png` | Bar chart: accuracy per confidence bin, before vs after |
| `confidence_histogram.png` | Distribution of model confidence scores |
| `ece_comparison.png` | ECE and MCE before vs after, side by side |

---

## Why This Matters

Calibration is critical in high-stakes applications:
- **Medical AI**: "90% confidence it's benign" should mean something precise
- **Autonomous vehicles**: Uncertainty must be trustworthy
- **Any system where humans act on model confidence**

---

## References

- Guo et al. (2017) — [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) *(the foundational paper for this project)*
- Niculescu-Mizil & Caruana (2005) — Predicting Good Probabilities with Supervised Learning

---

## Tech Stack

`Python` · `PyTorch` · `torchvision` · `matplotlib` · `NumPy` · `SciPy`