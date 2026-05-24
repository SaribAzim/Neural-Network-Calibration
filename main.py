"""
main.py
-------
Orchestrates the full calibration study, step by step.
 
Run with:
    python main.py
 
Steps
-----
1. Load data         — download CIFAR-10, split into train/val/test
2. Load model        — pretrained ResNet-50 adapted for CIFAR-10
3. Collect logits    — run inference on val + test sets, save raw logits
4. Measure (before)  — compute ECE, MCE, accuracy on test set (uncalibrated)
5. Fit temperature   — find optimal T using val logits
6. Measure (after)   — recompute metrics using calibrated logits
7. Plot results      — save reliability diagrams, histograms, comparison bar
8. Print summary     — clean table of all metrics
"""
 
import os
import torch
from utils import (
    get_cifar10_loaders,
    compute_ece,
    compute_mce,
    compute_reliability_data,
    TemperatureScaler,
    plot_reliability_diagram,
    plot_confidence_histogram,
    plot_ece_comparison,
)
from utils.calibration_metrics import collect_logits_and_labels, compute_accuracy
from models import load_resnet50
 
 
# ── Config ────────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
N_BINS     = 15
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
 
 
def main():
    print("=" * 60)
    print("  Model Calibration Study — Temperature Scaling on CIFAR-10")
    print("=" * 60)
 
    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("\n[Step 1] Loading CIFAR-10 data...")
    _, val_loader, test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)
 
    # ── Step 2: Load model ────────────────────────────────────────────────────
    print("\n[Step 2] Loading pretrained ResNet-50...")
    model = load_resnet50(num_classes=10, pretrained=True, device=DEVICE)
 
    # NOTE: In a real study, you would train the model on the training split.
    # Here we use the pretrained ImageNet weights + adapted head as a
    # proxy for a trained model to demonstrate the calibration pipeline.
    # The model will be somewhat miscalibrated by default — exactly what we want.
 
    # ── Step 3: Collect logits ────────────────────────────────────────────────
    print("\n[Step 3] Running inference to collect logits...")
    val_logits,  val_labels  = collect_logits_and_labels(model, val_loader,  DEVICE)
    test_logits, test_labels = collect_logits_and_labels(model, test_loader, DEVICE)
    print(f"  Val logits shape : {val_logits.shape}")
    print(f"  Test logits shape: {test_logits.shape}")
 
    # ── Step 4: Measure before calibration ───────────────────────────────────
    print("\n[Step 4] Computing metrics BEFORE temperature scaling...")
    ece_before = compute_ece(test_logits, test_labels, n_bins=N_BINS)
    mce_before = compute_mce(test_logits, test_labels, n_bins=N_BINS)
    acc_before = compute_accuracy(test_logits, test_labels)
    bins_before = compute_reliability_data(test_logits, test_labels, n_bins=N_BINS)
    print(f"  Accuracy : {acc_before * 100:.2f}%")
    print(f"  ECE      : {ece_before:.4f}")
    print(f"  MCE      : {mce_before:.4f}")
 
    # ── Step 5: Fit temperature scaling ──────────────────────────────────────
    print("\n[Step 5] Fitting temperature scaler on validation set...")
    scaler = TemperatureScaler(init_temperature=1.5)
    scaler.fit(val_logits, val_labels, verbose=True)
    T = scaler.T
    print(f"  Final temperature T = {T:.4f}")
 
    # ── Step 6: Measure after calibration ────────────────────────────────────
    print("\n[Step 6] Computing metrics AFTER temperature scaling...")
    ece_after  = compute_ece(test_logits, test_labels, n_bins=N_BINS, temperature=T)
    mce_after  = compute_mce(test_logits, test_labels, n_bins=N_BINS, temperature=T)
    acc_after  = compute_accuracy(test_logits / T, test_labels)  # accuracy unchanged
    bins_after = compute_reliability_data(test_logits, test_labels, n_bins=N_BINS, temperature=T)
    print(f"  Accuracy : {acc_after * 100:.2f}%  (unchanged by temperature scaling)")
    print(f"  ECE      : {ece_after:.4f}")
    print(f"  MCE      : {mce_after:.4f}")
 
    # ── Step 7: Plot results ──────────────────────────────────────────────────
    print("\n[Step 7] Generating plots...")
 
    plot_reliability_diagram(
        bins_before, bins_after,
        ece_before, ece_after,
        save_path=f"{RESULTS_DIR}/reliability_diagram.png",
    )
 
    plot_confidence_histogram(
        test_logits,
        test_logits / T,
        save_path=f"{RESULTS_DIR}/confidence_histogram.png",
    )
 
    plot_ece_comparison(
        {
            "ece_before": ece_before,
            "ece_after" : ece_after,
            "mce_before": mce_before,
            "mce_after" : mce_after,
        },
        save_path=f"{RESULTS_DIR}/ece_comparison.png",
    )
 
    # ── Step 8: Summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Optimal Temperature (T)  : {T:.4f}")
    print(f"  {'Metric':<25} {'Before':>10} {'After':>10} {'Δ':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Accuracy':<25} {acc_before*100:>9.2f}% {acc_after*100:>9.2f}% {'—':>10}")
    print(f"  {'ECE (↓ better)':<25} {ece_before:>10.4f} {ece_after:>10.4f} {ece_after - ece_before:>+10.4f}")
    print(f"  {'MCE (↓ better)':<25} {mce_before:>10.4f} {mce_after:>10.4f} {mce_after - mce_before:>+10.4f}")
    print("=" * 60)
    print(f"\n  Plots saved to ./{RESULTS_DIR}/")
 
 
if __name__ == "__main__":
    main()