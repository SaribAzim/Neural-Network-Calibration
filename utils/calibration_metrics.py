import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def _bin_predictions(confidences, predictions, labels, n_bins=15):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= lower) & (confidences < upper)
        else:
            mask = (confidences >= lower) & (confidences <= upper)
        if mask.sum() == 0:
            continue
        bins.append({
            "bin_lower": lower, "bin_upper": upper,
            "accuracy": (predictions[mask] == labels[mask]).mean(),
            "confidence": confidences[mask].mean(),
            "count": mask.sum(),
        })
    return bins

def collect_logits_and_labels(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            all_logits.append(model(images.to(device)).cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)

def compute_reliability_data(logits, labels, n_bins=15, temperature=1.0):
    probs = F.softmax(logits / temperature, dim=1).numpy()
    return _bin_predictions(probs.max(axis=1), probs.argmax(axis=1), labels.numpy(), n_bins)

def compute_ece(logits, labels, n_bins=15, temperature=1.0):
    bins = compute_reliability_data(logits, labels, n_bins, temperature)
    n = len(labels)
    return float(sum(b["count"] / n * abs(b["accuracy"] - b["confidence"]) for b in bins))

def compute_mce(logits, labels, n_bins=15, temperature=1.0):
    bins = compute_reliability_data(logits, labels, n_bins, temperature)
    return float(max(abs(b["accuracy"] - b["confidence"]) for b in bins)) if bins else 0.0

def compute_accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()
