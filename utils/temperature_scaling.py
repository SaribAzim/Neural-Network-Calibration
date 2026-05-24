"""
temperature_scaling.py
----------------------
Temperature Scaling — the simplest post-hoc calibration method.

The idea
--------
After training a model, we have raw logits z (a vector of scores per class).
We normally apply softmax to get probabilities:
    p = softmax(z)

Temperature scaling adds a single scalar T (the "temperature"):
    p = softmax(z / T)

- T > 1  →  softens the distribution  →  makes the model LESS confident
- T < 1  →  sharpens the distribution →  makes the model MORE confident
- T = 1  →  no change (original model)

We find the best T by minimising Negative Log-Likelihood (NLL) on the
validation set. NLL penalises overconfident wrong predictions heavily,
so minimising it pushes T toward good calibration.

This is a 1-parameter optimisation — very fast.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    """
    Wraps a single learnable temperature parameter T.

    Usage
    -----
        scaler = TemperatureScaler()
        scaler.fit(val_logits, val_labels)          # finds best T
        calibrated_probs = scaler.calibrate(logits) # apply to any logits
    """

    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        # nn.Parameter makes T learnable via autograd
        self.temperature = nn.Parameter(
            torch.tensor([init_temperature], dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Divide logits by T. Shape unchanged."""
        return logits / self.clamp_temperature()

    def clamp_temperature(self) -> torch.Tensor:
        """Keep T strictly positive to avoid numerical issues."""
        return self.temperature.clamp(min=1e-3)

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 1000,
        verbose: bool = True,
    ) -> "TemperatureScaler":
        """
        Optimise T on validation logits by minimising cross-entropy (NLL).

        Args:
            val_logits : shape (N, C) — raw logits from the model on val set
            val_labels : shape (N,)   — ground-truth class indices
            lr         : learning rate for L-BFGS optimiser
            max_iter   : maximum optimisation iterations
            verbose    : print T before and after fitting

        Returns:
            self (for chaining)
        """
        if verbose:
            print(f"[TemperatureScaler] Initial T = {self.temperature.item():.4f}")

        criterion = nn.CrossEntropyLoss()
        # L-BFGS is a second-order optimiser — converges much faster than
        # SGD/Adam for this 1-parameter problem.
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        if verbose:
            print(f"[TemperatureScaler] Optimal  T = {self.clamp_temperature().item():.4f}")

        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling and return calibrated probabilities.

        Args:
            logits : shape (N, C) — raw model logits

        Returns:
            probs  : shape (N, C) — calibrated softmax probabilities
        """
        with torch.no_grad():
            scaled = self.forward(logits)
            return torch.softmax(scaled, dim=1)

    @property
    def T(self) -> float:
        """Convenience property: returns T as a plain Python float."""
        return self.clamp_temperature().item()