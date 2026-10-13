from collections import deque
from pathlib import Path

import numpy as np
import torch

from ..models.pgm import ImprovedDualTaskTransformer


def quantile_uncertainty(quantiles, taus):
    indices = taus.squeeze(-1).argsort(dim=1)
    ordered = quantiles.gather(1, indices.unsqueeze(-1).expand_as(quantiles))
    count = max(1, int(ordered.shape[1] * 0.1))
    spread = ordered[:, -count:].mean(dim=1) - ordered[:, :count].mean(dim=1)
    return spread.max(dim=1).values


def guidance_distribution(fine_probs, coarse_probs, fine_threshold, coarse_threshold):
    if fine_probs.max().item() >= fine_threshold:
        return fine_probs
    if coarse_probs.max().item() >= coarse_threshold:
        return coarse_probs.repeat(3) / 3.0
    return None


class GuidanceGate:
    def __init__(self, window_size=100, percentile=90, initial_threshold=0.2):
        self.window = deque(maxlen=window_size)
        self.percentile = percentile
        self.initial_threshold = initial_threshold

    def observe(self, value):
        self.window.append(float(value))
        threshold = (float(np.percentile(self.window, self.percentile))
                     if len(self.window) == self.window.maxlen else self.initial_threshold)
        return value > threshold, threshold


class GuidanceModel:
    def __init__(self, checkpoint, device="cpu", fine_threshold=0.9, coarse_threshold=0.9):
        self.device = torch.device(device)
        self.fine_threshold = fine_threshold
        self.coarse_threshold = coarse_threshold
        self.model = ImprovedDualTaskTransformer().to(self.device)
        safe_types = [np.core.multiarray.scalar, np.dtype,
                      type(np.dtype(np.float32)), type(np.dtype(np.float64))]
        with torch.serialization.safe_globals(safe_types):
            data = torch.load(Path(checkpoint), map_location=self.device, weights_only=True)
        self.model.load_state_dict(data.get("model_state_dict", data), strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, state):
        state = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        fine, coarse = self.model(state)
        return guidance_distribution(fine.softmax(dim=1)[0], coarse.softmax(dim=1)[0],
                                     self.fine_threshold, self.coarse_threshold)
