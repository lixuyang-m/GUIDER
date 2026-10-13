from unittest.mock import patch

import numpy as np
import pytest
import torch

from guider.policies.guidance import GuidanceGate, guidance_distribution, quantile_uncertainty
from guider.training.agent import select_action


def test_quantile_order_invariance():
    taus = torch.linspace(0, 1, 32).reshape(1, 32, 1)
    quantiles = taus.expand(-1, -1, 9)
    expected = quantile_uncertainty(quantiles, taus)
    order = torch.randperm(32)
    actual = quantile_uncertainty(quantiles[:, order], taus[:, order])
    torch.testing.assert_close(expected, torch.tensor([29 / 31]))
    torch.testing.assert_close(actual, expected)


def test_quantile_order_uses_taus():
    taus = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1) / 10
    values = torch.arange(10, 0, -1, dtype=torch.float32).reshape(1, 10, 1)
    torch.testing.assert_close(quantile_uncertainty(values, taus), torch.tensor([-9.0]))


def test_fine_guidance():
    fine = torch.tensor([0.92] + [0.01] * 8)
    coarse = torch.tensor([0.05, 0.90, 0.05])
    torch.testing.assert_close(guidance_distribution(fine, coarse, 0.9, 0.9), fine)


def test_coarse_guidance():
    fine = torch.tensor([0.10] * 8 + [0.20])
    coarse = torch.tensor([0.95, 0.03, 0.02])
    actual = guidance_distribution(fine, coarse, 0.9, 0.9)
    torch.testing.assert_close(actual, torch.tensor([0.95 / 3, 0.01, 0.02 / 3] * 3))
    torch.testing.assert_close(actual.sum(), torch.tensor(1.0))


@pytest.mark.parametrize("epsilon,expected", [(0.0, 8), (1.0, 2)])
def test_rejected_guidance(epsilon, expected):
    probabilities = guidance_distribution(torch.full((9,), 1 / 9), torch.full((3,), 1 / 3), 0.9, 0.9)
    assert probabilities is None
    with patch("random.randrange", return_value=2):
        action, guided = select_action(torch.arange(9, dtype=torch.float32), probabilities, epsilon, 1.0)
    assert action == expected
    assert not guided


def test_guidance_flag():
    action, guided = select_action(torch.ones(9), torch.tensor([0.0] * 8 + [1.0]), 1.0, 1.0)
    assert action == 8
    assert guided


def test_rolling_threshold():
    gate = GuidanceGate(window_size=4, percentile=75, initial_threshold=10)
    assert not gate.observe(1)[0]
    gate.observe(2)
    gate.observe(3)
    triggered, threshold = gate.observe(8)
    assert triggered
    assert threshold == np.percentile([1, 2, 3, 8], 75)
    gate.observe(0)
    assert list(gate.window) == [2, 3, 8, 0]
