import json
from pathlib import Path

import numpy as np
import torch

from guider.training.agent import TrainingAgent
from guider.training.replay import ReplayBuffer, SumTree


ROOT = Path(__file__).resolve().parents[1]


def test_replay_indices_and_weights():
    replay = ReplayBuffer(8, 8, alpha=0.6, beta=0.4)
    for i in range(8):
        state = np.full(39, i, dtype=np.float32)
        replay.add(state, i % 9, float(i), state + 1, False, guided=i % 2 == 0)
    batch = replay.sample()
    for row, index in enumerate(batch["indices"]):
        stored = replay.tree.data[index - replay.tree.capacity + 1]
        np.testing.assert_array_equal(batch["states"][row], stored[0])
        assert batch["guided"][row] == stored[5]
    probabilities = replay.tree.tree[batch["indices"]] / replay.tree.total
    expected = (len(replay) * probabilities) ** (-0.4)
    np.testing.assert_allclose(batch["weights"], expected / expected.max())
    errors = np.arange(1, 9, dtype=np.float32)
    replay.update_priorities(batch["indices"], errors)
    for index in np.unique(batch["indices"]):
        expected = (errors[batch["indices"] == index].max() + 1e-5) ** 0.6
        np.testing.assert_allclose(replay.tree.tree[index], expected)


def test_tree_capacity_one():
    tree = SumTree(1)
    tree.add(2.0, "a")
    assert tree.get(1.0) == (0, 2.0, "a")
    tree.add(3.0, "b")
    assert tree.get(2.0) == (0, 3.0, "b")
    assert tree.size == 1


def test_priority_updates_follow_sample_order(monkeypatch):
    config = json.loads((ROOT / "configs/train/terrestrial.json").read_text())
    config.update(buffer_size=8, batch_size=4)
    agent = TrainingAgent(config)

    class Values(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, state, count, cvar):
            values = state[:, :1, None].expand(-1, count, 9) * self.scale
            taus = torch.full((len(state), count, 1), 0.5)
            return values, taus

    agent.model = Values()
    agent.target = Values()
    agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=1e-3)
    states = np.repeat(np.arange(1, 5, dtype=np.float32)[:, None], 39, axis=1)
    batch = {"states": states, "next_states": states, "actions": np.zeros(4, dtype=np.int64),
             "rewards": np.zeros(4, dtype=np.float32), "dones": np.ones(4, dtype=bool),
             "guided": np.array([True, False, True, False]), "weights": np.ones(4, dtype=np.float32),
             "indices": np.array([7, 3, 6, 4])}
    monkeypatch.setattr(agent.replay, "sample", lambda: batch)
    captured = []
    monkeypatch.setattr(agent.replay, "update_priorities", lambda indices, errors: captured.append((indices, errors)))
    assert np.isfinite(agent.update())
    np.testing.assert_array_equal(captured[0][0], batch["indices"])
    np.testing.assert_allclose(captured[0][1], -np.arange(1, 5))
    assert agent.model.scale.item() != 1.0
