import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from guider.agent import load_model
from guider.models.pgm import ImprovedDualTaskTransformer
from guider.policies.guidance import GuidanceModel
from guider.training.trainer import Trainer
from train_pgm import load_data


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("domain", ["terrestrial", "maritime"])
@pytest.mark.parametrize("method", ["iqn", "pgm_iqn"])
def test_training_roundtrip(tmp_path, domain, method):
    config = json.loads((ROOT / "configs/train" / f"{domain}.json").read_text())
    config.update(method=method, total_timesteps=12, learning_starts=0, update_every=1,
                  target_update_interval=4, batch_size=4, buffer_size=64, max_episode_steps=4,
                  save_every=12, eval_every=12, eval_episodes=1, eval_max_steps=2,
                  eval_config=str(ROOT / "configs/eval" / f"{domain}.json"),
                  uncertainty_initial_threshold=-1e6)
    pgm = None
    if method == "pgm_iqn":
        pgm = tmp_path / "pgm.pth"
        model = ImprovedDualTaskTransformer()
        with torch.no_grad():
            model.fine_classifier[-1].weight.zero_()
            model.fine_classifier[-1].bias.zero_()
            model.fine_classifier[-1].bias[1] = 20
        torch.save({"model_state_dict": model.state_dict()}, pgm)
    trainer = Trainer(config, tmp_path / "run", pgm_checkpoint=pgm)
    before = {k: v.clone() for k, v in trainer.agent.model.state_dict().items()}
    result = trainer.run()
    assert result["updates"] > 0
    assert np.isfinite(result["loss"])
    assert (result["guided_actions"] > 0) == (method == "pgm_iqn")
    assert any(not torch.equal(v, before[k]) for k, v in trainer.agent.model.state_dict().items())
    loaded = load_model(result["checkpoint"], "iqn")
    for name, tensor in trainer.agent.model.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), loaded.state_dict()[name].cpu(), rtol=0, atol=0)
    assert (tmp_path / "run/evaluations/step_12.json").is_file()


def test_pgm_training_roundtrip(tmp_path):
    generator = np.random.default_rng(0)
    rows = np.column_stack((generator.normal(size=(90, 39)), np.tile(np.arange(9), 10)))
    dataset = tmp_path / "data.npz"
    np.savez(dataset, rows)
    output = tmp_path / "pgm"
    result = subprocess.run([sys.executable, "-B", str(ROOT / "train_pgm.py"), "--dataset", str(dataset),
                             "--epochs", "1", "--batch-size", "32", "--output", str(output)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    model = GuidanceModel(output / "best_model.pth", fine_threshold=0)
    probabilities = model.predict(rows[0, :39])
    assert probabilities.shape == (9,)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))


def test_pgm_dataset():
    states, labels = load_data([ROOT / "data/pgm.npz"])
    assert states.shape == (1348, 39)
    assert labels.shape == (1348,)
    assert np.isfinite(states).all()
