import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from guider.agent import METHODS, Policy, checkpoint_path, load_model
from guider.evaluation import make_environment, run_episode, summarize


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_hashes():
    manifest = json.loads((ROOT / "checkpoints/manifest.json").read_text())
    assert len(manifest) == 180
    for filename, digest in manifest.items():
        assert hashlib.sha256((ROOT / "checkpoints" / filename).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("domain", ["terrestrial", "maritime"])
@pytest.mark.parametrize("family", ["iqn", "pgm_iqn", "ieqn"])
@pytest.mark.parametrize("seed", range(1, 6))
def test_checkpoints(domain, family, seed):
    model = load_model(ROOT / "checkpoints" / domain / family / f"seed_{seed}", family)
    with torch.inference_mode():
        output = model(torch.zeros(1, 39), 32, 0.5)
    quantiles = output["quantiles"] if isinstance(output, dict) else output[0]
    assert quantiles.shape == (1, 32, 9)
    assert torch.isfinite(quantiles).all()


@pytest.mark.parametrize("domain", ["terrestrial", "maritime"])
@pytest.mark.parametrize("method", METHODS)
def test_rollouts(domain, method):
    scene = json.loads((ROOT / "configs/eval" / f"{domain}.json").read_text())[0]
    env = make_environment(domain, 1001)
    observations, _, _ = env.reset_with_eval_config(scene)
    policy = Policy(method, domain, checkpoint_path(ROOT / "checkpoints", domain, method, 1))
    result = run_episode(env, observations, policy, 1001, 2)
    assert result["steps"] == 2
    for robot, record in zip(env.robots, result["robots"]):
        assert len(robot.trajectory) == record["steps"] + 1
        assert len(robot.action_history) == len(record["cvar"]) == record["steps"]
        assert np.isfinite(robot.trajectory).all()
    summary = summarize([result])
    assert np.isclose(sum(summary[key] for key in (
        "robot_success_rate", "robot_collision_rate", "robot_timeout_rate")), 1)
    if domain == "maritime":
        assert len(env.cores) > 0
        assert np.linalg.norm(env.get_velocity(env.robots[0].x, env.robots[0].y)) > 0
    env.close()
