import copy
import json
from pathlib import Path
import random

import numpy as np
import torch

from ..agent import load_model
from ..evaluation import write_json
from ..models.iqn import IQN_Policy
from ..policies.guidance import quantile_uncertainty
from .replay import ReplayBuffer


def select_action(values, probabilities, epsilon, weight):
    if probabilities is not None:
        offset = -values.min() + 1 if values.min().item() < 0 else 0
        guidance_values = (values + offset) * probabilities - offset
        return int(((1 - weight) * values + weight * guidance_values).argmax().item()), True
    if random.random() < epsilon:
        return random.randrange(values.numel()), False
    return int(values.argmax().item()), False


class TrainingAgent:
    def __init__(self, config, device="cpu", checkpoint=None):
        self.config = config
        self.device = torch.device(device)
        self.constructor = dict(config["network"], seed=config["seed"] + 100)
        if checkpoint is None:
            self.model = IQN_Policy(**self.constructor, device=str(self.device)).to(self.device)
        else:
            self.model = load_model(checkpoint, "iqn", self.device)
            self.constructor = json.loads((Path(checkpoint) / "constructor_params.json").read_text())
            self.constructor.pop("device", None)
        self.target = copy.deepcopy(self.model).eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.replay = ReplayBuffer(config["buffer_size"], config["batch_size"],
                                   config["priority_alpha"], config["priority_beta"],
                                   config["guidance_priority_scale"])

    @torch.inference_mode()
    def action_values(self, observation):
        self.model.eval()
        state = torch.as_tensor(np.asarray(observation), dtype=torch.float32, device=self.device).unsqueeze(0)
        quantiles, taus = self.model(state, self.config["action_quantiles"], 1.0)
        return quantiles.mean(dim=1)[0], quantile_uncertainty(quantiles, taus).item()

    def update(self):
        self.model.train()
        batch = self.replay.sample()
        to_tensor = lambda values, dtype: torch.as_tensor(values, dtype=dtype, device=self.device)
        states = to_tensor(batch["states"], torch.float32)
        next_states = to_tensor(batch["next_states"], torch.float32)
        actions = to_tensor(batch["actions"], torch.long)
        rewards = to_tensor(batch["rewards"], torch.float32)
        dones = to_tensor(batch["dones"], torch.float32)
        weights = to_tensor(batch["weights"], torch.float32)
        guided = to_tensor(batch["guided"], torch.bool)
        count = self.config["train_quantiles"]
        with torch.no_grad():
            targets, _ = self.target(next_states, count, 1.0)
            targets = rewards[:, None, None] + self.config["gamma"] * (
                1 - dones[:, None, None]) * targets.max(dim=2).values[:, None, :]
        predictions, taus = self.model(states, count, 1.0)
        predictions = predictions.gather(2, actions[:, None, None].expand(-1, count, 1))
        errors = targets - predictions
        huber = torch.where(errors.abs() <= 1, 0.5 * errors.square(), errors.abs() - 0.5)
        losses = ((taus - (errors.detach() < 0).float()).abs() * huber).sum(dim=1).mean(dim=1)
        loss = sum((losses[mask] * weights[mask]).mean()
                   for mask in (~guided, guided) if mask.any())
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
        self.optimizer.step()
        self.replay.update_priorities(batch["indices"], errors.detach().mean(dim=(1, 2)).cpu().numpy())
        return float(loss.detach())

    @torch.no_grad()
    def update_target(self):
        tau = self.config["target_tau"]
        for target, source in zip(self.target.parameters(), self.model.parameters()):
            target.lerp_(source, tau)

    def save(self, directory, step):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "network_params.pth")
        write_json(directory / "constructor_params.json", self.constructor)
        write_json(directory / "trial_config.json", dict(self.config, completed_steps=step))
        torch.save({"step": step, "optimizer": self.optimizer.state_dict(),
                    "target": self.target.state_dict()}, directory / "training_state.pth")
        torch.save(torch.get_rng_state(), directory / "rng_state.pth")
        if torch.cuda.is_available():
            torch.save(torch.cuda.get_rng_state_all(), directory / "cuda_rng_state.pth")
