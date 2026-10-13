import copy
import json
from pathlib import Path
import random

import numpy as np
import torch

from ..agent import Policy
from ..evaluation import make_environment, run_episode, seed_inference, summarize, validate_scene, write_json
from ..policies.guidance import GuidanceGate, GuidanceModel
from .agent import TrainingAgent, select_action


def validate_config(config):
    for key in ("total_timesteps", "batch_size", "buffer_size", "update_every", "target_update_interval",
                "save_every", "max_episode_steps", "action_quantiles", "train_quantiles", "uncertainty_window"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(key)
    if config["buffer_size"] < config["batch_size"]:
        raise ValueError("buffer_size < batch_size")
    for key in ("gamma", "initial_epsilon", "final_epsilon", "fine_threshold", "coarse_threshold",
                "priority_alpha", "priority_beta", "guidance_weight_start", "guidance_weight_end"):
        if not 0 <= config[key] <= 1:
            raise ValueError(key)
    if not 0 < config["target_tau"] <= 1 or not 0 < config["exploration_fraction"] <= 1:
        raise ValueError("target_tau / exploration_fraction")
    if not 0 <= config["uncertainty_percentile"] <= 100:
        raise ValueError("uncertainty_percentile")
    if config["learning_rate"] <= 0 or config["max_grad_norm"] <= 0:
        raise ValueError("learning_rate / max_grad_norm")
    if config["learning_starts"] < 0 or config["eval_every"] < 0 or config["eval_episodes"] < 1:
        raise ValueError("learning_starts / eval_every / eval_episodes")
    schedule = config["schedule"]
    if not schedule or schedule[0]["step"] != 0:
        raise ValueError("schedule[0].step")
    if any(b["step"] <= a["step"] for a, b in zip(schedule, schedule[1:])):
        raise ValueError("schedule.step")
    for item in schedule:
        if item["robots"] < 1 or item["obstacles"] < 0 or item["cores"] < 0:
            raise ValueError("schedule.robots / obstacles / cores")
    if config["method"] not in ("iqn", "pgm_iqn"):
        raise ValueError("method")
    if config["env"] not in ("terrestrial", "maritime"):
        raise ValueError("env")
    if not 0 <= config["seed"] < 2 ** 32 - 100:
        raise ValueError("seed")


class Trainer:
    def __init__(self, config, output, device="cpu", pgm_checkpoint=None, checkpoint=None):
        validate_config(config)
        if config["method"] == "pgm_iqn" and pgm_checkpoint is None:
            raise ValueError("--pgm-checkpoint")
        self.config = copy.deepcopy(config)
        self.output = Path(output)
        self.device = device
        seed_inference(config["seed"])
        self.guidance = (GuidanceModel(pgm_checkpoint, device, config["fine_threshold"],
                                       config["coarse_threshold"])
                         if config["method"] == "pgm_iqn" else None)
        self.agent = TrainingAgent(config, device, checkpoint)
        self.env = make_environment(config["env"], config["seed"])
        self.env.collision_penalty = config["collision_penalty"]
        self.env.goal_reward = config["goal_reward"]
        self.env.timestep_penalty = config["timestep_penalty"]
        self.gates = []
        self.scenes = []
        if config["eval_every"]:
            self.scenes = json.loads(Path(config["eval_config"]).read_text())
            if not isinstance(self.scenes, list) or len(self.scenes) < config["eval_episodes"]:
                raise ValueError("eval_config / eval_episodes")
            self.scenes = self.scenes[:config["eval_episodes"]]
            for scene in self.scenes:
                validate_scene(scene, config["env"])
        self.output.mkdir(parents=True, exist_ok=False)
        write_json(self.output / "config.json", config)

    def reset(self, step):
        schedule = self.config["schedule"][0]
        for stage in self.config["schedule"]:
            if step >= stage["step"]:
                schedule = stage
        self.env.num_cooperative = schedule["robots"]
        self.env.num_non_cooperative = 0
        self.env.num_obs = schedule["obstacles"]
        self.env.min_start_goal_dis = schedule["min_start_goal_distance"]
        if self.config["env"] == "maritime":
            self.env.num_cores = schedule["cores"]
        observations, _, _ = self.env.reset()
        if len(self.env.robots) != schedule["robots"] or len(self.env.obstacles) != schedule["obstacles"]:
            raise RuntimeError("Scene placement failed")
        if self.config["env"] == "maritime" and len(self.env.cores) != schedule["cores"]:
            raise RuntimeError("Core placement failed")
        while len(self.gates) < len(self.env.robots):
            self.gates.append(GuidanceGate(self.config["uncertainty_window"],
                                           self.config["uncertainty_percentile"],
                                           self.config["uncertainty_initial_threshold"]))
        return observations

    def evaluate(self, step, checkpoint):
        states = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            policy = Policy(self.config["method"], self.config["env"], checkpoint, self.device)
            episodes = []
            for index, scene in enumerate(self.scenes):
                env = make_environment(self.config["env"], index)
                try:
                    observations, _, _ = env.reset_with_eval_config(scene)
                    episodes.append(run_episode(env, observations, policy, 1001 + index,
                                                self.config["eval_max_steps"]))
                finally:
                    env.close()
            directory = self.output / "evaluations"
            directory.mkdir(exist_ok=True)
            write_json(directory / f"step_{step}.json", summarize(episodes))
        finally:
            random.setstate(states[0])
            np.random.set_state(states[1])
            torch.set_rng_state(states[2])
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def run(self):
        config = self.config
        observations = self.reset(0)
        episode_steps = 0
        episode_reward = 0.0
        episodes = 0
        updates = 0
        guided_total = 0
        last_loss = None
        saved_step = 0
        try:
            with (self.output / "metrics.jsonl").open("w", encoding="utf-8") as log:
                for step in range(1, config["total_timesteps"] + 1):
                    progress = min((step - 1) / (config["total_timesteps"] * config["exploration_fraction"]), 1.0)
                    epsilon = config["initial_epsilon"] + progress * (config["final_epsilon"] - config["initial_epsilon"])
                    fraction = min((step - 1) / config["total_timesteps"], 1.0)
                    weight = config["guidance_weight_start"] + fraction * (
                        config["guidance_weight_end"] - config["guidance_weight_start"])
                    active = [i for i, robot in enumerate(self.env.robots) if not robot.deactivated]
                    actions = [0] * len(self.env.robots)
                    guided = [False] * len(self.env.robots)
                    for i in active:
                        values, uncertainty = self.agent.action_values(observations[i])
                        probabilities = None
                        if self.guidance is not None:
                            intervene, _ = self.gates[i].observe(uncertainty)
                            if intervene:
                                probabilities = self.guidance.predict(observations[i])
                        actions[i], guided[i] = select_action(values, probabilities, epsilon, weight)
                    next_observations, rewards, dones, _ = self.env.step(actions, coll_robots_list=[])
                    episode_steps += 1
                    for i in active:
                        self.agent.replay.add(observations[i], actions[i], rewards[i],
                                              next_observations[i], dones[i], guided[i])
                        episode_reward += rewards[i]
                        guided_total += int(guided[i])
                        if dones[i]:
                            self.env.robots[i].deactivated = True
                    if step >= config["learning_starts"] and len(self.agent.replay) >= config["batch_size"]:
                        if step % config["update_every"] == 0:
                            last_loss = self.agent.update()
                            updates += 1
                        if step % config["target_update_interval"] == 0:
                            self.agent.update_target()
                    ended = self.env.check_all_deactivated() or episode_steps >= config["max_episode_steps"]
                    if ended:
                        episodes += 1
                        metrics = {"step": step, "episode": episodes, "episode_steps": episode_steps,
                                   "reward": float(episode_reward), "loss": last_loss,
                                   "updates": updates, "guided_actions": guided_total}
                        log.write(json.dumps(metrics, allow_nan=False) + "\n")
                        log.flush()
                        print(json.dumps(metrics), flush=True)
                        if step < config["total_timesteps"]:
                            observations = self.reset(step)
                            episode_steps, episode_reward = 0, 0.0
                    else:
                        observations = next_observations
                    evaluate_now = bool(config["eval_every"] and step % config["eval_every"] == 0)
                    if step % config["save_every"] == 0 or evaluate_now or step == config["total_timesteps"]:
                        checkpoint = self.output / "checkpoints" / f"step_{step}"
                        self.agent.save(checkpoint, step)
                        saved_step = step
                        if evaluate_now:
                            self.evaluate(step, checkpoint)
                summary = {"steps": config["total_timesteps"], "episodes": episodes,
                           "updates": updates, "guided_actions": guided_total, "loss": last_loss,
                           "checkpoint": str(self.output / "checkpoints" / f"step_{saved_step}")}
                write_json(self.output / "summary.json", summary)
                print(json.dumps(summary), flush=True)
                return summary
        finally:
            self.env.close()
