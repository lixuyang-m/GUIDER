import json
import random
from pathlib import Path
from time import perf_counter
import numpy as np
import torch

def seed_inference(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_environment(domain, seed):
    if domain == 'terrestrial':
        from .envs.terrestrial import MarineNavEnv2
    else:
        from .envs.maritime import MarineNavEnv2
    return MarineNavEnv2(seed=seed)

def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}')

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=json_default) + '\n', encoding='utf-8')

def validate_scene(scene, domain):
    if not isinstance(scene, dict) or 'env' not in scene or 'robots' not in scene:
        raise ValueError("Missing scene fields: env, robots")
    marine_scene = 'cores' in scene['env']
    if marine_scene != (domain == 'maritime'):
        raise ValueError(f'Scene domain does not match --env {domain}; check the scene JSON file.')
    cooperative = scene['robots'].get('cooperative', [])
    if not cooperative or not all(cooperative):
        raise ValueError('This evaluation entry supports cooperative robots (39-dimensional observations).')

def run_episode(env, observations, policy, seed, max_steps):
    seed_inference(seed)
    records = [{'id': i, 'status': 'timeout', 'steps': 0, 'time_s': 0.0, 'energy': 0.0, 'reward': 0.0, 'discounted_reward': 0.0, 'decision_time_s': 0.0, 'cvar': []} for i in range(len(env.robots))]
    for robot, record in zip(env.robots, records):
        if robot.collision or robot.reach_goal:
            record['status'] = 'collision' if robot.collision else 'success'
            robot.deactivated = True
    episode_steps = 0
    for step in range(max_steps):
        if env.check_all_deactivated():
            break
        active = [i for i, robot in enumerate(env.robots) if not robot.deactivated]
        actions = [0] * len(env.robots)
        for i in active:
            robot = env.robots[i]
            start = perf_counter()
            actions[i], cvar = policy.act(observations[i], robot)
            records[i]['decision_time_s'] += perf_counter() - start
            records[i]['cvar'].append(cvar)
            if not 0 <= actions[i] < robot.compute_actions_dimension():
                raise ValueError(f'Invalid action {actions[i]} for robot {i}.')
        observations, rewards, dones, infos = env.step(actions, coll_robots_list=[])
        episode_steps = step + 1
        for i in active:
            robot, record = (env.robots[i], records[i])
            record['steps'] += 1
            record['time_s'] += float(robot.dt * robot.N)
            record['energy'] += float(robot.compute_action_energy_cost(actions[i]))
            record['reward'] += float(rewards[i])
            record['discounted_reward'] += float(rewards[i]) * 0.99 ** step
            if robot.collision:
                record['status'] = 'collision'
            elif robot.reach_goal:
                record['status'] = 'success'
            if dones[i]:
                robot.deactivated = True
    return {'inference_seed': seed, 'steps': episode_steps, 'team_success': all((r['status'] == 'success' for r in records)), 'robots': records, 'scene': env.episode_data()}

def summarize(episodes):
    robots = [robot for episode in episodes for robot in episode['robots']]
    successes = [robot for robot in robots if robot['status'] == 'success']
    decisions = sum((robot['steps'] for robot in robots))
    mean = lambda values: float(np.mean(values)) if values else None
    return {'episodes': len(episodes), 'robots': len(robots), 'team_success_rate': mean([e['team_success'] for e in episodes]), 'robot_success_rate': mean([r['status'] == 'success' for r in robots]), 'robot_collision_rate': mean([r['status'] == 'collision' for r in robots]), 'robot_timeout_rate': mean([r['status'] == 'timeout' for r in robots]), 'mean_success_time_s': mean([r['time_s'] for r in successes]), 'mean_success_energy': mean([r['energy'] for r in successes]), 'mean_robot_reward': mean([r['reward'] for r in robots]), 'mean_robot_discounted_reward': mean([r['discounted_reward'] for r in robots]), 'mean_decision_time_ms': 1000 * sum((r['decision_time_s'] for r in robots)) / decisions if decisions else None}
