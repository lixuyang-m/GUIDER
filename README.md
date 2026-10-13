# GUIDER

Code and model checkpoints for **Guided Distillation and Risk Adaptive Evolution for Multi-Robot Navigation** (AAAI 2026).

GUIDER supports cooperative multi-robot navigation in terrestrial and maritime environments. The repository includes navigation policies, a Policy Guidance Model (PGM), environment-specific risk rules, training, evaluation, and trajectory visualization.

## Installation

Python 3.10 and PyTorch 2.5.1. Package versions are listed in [requirements.txt](requirements.txt).

```bash
git clone https://github.com/lixuyang-m/GUIDER.git
cd GUIDER
conda create -n guider python=3.10 -y
conda activate guider
python -m pip install -r requirements.txt
```

The command-line tools default to `--device cpu`. Use `--device cuda:0` with a CUDA-capable PyTorch installation and an available GPU.

## Repository Structure

| Path | Contents |
| --- | --- |
| [guider/envs/](guider/envs/) | Terrestrial and maritime environments and robot dynamics |
| [guider/models/](guider/models/) | IQN, IEQN, and PGM networks |
| [guider/policies/](guider/policies/) | PGM guidance and uncertainty thresholds |
| [guider/risks/](guider/risks/) | Terrestrial, maritime, and distance-based CVaR rules |
| [guider/training/](guider/training/) | Navigation training and experience replay |
| [guider/baselines/](guider/baselines/) | IAPF and RVO controllers |
| [configs/train/](configs/train/) | Navigation training configurations |
| [configs/eval/](configs/eval/) | Fixed evaluation scenes |
| [checkpoints/](checkpoints/) | Navigation model checkpoints and metadata |
| [data/pgm.npz](data/pgm.npz) | PGM training dataset |
| [evaluate.py](evaluate.py) | Navigation evaluation |
| [train.py](train.py) | IQN and PGM-IQN training |
| [train_pgm.py](train_pgm.py) | PGM training |
| [plot_trajectories.py](plot_trajectories.py) | Trajectory plotting |
| [tests/](tests/) | Automated tests |

## Model Checkpoints

The repository contains 30 navigation checkpoints: two environments, three model families, and five checkpoint indices per family.

```text
checkpoints/
├── terrestrial/
│   ├── iqn/seed_1/ ... seed_5/
│   ├── pgm_iqn/seed_1/ ... seed_5/
│   └── ieqn/seed_1/ ... seed_5/
├── maritime/
│   ├── iqn/seed_1/ ... seed_5/
│   ├── pgm_iqn/seed_1/ ... seed_5/
│   └── ieqn/seed_1/ ... seed_5/
└── manifest.json
```

A navigation checkpoint directory must contain `network_params.pth` and `constructor_params.json`. File checksums are listed in [checkpoints/manifest.json](checkpoints/manifest.json).

`--model-seed` selects the checkpoint directory index, from 1 to 5. `--checkpoint` selects a specific directory. `--weights-root` selects an alternative root with the same directory structure. Select the environment and method matching the checkpoint.

### Evaluation Methods

| `--method` | Checkpoint family | Risk setting |
| --- | --- | --- |
| `guider` | `pgm_iqn` | Environment-specific GUIDER rule |
| `pgm_iqn` | `pgm_iqn` | CVaR = 1 |
| `iqn` | `iqn` | CVaR = 1 |
| `ada_iqn` | `iqn` | Nearest-obstacle distance |
| `art_iqn` | `iqn` | Return-distribution risk metric |
| `guider_ada` | `pgm_iqn` | Nearest-obstacle distance |
| `guider_art` | `pgm_iqn` | Return-distribution risk metric |
| `ieqn` | `ieqn` | CVaR = 1 |
| `iapf` | None | Potential-field controller |
| `rvo` | None | Reciprocal velocity-obstacle controller |

The GUIDER rules are [guider/risks/terrestrial.py](guider/risks/terrestrial.py) and [guider/risks/maritime.py](guider/risks/maritime.py).

## Evaluation

### Terrestrial and Maritime Environments

```bash
python evaluate.py --env terrestrial --method guider --model-seed 1 --episodes 1 --output results/terrestrial_guider
python evaluate.py --env maritime --method guider --model-seed 1 --episodes 1 --output results/maritime_guider
```

Each environment has 30 fixed scenes in `configs/eval/`. Use `--episodes 30` to evaluate all scenes. All methods use the selected environment's scene file.

### Custom Checkpoints and Scenes

```bash
python evaluate.py --env maritime --method guider --checkpoint /path/to/checkpoint --episodes 30
python evaluate.py --env terrestrial --scene-config /path/to/scenes.json --start-index 5 --episodes 3
python evaluate.py --env maritime --method ieqn --model-seed 1 --episodes 1
```

`--scene-config` accepts a scene list, a single scene, or a saved `episode_*.json` file. Scenes use cooperative robots with 39-dimensional observations and nine discrete actions.

### Random Scenes

```bash
python evaluate.py --env terrestrial --random-scenes --robots 3 --obstacles 8 --seed 1001
python evaluate.py --env maritime --random-scenes --robots 3 --obstacles 8 --cores 6 --seed 1001
```

### Evaluation Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--env` | `terrestrial` | `terrestrial` or `maritime` |
| `--method` | `guider` | Method from the evaluation table |
| `--model-seed` | `1` | Checkpoint directory index |
| `--checkpoint` | Automatic selection | Navigation checkpoint directory |
| `--weights-root` | `checkpoints/` | Root directory for automatic checkpoint selection |
| `--episodes` | `1` | Number of episodes |
| `--max-steps` | `361` | Maximum control steps per episode |
| `--seed` | `1001` | Initial inference seed; also the initial scene seed for random scenes |
| `--scene-config` | `configs/eval/<env>.json` | Fixed scene file |
| `--start-index` | `0` | Zero-based index of the first fixed scene |
| `--random-scenes` | Disabled | Generate random scenes |
| `--robots` | `3` in random scenes | Number of cooperative robots |
| `--obstacles` | `8` in random scenes | Number of static obstacles |
| `--cores` | `6` in random maritime scenes | Number of vortex cores |
| `--device` | `cpu` | PyTorch device |
| `--threads` | `1` | PyTorch CPU thread count |
| `--output` | `results/<timestamp>_<env>_<method>_<id>/` | New output directory |

`--robots`, `--obstacles`, and `--cores` require `--random-scenes`. `--cores` applies to maritime environments. `--random-scenes` uses generated scenes without `--scene-config` or `--start-index`. Seeds increment by one per episode. Output directories supplied with `--output` must not already exist.

## Training

### Navigation Policies

```bash
python train.py --env terrestrial --method iqn --device cuda:0 --seed 1
python train.py --env maritime --method iqn --device cuda:0 --seed 1
python train.py --env terrestrial --method pgm_iqn --pgm-checkpoint /path/to/best_model_9.pth --device cuda:0
python train.py --env maritime --method pgm_iqn --pgm-checkpoint /path/to/best_model_9.pth --device cuda:0
```

PGM-IQN requires a PGM checkpoint with nine fine-action outputs and three coarse-action outputs. Supply the checkpoint file with `--pgm-checkpoint`.

The environment configurations are [configs/train/terrestrial.json](configs/train/terrestrial.json) and [configs/train/maritime.json](configs/train/maritime.json).

```bash
python train.py --config configs/train/terrestrial.json --method iqn --output outputs/terrestrial_iqn
python train.py --env terrestrial --method iqn --checkpoint checkpoints/terrestrial/iqn/seed_1 --output outputs/terrestrial_iqn_initialized
```

| Argument | Default | Description |
| --- | --- | --- |
| `--env` | Configuration value; `terrestrial` without arguments | Navigation environment |
| `--method` | Configuration value; `iqn` in supplied configurations | `iqn` or `pgm_iqn` |
| `--config` | `configs/train/<env>.json` | Training configuration file |
| `--seed` | `1` in supplied configurations | Training seed |
| `--total-timesteps` | `1500000` in supplied configurations | Number of environment steps |
| `--pgm-checkpoint` | Unset | PGM guidance checkpoint file |
| `--checkpoint` | Unset | Initial navigation network checkpoint directory |
| `--device` | `cpu` | PyTorch device |
| `--threads` | `1` | PyTorch CPU thread count |
| `--output` | `outputs/<timestamp>_<env>_<method>_<id>/` | New output directory |

Command-line values for `env`, `method`, `seed`, and `total_timesteps` override the corresponding configuration values. `--checkpoint` initializes network parameters; the optimizer and replay buffer start from their initial state.

### Navigation Configuration

| Field | Default | Description |
| --- | --- | --- |
| `learning_rate` | `0.0001` | Optimizer learning rate |
| `batch_size` | `32` | Replay batch size |
| `buffer_size` | `2000000` | Replay capacity |
| `gamma` | `0.99` | Discount factor |
| `learning_starts` | `2000` | First environment step eligible for updates |
| `update_every` | `4` | Model update interval |
| `target_update_interval` | `10000` | Target-network update interval |
| `action_quantiles` | `32` | Quantile samples for action selection |
| `train_quantiles` | `8` | Quantile samples for optimization |
| `save_every` | `15000` | Checkpoint interval |
| `eval_every` | `15000` | Evaluation interval; `0` disables periodic evaluation |
| `eval_episodes` | `30` | Episodes per periodic evaluation |
| `fine_threshold`, `coarse_threshold` | `0.9`, `0.9` | PGM confidence thresholds |
| `schedule` | Environment configuration | Step, robot count, obstacle count, core count, and minimum start-to-goal distance |

### PGM

```bash
python train_pgm.py --dataset data/pgm.npz --device cuda:0 --output outputs/pgm
python train.py --env terrestrial --method pgm_iqn --pgm-checkpoint outputs/pgm/best_model.pth --device cuda:0
```

The dataset is an NPZ file whose first array has shape `(N, 40)`: 39 state features followed by an integer action label from 0 to 8. The supplied `data/pgm.npz` contains 1,348 samples. `--dataset` accepts one or more NPZ paths.

| Argument | Default | Description |
| --- | --- | --- |
| `--dataset` | `data/pgm.npz` | Dataset paths |
| `--epochs` | `100` | Training epochs |
| `--batch-size` | `64` | Batch size |
| `--learning-rate` | `0.0001` | Optimizer learning rate |
| `--seed` | `42` | Data split and training seed |
| `--fine-threshold` | `0.9` | Fine-action acceptance threshold |
| `--coarse-threshold` | `0.9` | Coarse-action acceptance threshold |
| `--device` | `cpu` | PyTorch device |
| `--threads` | `1` | PyTorch CPU thread count |
| `--output` | `outputs/pgm/` | New output directory |

## Results and Visualization

### Evaluation Files

| File | Contents |
| --- | --- |
| `run.json` | Arguments, checkpoint path and checksum, scene source, and rule path |
| `episode_0000.json`, ... | Robot status, rewards, elapsed time, action cost, CVaR values, actions, trajectories, and scene data |
| `summary.json` | Aggregate evaluation metrics |

| Summary field | Description |
| --- | --- |
| `episodes` | Number of evaluated episodes |
| `robots` | Total robot instances across episodes |
| `team_success_rate` | Fraction of episodes in which all robots succeed |
| `robot_success_rate` | Fraction of robot instances reaching their goals |
| `robot_collision_rate` | Fraction of robot instances ending in collision |
| `robot_timeout_rate` | Fraction of robot instances ending without success or collision |
| `mean_success_time_s` | Mean elapsed simulation time for successful robots, in seconds |
| `mean_success_energy` | Mean accumulated normalized action cost for successful robots |
| `mean_robot_reward` | Mean undiscounted reward per robot instance |
| `mean_robot_discounted_reward` | Mean discounted reward per robot instance, with discount factor 0.99 |
| `mean_decision_time_ms` | Mean wall-clock action-selection time per robot decision, in milliseconds |

Success-time and success-energy fields are `null` when no robot succeeds. Simultaneous collision and goal arrival is recorded as a collision.

### Training Files

Navigation runs save `config.json`, `metrics.jsonl`, `summary.json`, model files under `checkpoints/step_<step>/`, and periodic metrics under `evaluations/`.

PGM runs save `best_model.pth`, `config.json`, `metrics.jsonl`, `split_indices.npz`, and `test.json`. `best_model.pth` is selected by validation fine-action accuracy.

### Trajectory Plots

```bash
python plot_trajectories.py results/terrestrial_guider/episode_0000.json --output results/terrestrial_guider/trajectory.png
```

The plot shows robot paths, start positions, goals, obstacles, and maritime vortex-core positions. Without `--output`, the image uses the episode JSON's basename with a `.png` extension.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Citation

```bibtex
@inproceedings{li2026guider,
  title     = {Guided Distillation and Risk Adaptive Evolution for Multi-Robot Navigation},
  author    = {Li, Xuyang and Fang, Jianwu and Li, Lin and Chen, Boyuan and Li, Guangliang and Xue, Jianru},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```

## License

[MIT License](LICENSE).
