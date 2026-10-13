import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import uuid
from guider.agent import METHODS, Policy, checkpoint_path
from guider.evaluation import make_environment, run_episode, summarize, validate_scene, write_json
import torch
ROOT = Path(__file__).resolve().parent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['terrestrial', 'maritime'], default='terrestrial')
    parser.add_argument('--method', choices=list(METHODS), default='guider')
    parser.add_argument('--model-seed', type=int, default=1)
    parser.add_argument('--weights-root', type=Path, default=ROOT / 'checkpoints')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=361)
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--scene-config', type=Path)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--random-scenes', action='store_true')
    parser.add_argument('--robots', type=int, default=None)
    parser.add_argument('--obstacles', type=int, default=None)
    parser.add_argument('--cores', type=int, default=None)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.episodes, args.max_steps, args.threads) < 1 or args.start_index < 0:
        parser.error('episodes, max-steps and threads must be positive; start-index must be nonnegative')
    if not 0 <= args.seed < 2 ** 32 - args.episodes:
        parser.error('seed must be nonnegative and seed + episodes must be below 2**32')
    if args.random_scenes and (args.scene_config or args.start_index):
        parser.error('random-scenes cannot be combined with scene-config or start-index')
    if not args.random_scenes and any((x is not None for x in (args.robots, args.obstacles, args.cores))):
        parser.error('robots/obstacles/cores require --random-scenes')
    if args.robots is not None and args.robots < 1:
        parser.error('robots must be positive')
    if any((x is not None and x < 0 for x in (args.obstacles, args.cores))):
        parser.error('obstacles and cores must be nonnegative')
    if args.env == 'terrestrial' and args.cores is not None:
        parser.error('cores applies only to maritime scenes')
    if args.checkpoint and METHODS[args.method][0] is None:
        parser.error('classical methods do not use a checkpoint')
    return args

def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    scenes = None
    scene_path = None
    if not args.random_scenes:
        scene_path = (args.scene_config or ROOT / 'configs' / 'eval' / f'{args.env}.json').expanduser().resolve()
        data = json.loads(scene_path.read_text(encoding='utf-8'))
        if isinstance(data, dict) and 'scene' in data:
            data = data['scene']
        scenes = data if isinstance(data, list) else [data]
        if args.start_index + args.episodes > len(scenes):
            raise ValueError(f'Only {len(scenes)} scenes available; reduce --episodes/--start-index.')
        scenes = scenes[args.start_index:args.start_index + args.episodes]
        for scene in scenes:
            validate_scene(scene, args.env)
    checkpoint = args.checkpoint or checkpoint_path(args.weights_root, args.env, args.method, args.model_seed)
    checkpoint = checkpoint.expanduser().resolve() if checkpoint else None
    policy = Policy(args.method, args.env, checkpoint, args.device)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = args.output or ROOT / 'results' / f'{stamp}_{args.env}_{args.method}_{uuid.uuid4().hex[:6]}'
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    parameters = checkpoint / 'network_params.pth' if checkpoint else None
    metadata = {'arguments': vars(args), 'checkpoint': checkpoint, 'checkpoint_sha256': hashlib.sha256(parameters.read_bytes()).hexdigest() if parameters else None, 'scene_config': scene_path, 'final_rule': f'guider/risks/{args.env}.py' if args.method == 'guider' else None, 'torch_version': torch.__version__, 'output': output}
    write_json(output / 'run.json', metadata)
    episodes = []
    for index in range(args.episodes):
        seed = args.seed + index
        env = make_environment(args.env, seed)
        try:
            if scenes is not None:
                observations, _, _ = env.reset_with_eval_config(scenes[index])
            else:
                env.num_cooperative = args.robots if args.robots is not None else 3
                env.num_non_cooperative = 0
                env.num_obs = args.obstacles if args.obstacles is not None else 8
                if args.env == 'maritime':
                    env.num_cores = args.cores if args.cores is not None else 6
                observations, _, _ = env.reset()
                if len(env.robots) != env.num_cooperative or len(env.obstacles) != env.num_obs:
                    raise RuntimeError('Random placement exhausted its attempts; reduce scene density or change seed.')
                if args.env == 'maritime' and len(env.cores) != env.num_cores:
                    raise RuntimeError('Could not place all requested vortex cores; change seed or core count.')
            result = run_episode(env, observations, policy, seed, args.max_steps)
            result['domain'] = args.env
            result['method'] = args.method
            result['scene_index'] = args.start_index + index if scenes is not None else None
            write_json(output / f'episode_{index:04d}.json', result)
            episodes.append(result)
            write_json(output / 'summary.json', summarize(episodes))
            print(f"Episode {index + 1}/{args.episodes}: steps={result['steps']}, statuses={[r['status'] for r in result['robots']]}", flush=True)
        finally:
            env.close()
    print(json.dumps(summarize(episodes), indent=2))
    print(f'Results: {output}')
if __name__ == '__main__':
    main()
