import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('episode', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.loads(args.episode.read_text(encoding='utf-8'))
    scene = result['scene']
    fig, ax = plt.subplots(figsize=(7, 7))
    for (x, y), radius in zip(scene['env']['obstacles']['positions'], scene['env']['obstacles']['r']):
        ax.add_patch(plt.Circle((x, y), radius, color='gray', alpha=0.7))
    cores = scene['env'].get('cores', {}).get('positions', [])
    if cores:
        points = np.asarray(cores)
        ax.scatter(points[:, 0], points[:, 1], marker='x', color='steelblue', label='Vortex cores')
    for i, trajectory in enumerate(scene['robots']['trajectory']):
        points = np.asarray(trajectory)
        status = result['robots'][i]['status']
        line, = ax.plot(points[:, 0], points[:, 1], label=f'Robot {i}: {status}')
        ax.scatter(*scene['robots']['start'][i], color=line.get_color(), marker='o')
        ax.scatter(*scene['robots']['goal'][i], color=line.get_color(), marker='*')
    ax.set(xlim=(0, scene['env']['width']), ylim=(0, scene['env']['height']), xlabel='x (m)', ylabel='y (m)', title=f"{result['domain']} / {result['method']}")
    ax.set_aspect('equal')
    ax.legend(fontsize='small', loc='best')
    fig.tight_layout()
    output = args.output or args.episode.with_suffix('.png')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output.resolve())
if __name__ == '__main__':
    main()
