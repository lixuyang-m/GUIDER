import argparse
from datetime import datetime
import json
from pathlib import Path
import uuid

import torch

from guider.training.trainer import Trainer


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["terrestrial", "maritime"], default=None)
    parser.add_argument("--method", choices=["iqn", "pgm_iqn"], default=None)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--pgm-checkpoint", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads >= 1")
    torch.set_num_threads(args.threads)
    config_path = (args.config or ROOT / "configs/train" / f"{args.env or 'terrestrial'}.json").expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for field in ("env", "method", "seed", "total_timesteps"):
        value = getattr(args, field)
        if value is not None:
            config[field] = value
    if config.get("eval_config") is None:
        config["eval_config"] = str(ROOT / "configs/eval" / f"{config['env']}.json")
    else:
        eval_path = Path(config["eval_config"]).expanduser()
        config["eval_config"] = str((eval_path if eval_path.is_absolute() else config_path.parent / eval_path).resolve())
    pgm_checkpoint = args.pgm_checkpoint or config.get("pgm_checkpoint")
    if pgm_checkpoint is not None:
        pgm_checkpoint = Path(pgm_checkpoint).expanduser()
        if not args.pgm_checkpoint and not pgm_checkpoint.is_absolute():
            pgm_checkpoint = config_path.parent / pgm_checkpoint
        pgm_checkpoint = pgm_checkpoint.resolve()
        config["pgm_checkpoint"] = str(pgm_checkpoint)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or ROOT / "outputs" / f"{stamp}_{config['env']}_{config['method']}_{uuid.uuid4().hex[:6]}"
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    trainer = Trainer(config, output.expanduser().resolve(), args.device, pgm_checkpoint, checkpoint)
    trainer.run()


if __name__ == "__main__":
    main()
