```bash
python -m pip install -r requirements.txt
```

```bash
python evaluate.py --env terrestrial --method guider --model-seed 1 --episodes 1
python evaluate.py --env maritime --method guider --model-seed 1 --episodes 1
python evaluate.py --env maritime --method guider --checkpoint /path/to/checkpoint --episodes 30
python evaluate.py --env maritime --method ieqn --model-seed 1 --episodes 1
python evaluate.py --env maritime --random-scenes --robots 3 --obstacles 8 --cores 6
```

```bash
python train.py --env terrestrial --method iqn --device cuda:0 --seed 1
python train.py --env maritime --method iqn --device cuda:0 --seed 1
python train.py --env terrestrial --method pgm_iqn --pgm-checkpoint /path/to/best_model_9.pth --device cuda:0
python train.py --env maritime --method pgm_iqn --pgm-checkpoint /path/to/best_model_9.pth --device cuda:0
python train.py --config configs/train/terrestrial.json --method iqn --output outputs/terrestrial_iqn
```

```bash
python train_pgm.py --dataset data/pgm.npz --device cuda:0 --output outputs/pgm
python train.py --env terrestrial --method pgm_iqn --pgm-checkpoint outputs/pgm/best_model.pth --device cuda:0
```

```bash
python plot_trajectories.py results/<run>/episode_0000.json
```

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
