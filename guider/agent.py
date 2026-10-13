import json
from pathlib import Path
import numpy as np
import torch
from .models.iqn import IQN_Policy, IEQN_Policy
from .risks import terrestrial, maritime
from .risks.distance import adjust_cvar
METHODS = {'guider': ('pgm_iqn', 'final'), 'pgm_iqn': ('pgm_iqn', 'neutral'), 'iqn': ('iqn', 'neutral'), 'ada_iqn': ('iqn', 'distance'), 'art_iqn': ('iqn', 'metric'), 'guider_ada': ('pgm_iqn', 'distance'), 'guider_art': ('pgm_iqn', 'metric'), 'ieqn': ('ieqn', 'neutral'), 'iapf': (None, None), 'rvo': (None, None)}

def checkpoint_path(root, domain, method, seed):
    family, _ = METHODS[method]
    return Path(root) / domain / family / f'seed_{seed}' if family else None

def load_model(directory, family, device='cpu'):
    directory = Path(directory).expanduser().resolve()
    constructor = directory / 'constructor_params.json'
    parameters = directory / 'network_params.pth'
    for path in (constructor, parameters):
        if not path.is_file():
            raise FileNotFoundError(f'Missing checkpoint file: {path}')
    config = json.loads(constructor.read_text(encoding='utf-8'))
    config['device'] = str(device)
    model_type = IEQN_Policy if family == 'ieqn' else IQN_Policy
    model = model_type(**config)
    state_dict = torch.load(parameters, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()

class Policy:

    def __init__(self, method, domain, checkpoint=None, device='cpu'):
        self.method = method
        self.family, self.risk_mode = METHODS[method]
        self.device = torch.device(device)
        self.num_quantiles = 32
        self.final_rule = {'terrestrial': terrestrial, 'maritime': maritime}[domain].llm_adjust_cvar
        self.model = load_model(checkpoint, self.family, self.device) if self.family else None
        self.classical = {}

    def _quantiles(self, state, cvar):
        output = self.model(state, self.num_quantiles, cvar)
        return output['quantiles'] if isinstance(output, dict) else output[0]

    @torch.inference_mode()
    def act(self, observation, robot):
        raw_state = np.asarray(observation)
        if raw_state.shape != (39,) or not np.isfinite(raw_state).all():
            raise ValueError('Evaluation requires a finite 39-dimensional cooperative observation.')
        if self.family is None:
            key = (tuple(robot.a), tuple(robot.w), float(robot.max_speed))
            if key not in self.classical:
                if self.method == 'iapf':
                    from .baselines.iapf import APF_agent
                    self.classical[key] = APF_agent(robot.a, robot.w)
                else:
                    from .baselines.rvo import RVO_agent
                    self.classical[key] = RVO_agent(robot.a, robot.w, robot.max_speed)
            return (int(self.classical[key].act(raw_state)), None)
        state = torch.as_tensor(raw_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        cvar = 1.0
        if self.risk_mode == 'final':
            cvar = self.final_rule(None, raw_state)
        elif self.risk_mode == 'distance':
            cvar = adjust_cvar(None, raw_state)
        elif self.risk_mode == 'metric':
            quantiles = self._quantiles(state, 1.0)
            raw_metric = torch.std(quantiles, dim=2).max().item()
            cvar = float(np.clip(1.0 - raw_metric / 50.0, 0.0, 1.0))
        if not np.isfinite(cvar):
            raise ValueError(f'Non-finite CVaR returned by {self.risk_mode} rule.')
        quantiles = self._quantiles(state, cvar)
        if not torch.isfinite(quantiles).all():
            raise ValueError('Model returned non-finite action values.')
        return (int(quantiles.mean(dim=1).argmax(dim=1).item()), float(cvar))
