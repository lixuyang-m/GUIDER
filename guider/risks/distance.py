import numpy as np

def adjust_cvar(self, state):
    assert len(state) == 39, 'The state size does not equal 39'
    static = state[4:19]
    dynamic = state[19:]
    closest_d = np.inf
    for i in range(0, len(static), 3):
        if np.abs(static[i]) < 0.001 and np.abs(static[i + 1]) < 0.001:
            continue
        dist = np.linalg.norm(static[i:i + 2]) - static[i + 2] - 0.8
        closest_d = min(closest_d, dist)
    for i in range(0, len(dynamic), 4):
        if np.abs(dynamic[i]) < 0.001 and np.abs(dynamic[i + 1]) < 0.001:
            continue
        dist = np.linalg.norm(dynamic[i:i + 2]) - 1.6
        closest_d = min(closest_d, dist)
    cvar = 1.0
    if closest_d < 10.0:
        cvar = closest_d / 10.0
    return cvar
