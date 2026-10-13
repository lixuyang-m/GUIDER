def llm_adjust_cvar(self, state):
    import numpy as np
    import math
    PREDICTION_TIME = 0.5
    SAFE_DISTANCE = 2.0
    COLLISION_RADIUS = 0.8

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def normalize_angle(angle):
        return abs(angle) / math.pi
    goal_dist = state[0]
    goal_angle = state[1]
    static_risks = []
    for i in range(5):
        idx = 4 + i * 3
        rel_x, rel_y, radius = (state[idx], state[idx + 1], state[idx + 2])
        if radius == 0:
            continue
        dist = math.sqrt(rel_x ** 2 + rel_y ** 2) - radius - COLLISION_RADIUS
        if dist <= 0:
            risk = 1.0
        elif dist < SAFE_DISTANCE:
            risk = 1 - dist / SAFE_DISTANCE
        else:
            risk = 0.0
        static_risks.append(risk)
    dynamic_risks = []
    for i in range(5):
        idx = 19 + i * 4
        rel_x, rel_y, vel_x, vel_y = (state[idx], state[idx + 1], state[idx + 2], state[idx + 3])
        pred_x = rel_x + vel_x * PREDICTION_TIME
        pred_y = rel_y + vel_y * PREDICTION_TIME
        pred_dist = math.sqrt(pred_x ** 2 + pred_y ** 2) - COLLISION_RADIUS * 2
        if pred_dist <= 0:
            risk = 1.0
        else:
            risk = math.exp(-pred_dist / (SAFE_DISTANCE * 0.5))
        dynamic_risks.append(risk)
    max_static_risk = max(static_risks) if static_risks else 0
    max_dynamic_risk = max(dynamic_risks) if dynamic_risks else 0
    overall_risk = max(max_static_risk, max_dynamic_risk)
    norm_dist = min(goal_dist / 10.0, 1.0)
    norm_angle = normalize_angle(goal_angle)
    goal_weight = 0.3 + 0.7 * sigmoid(2 - 5 * norm_dist - 2 * norm_angle)
    adjusted_risk = overall_risk * (1 - goal_weight * 0.8)
    cvar = 1.0 - min(max(adjusted_risk, 0.0), 1.0)
    cvar = sigmoid(4 * (cvar - 0.5))
    return float(cvar)
