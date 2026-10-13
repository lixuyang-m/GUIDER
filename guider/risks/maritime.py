def llm_adjust_cvar(self, state):
    import numpy as np

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    goal_dist, goal_angle = (state[0], state[1])
    self_speed_x, self_speed_y = (state[2], state[3])
    self_speed = np.sqrt(self_speed_x ** 2 + self_speed_y ** 2)
    self_dir = np.arctan2(self_speed_y, self_speed_x) if self_speed > 0 else 0
    static_risks = []
    min_static_dist = float('inf')
    for i in range(5):
        idx = 4 + i * 3
        rel_x, rel_y, radius = state[idx:idx + 3]
        dist = np.sqrt(rel_x ** 2 + rel_y ** 2)
        if dist < 1e-06:
            continue
        adjusted_dist = dist - radius - 0.8
        if adjusted_dist <= 0:
            return 0.0
        obstacle_angle = np.arctan2(rel_y, rel_x)
        relative_angle = min(abs(self_dir - obstacle_angle), 2 * np.pi - abs(self_dir - obstacle_angle))
        angle_risk = 0.5 * (1 - np.cos(relative_angle))
        static_risk = np.exp(-adjusted_dist) * (1 + angle_risk)
        static_risks.append(static_risk)
        if adjusted_dist < min_static_dist:
            min_static_dist = adjusted_dist
    dynamic_risks = []
    time_horizon = 1.5
    for i in range(5):
        idx = 19 + i * 4
        rel_x, rel_y, rel_vx, rel_vy = state[idx:idx + 4]
        dist = np.sqrt(rel_x ** 2 + rel_y ** 2)
        if dist < 1e-06:
            continue
        adjusted_dist = dist - 1.6
        if adjusted_dist <= 0:
            return 0.0
        rel_speed = np.sqrt(rel_vx ** 2 + rel_vy ** 2)
        if rel_speed < 0.5:
            dynamic_risks.append(1.5 / (adjusted_dist + 0.1))
            continue
        relative_dir = np.arctan2(rel_y, rel_x)
        relative_angle = min(abs(self_dir - relative_dir), 2 * np.pi - abs(self_dir - relative_dir))
        dot_product = rel_x * (self_speed_x - rel_vx) + rel_y * (self_speed_y - rel_vy)
        if dot_product > 0:
            approach_speed = dot_product / (dist + 1e-06)
            ttc = adjusted_dist / (approach_speed + 1e-06)
            ttc = max(ttc, 0.1)
            angle_factor = 0.5 + 0.5 * (1 - relative_angle / np.pi)
            future_x = rel_x + rel_vx * time_horizon
            future_y = rel_y + rel_vy * time_horizon
            future_dist = np.sqrt(future_x ** 2 + future_y ** 2) - 1.6
            ttc_risk = angle_factor / ttc
            future_risk = np.exp(-future_dist) if future_dist > 0 else 1.0
            dynamic_risk = 0.6 * ttc_risk + 0.4 * future_risk
            dynamic_risks.append(min(1.5, dynamic_risk))
    if goal_dist < 1e-06:
        goal_risk = 0.0
    else:
        dist_risk = np.arctan(goal_dist / 5.0) / (np.pi / 2)
        angle_risk = 0.3 * (abs(goal_angle) / np.pi)
        close_risk = 0.0
        if goal_dist < 2.0:
            close_risk = 0.2 * (2.0 - goal_dist) / 2.0
        goal_risk = 0.6 * dist_risk + 0.3 * angle_risk + 0.1 * close_risk
    max_static = max(static_risks) if static_risks else 0.0
    max_dynamic = max(dynamic_risks) if dynamic_risks else 0.0
    if max_static > 2.0 or max_dynamic > 2.0:
        return 0.0
    if goal_dist < 1.0 and max_static + max_dynamic < 0.2:
        return 1.0
    total_risk = 0.35 * max_static + 0.5 * max_dynamic + 0.15 * goal_risk
    cvar = 1.0 - sigmoid(5.0 * (total_risk - 0.3))
    cvar = np.clip(cvar, 0.0, 1.0)
    return float(cvar)
