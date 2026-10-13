import numpy as np
import copy

class APF_agent:

    def __init__(self, a, w):
        self.k_att = 50.0
        self.k_rep = 500.0
        self.k_v = 1.0
        self.m = 500
        self.d0 = 15.0
        self.n = 2
        self.min_vel = 1.0
        self.a = a
        self.w = w
        self.GAMMA = 0.99

    def position_force(self, position, radius, goal):
        d_obs = np.linalg.norm(position) - radius - 0.8
        d_goal = np.linalg.norm(goal)
        mag_1 = self.k_rep * (1 / d_obs - 1 / self.d0) * d_goal ** self.n / d_obs ** 2
        dir_1 = -1.0 * position / np.linalg.norm(position)
        F_rep_1 = mag_1 * dir_1
        mag_2 = self.n / 2 * self.k_rep * (1 / d_obs - 1 / self.d0) ** 2 * d_goal ** (self.n - 1)
        dir_2 = -1.0 * goal / d_goal
        F_rep_2 = mag_2 * dir_2
        return F_rep_1 + F_rep_2

    def velocity_force(self, v_ao, position, radius):
        d_obs = np.linalg.norm(position) - radius - 0.8
        mag = -self.k_v * v_ao / d_obs
        dir = position / np.linalg.norm(position)
        F_rep = mag * dir
        return F_rep

    def act(self, observation):
        assert len(observation) == 39, 'The state size does not equal 39'
        obs_array = np.array(observation)
        ego = obs_array[:4]
        static = obs_array[4:19]
        dynamic = obs_array[19:]
        F_att = self.k_att * ego[:2]
        F_rep = np.zeros(2)
        goal = ego[:2]
        velocity = ego[2:]
        for i in range(0, len(static), 3):
            if np.abs(static[i]) < 0.001 and np.abs(static[i + 1]) < 0.001:
                continue
            F_rep += self.position_force(static[i:i + 2], static[i + 2], goal)
        for i in range(0, len(dynamic), 4):
            if np.abs(dynamic[i]) < 0.001 and np.abs(dynamic[i + 1]) < 0.001:
                continue
            e_ao = dynamic[i:i + 2] / np.linalg.norm(dynamic[i:i + 2])
            v_ao = np.dot(velocity - dynamic[i + 2:i + 4], e_ao)
            if v_ao >= 0.0:
                F_rep += self.position_force(dynamic[i:i + 2], 0.8, goal)
                F_rep += self.velocity_force(v_ao, dynamic[i:i + 2], 0.8)
        F_total = F_att + F_rep
        V_angle = 0.0
        if np.linalg.norm(velocity) > 0.001:
            V_angle = np.arctan2(velocity[1], velocity[0])
        F_angle = np.arctan2(F_total[1], F_total[0])
        diff_angle = F_angle - V_angle
        while diff_angle < -np.pi:
            diff_angle += 2 * np.pi
        while diff_angle >= np.pi:
            diff_angle -= 2 * np.pi
        w_idx = np.argmin(np.abs(self.w - diff_angle))
        a_total = F_total / self.m
        V_dir = np.array([1.0, 0.0])
        if np.linalg.norm(velocity) > 0.001:
            V_dir = velocity / np.linalg.norm(velocity)
        a_proj = np.dot(a_total, V_dir)
        a = copy.deepcopy(self.a)
        if np.linalg.norm(velocity) < self.min_vel:
            a[a <= 0.0] = -np.inf
        a_diff = a - a_proj
        a_idx = np.argmin(np.abs(a_diff))
        return a_idx * len(self.w) + w_idx
