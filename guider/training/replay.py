import random

import numpy as np


class SumTree:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity >= 1")
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write = 0
        self.size = 0

    @property
    def total(self):
        return float(self.tree[0])

    def update(self, index, priority):
        delta = priority - self.tree[index]
        self.tree[index] = priority
        while index > 0:
            index = (index - 1) // 2
            self.tree[index] += delta

    def add(self, priority, transition):
        self.data[self.write] = transition
        self.update(self.write + self.capacity - 1, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self, value):
        index = 0
        while 2 * index + 1 < len(self.tree):
            left = 2 * index + 1
            if value < self.tree[left]:
                index = left
            else:
                value -= self.tree[left]
                index = left + 1
        return index, self.tree[index], self.data[index - self.capacity + 1]


class ReplayBuffer:
    def __init__(self, capacity, batch_size, alpha=0.6, beta=0.4, guidance_scale=2.0):
        if batch_size < 1 or capacity < batch_size or not 0 <= alpha <= 1 or not 0 <= beta <= 1:
            raise ValueError("Invalid replay parameters")
        self.tree = SumTree(capacity)
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.guidance_scale = guidance_scale
        self.max_priority = 1.0

    def __len__(self):
        return self.tree.size

    def add(self, state, action, reward, next_state, done, guided=False):
        transition = (np.asarray(state, dtype=np.float32).copy(), int(action), float(reward),
                      np.asarray(next_state, dtype=np.float32).copy(), bool(done), bool(guided))
        priority = self.max_priority * (self.guidance_scale if guided else 1.0)
        self.tree.add(priority ** self.alpha, transition)

    def sample(self):
        if len(self) < self.batch_size:
            raise ValueError("Insufficient replay samples")
        segment = self.tree.total / self.batch_size
        records = []
        for i in range(self.batch_size):
            value = (i + random.random()) * segment
            records.append(self.tree.get(min(value, np.nextafter(self.tree.total, 0.0))))
        indices, priorities, transitions = zip(*records)
        probabilities = np.asarray(priorities) / self.tree.total
        weights = (len(self) * probabilities) ** (-self.beta)
        weights /= weights.max()
        states, actions, rewards, next_states, dones, guided = zip(*transitions)
        return {
            "states": np.stack(states), "actions": np.asarray(actions),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "next_states": np.stack(next_states), "dones": np.asarray(dones),
            "guided": np.asarray(guided), "indices": np.asarray(indices),
            "weights": weights.astype(np.float32),
        }

    def update_priorities(self, indices, errors):
        if len(indices) != len(errors):
            raise ValueError("Priority length mismatch")
        updates = {}
        for index, error in zip(indices, errors):
            priority = abs(float(error)) + 1e-5
            if not np.isfinite(priority):
                raise ValueError("Non-finite replay priority")
            updates[int(index)] = max(updates.get(int(index), 0.0), priority)
        for index, priority in updates.items():
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(index, priority ** self.alpha)
