import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        b = {}
        for k in batch[0].keys():
            b[k] = np.array([d[k] for d in batch])
        return b

    def __len__(self):
        return len(self.buffer)
