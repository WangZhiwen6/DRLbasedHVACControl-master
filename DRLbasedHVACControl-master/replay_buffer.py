import collections
import random
from torch import FloatTensor
#存储动作和观察的结果
class ReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = collections.deque(maxlen=maxlen)

    def append(self, exp):
        self.buffer.append(exp)
#随机采样
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)
        obs_batch = FloatTensor(obs_batch)
        action_batch = FloatTensor(action_batch)
        reward_batch = FloatTensor(reward_batch)
        next_obs_batch = FloatTensor(next_obs_batch)
        done_batch = FloatTensor(done_batch)
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self):
        return len(self.buffer)