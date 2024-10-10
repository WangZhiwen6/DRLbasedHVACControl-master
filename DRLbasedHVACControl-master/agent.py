#实现了DQN算法
import torch
import torch.nn.functional as F
import numpy as np
from model import DNN_5


class DQN:
        # 初始化函数，用于初始化DQN算法的参数
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, device, update_interval):
        # state_dim：状态维度
        # action_dim：动作维度
        # lr：学习率
        # gamma：折扣因子
        # epsilon：探索率
        # device：设备（CPU或GPU）
        # update_interval：更新间隔
        pass

        self.Q_net = DNN_5(state_dim, action_dim).to(device)
        self.target_Q_net = DNN_5(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=lr)

        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device

        self.update_interval = update_interval
        self.count = 0

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action_index = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action_index = self.Q_net(state).argmax().item()
        return action_index

    def take_action_for_validation(self, state):
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action_index = self.target_Q_net(state).argmax().item()
        return action_index

    def update(self, transition):
        # current_state = torch.tensor(current_state, dtype=torch.float).to(self.device)

        state = torch.tensor(transition['states'], dtype=torch.float).to(self.device)
        action = torch.tensor(transition['actions']).view(-1, 1).to(self.device)
        reward = torch.tensor(transition['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor(transition['next_states'], dtype=torch.float).to(self.device)
        done = torch.tensor(transition['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        #计算Q值
        Q_value = self.Q_net(state).gather(1, action)
        #计算目标Q值
        next_Q_value_max = self.target_Q_net(next_state).max(1)[0].view(-1, 1)
        Q_target = reward + self.gamma * next_Q_value_max * (1 - done).view(-1, 1)
        #计算损失
        loss = torch.mean(F.mse_loss(Q_value, Q_target))
        #更新Q网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.update_interval == 0:
            # print(f'Q网络已更新，count={self.count}')
            # 将Q_net的状态字典加载到target_Q_net中
            self.target_Q_net.load_state_dict(
                self.Q_net.state_dict()
            )
        self.count += 1
        # if loss < 100:
        #     torch.save(self.target_Q_net.state_dict(), f'./BESTMODEL/episode_{self.count}.pth')
        return loss.item()
