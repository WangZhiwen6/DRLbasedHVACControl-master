# DDPG
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

from model import Actor
from model import Critic


class DDPG:
        # 初始化函数，用于初始化DQN算法的参数
    def __init__(self, state_dim, action_dim, hidden_dim, actor_lr, critic_lr, gamma, sigma, tau, device,):

        pass

        self.actor = Actor(state_dim, action_dim, hidden_dim, action_bound=1.0).to(device)
        self.target_actor = Actor(state_dim, action_dim, hidden_dim, action_bound=1.0).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim).to(device)

        # 初始化价值网络的参数，两个价值网络的参数相同
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化策略网络的参数，两个策略网络的参数相同
        self.target_actor.load_state_dict(self.actor.state_dict())

        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 属性分配
        self.gamma = gamma  # 折扣因子
        self.sigma = sigma  # 高斯噪声的标准差，均值设为0
        self.tau = tau  # 目标网络的软更新参数
        self.action_dim = action_dim
        self.device = device
        self.count = 0

        # 动作选择
    def take_action(self, state):
        # 维度变换
        state = torch.tensor(state, dtype=torch.float).view(1, -1).to(self.device)
        # 策略网络计算出当前状态下的动作价值
        #action = self.actor(state).item()
        action_tensor = self.actor(state)
        if action_tensor.dim() == 1 and action_tensor.size(0) == 1:
            action = action_tensor.item()
        else:
            # 处理张量元素超过一个的情况
            action = action_tensor.argmax().item()
        # 给动作添加噪声，增加搜索
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

        # 软更新, 意思是每次learn的时候更新部分参数
    def soft_update(self, net, target_net):
        # 获取训练网络和目标网络需要更新的参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
        # 训练网络的参数更新要综合考虑目标网络和训练网络
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)

    def update(self, transition):
        # 从训练集中取出数据
        states = torch.tensor(transition['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 价值目标网络获取下一时刻的每个动作价值[b,n_states]-->[b,n_actors]
        next_q_values = self.target_actor(next_states)
        # 策略目标网络获取下一时刻状态选出的动作价值 [b,n_states+n_actions]-->[b,1]
        next_q_values = self.target_critic(next_states, next_q_values)
        # 当前时刻的动作价值的目标值 [b,1]
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        # 当前时刻动作价值的预测值 [b,n_states+n_actions]-->[b,1]
        q_values = self.critic(states, actions)

        # 预测值和目标值之间的均方差损失
        critic_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # 价值网络梯度
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 当前状态的每个动作的价值
        actor_q_values = self.actor(states)
        # 当前状态选出的动作价值
        score = self.critic(states, actor_q_values)
        # 计算损失
        actor_loss = -torch.mean(score)
        # 策略网络梯度
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新策略网络的参数
        self.soft_update(self.actor, self.target_actor)
        # 软更新价值网络的参数
        self.soft_update(self.critic, self.target_critic)
