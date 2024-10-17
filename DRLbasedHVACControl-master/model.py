import torch
import torch.nn as nn
from torchsummary import summary


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, action_bound=1.0):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        #定义激活函数
        self.ReLU = nn.ReLU()
        #定义网络结构
        self.f1 = nn.Linear(state_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
            x = torch.relu(self.f1(x))

            x = torch.relu(self.f2(x))

            return torch.tanh(self.f3(x)) * self.action_bound



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_dim=64):
        super(Critic, self).__init__()
        #定义激活函数
        self.ReLU = nn.ReLU()
        #定义网络结构
        self.f1 = nn.Linear(state_dim+action_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, 1)


    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = self.f1(cat)
        x = torch.relu(x)
        x = torch.relu(self.f2(x))

        return self.f3(x)




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device to use has been set to: "{device}"')
    state_dim = 14
    action_dim = 64
    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    print(summary(actor, (1, 14)))