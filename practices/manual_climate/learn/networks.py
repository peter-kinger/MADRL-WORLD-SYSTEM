"""作为智能体之间通用的网络结构
特殊的网络权重部分
"""


import torch.nn as nn
import torch.nn.functional as F 


class PolicyNet(nn.Module):
    """
    考虑其中对应相关的维度
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNet, self).__init__()

        # 注意对应维度网络的相关匹配
        self.layer = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   )
        self.a = nn.Linear(hidden_dim, action_dim)  

    def forward(self, x):   
        l = self.layer(x)
        out = self.a(l)
        return out
    
class Net(nn.Module):
    """输出王树森对应的那个图片，Q——values 的直接估计
    就是在传统 q_tables 加入了神经网络拟合并利用梯度进行更新的操作，可以看下面的 optimizer
    """
    # 对应输入的事 state 这些，输出是 q_values
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Net, self).__init__()

        self.layer = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    )

        self.q = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        l = self.layer(obs)
        q_values = self.q(l)

        return q_values


