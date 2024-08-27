import numpy as np
import torch
import torch.nn as nn


try:
    from . import networks as nets
except:
    import networks as nets


# 下面是特别的搬运写法操作
# 这是唯一用法
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def numpy_to_cuda(numpy_array):
    return torch.from_numpy(numpy_array).float().to(DEVICE)

"""
这里是对过去的 ays 中的DQN 部分的重写，

"""
class DQN:
    """DQN implementation with epsilon greedy actions selection"""

    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.002357, tau=0.0877, rho=0.7052, epsilon=1., polyak=False,
                 decay=0.5, step_decay=50000):

        # create simple networks that output Q-values, both target and policy are identical
        self.target_net = self.create_net(state_dim, action_dim, duelling=False).to(DEVICE)
        self.policy_net = self.create_net(state_dim, action_dim, duelling=False).to(DEVICE)

        # We use the Adam optimizer
        self.lr = lr
        self.decay = decay
        self.step_decay = step_decay # 这是 DL 中的衰减
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=self.decay, step_size=step_decay)
        self.action_size = action_dim
        self.gamma = gamma

        # Squared loss, as it is easier to work with than the Huber loss
        self.loss = nn.MSELoss()

        # For copying networks
        self.tau = tau
        self.counter = 0
        self.polyak = polyak

        # We decay epsilon according to the following formula
        self.t = 1
        self.rho = rho
        self.epsilon = lambda t: 0.01 + epsilon / (t ** self.rho)

    @staticmethod
    def create_net(s_dim, a_dim, duelling):
        """We create action-out networks that can be duelling or not,
         Duelling is more stable to optimisation"""
        if duelling:
            net = nets.DuellingNet(s_dim, a_dim)
        else:
            net = nets.Net(s_dim, a_dim)
        return net

    # 这个的相关用法就是类似于神经网络中的梯度的相关问题
    # 等同于 model.eval()
    # with torch.no_grad():
    #     pass
    @torch.no_grad()
    def get_action(self, state: np.ndarray, testing=False):
        """We select actions according to epsilon-greedy policy"""
        self.t += 1
        # 本来就是默认的，这是贪婪策略的运行部分
        # 如果超过一定的大小或者测试开启那么就停止贪婪策略
        # 那么测试部分就会受到影响
        #
        if np.random.uniform() > self.epsilon(self.t) or testing:
            q_values = self.policy_net(torch.Tensor(state).to(DEVICE)).cpu().numpy()
            return np.argmax(q_values)
        else:
            return np.random.choice(self.action_size)
        
    

    # 关键部分
    def update(self, batch_sample, weights=None):
        """To update our networks"""
        # Unpack batch: 5-tuple
        # 五元组的相关更新
        state, action, reward, next_state, done = batch_sample

        # convert to torch.cuda
        states = numpy_to_cuda(state)
        actions = numpy_to_cuda(action).type(torch.int64).unsqueeze(1)
        next_states = numpy_to_cuda(next_state)
        rewards = numpy_to_cuda(reward)

        # get the Q-values of the actions at time t
        state_qs = self.policy_net(states).gather(1, actions).squeeze(1)

        # get the max Q-values at t+1 from the target network
        next_state_values = self.next_state_value_estimation(next_states, done)

        # target: y_t = r_t + gamma * max[Q(s,a)]
        targets = (rewards + self.gamma * next_state_values.squeeze(1))

        # if we have weights from importance sampling
        if weights is not None:
            weights = numpy_to_cuda(weights)
            # 利用这个来进行相关计算操作
            # 里面有点类似于 TD 相关计算的内容
            loss = ((targets - state_qs).pow(2) * weights).mean()
        # otherwise we use the standard MSE loss
        else:
            loss = self.loss(state_qs, targets)

        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # scheduler step
        self.scheduler.step()

        # to copy the policy parameters to the target network
        self.copy_nets()
        # we return the loss for logging and the TDs for Prioritised Experience Replay
        return loss, (state_qs - targets).detach()

    @torch.no_grad()
    def next_state_value_estimation(self, next_states, done):
        """Function to define the value of the next state, makes inheritance cleaner"""
        next_state_values = self.target_net(next_states).max(1)[0].detach()
        # the value of a state after a terminal state is 0
        next_state_values[done] = 0
        return next_state_values.unsqueeze(1)

    def copy_nets(self):
        """Copies the parameters from the policy network to the target network, either all at once or incrementally."""
        # Dueling才考虑使用
        self.counter += 1
        if not self.polyak and self.counter >= 1 / self.tau:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.counter = 0
        else:
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    # 关键，写入对应的字符对应
    # 可以读取相关网络对应的当前名字
    def __str__(self):
        return "DQN"
