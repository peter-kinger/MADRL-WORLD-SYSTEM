"""绘图补充显示的部分

- 经验操作
- 绘图轨迹操作

"""
import random
import torch
from scipy.integrate import odeint
# from . import ays_model as ays
# 导入 ays_model 模块
from envs.AYS import ays_model  as ays


# 放置了 RL 中改进的辅助算法
class ReplayBuffer:
    """To store experience for uncorrelated learning"""

    # 内容的写法非常类似于叶强的 DQN 算法
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
#
# class PER_IS_ReplayBuffer:
#     """
#         Adapted from https://github.com/labmlai/annotated_deep_learning_paper_implementations
#         """
#
#     def __init__(self, capacity, alpha, state_dim=3):
#         self.capacity = capacity
#         self.alpha = alpha
#         self.priority_sum = [0 for _ in range(2 * self.capacity)]
#         self.priority_min = [float('inf') for _ in range(2 * self.capacity)]
#         self.max_priority = 1.
#         self.data = {
#             'obs': np.zeros(shape=(capacity, state_dim), dtype=np.float64),
#             'action': np.zeros(shape=capacity, dtype=np.int32),
#             'reward': np.zeros(shape=capacity, dtype=np.float32),
#             'next_obs': np.zeros(shape=(capacity, state_dim), dtype=np.float64),
#             'done': np.zeros(shape=capacity, dtype=np.bool)
#         }
#         self.next_idx = 0
#         self.size = 0
#
#     def push(self, obs, action, reward, next_obs, done):
#         idx = self.next_idx
#         self.data['obs'][idx] = obs
#         self.data['action'][idx] = action
#         self.data['reward'][idx] = reward
#         self.data['next_obs'][idx] = next_obs
#         self.data['done'][idx] = done
#
#         self.next_idx = (idx + 1) % self.capacity
#         self.size = min(self.capacity, self.size + 1)
#
#         priority_alpha = self.max_priority ** self.alpha
#         self._set_priority_min(idx, priority_alpha)
#         self._set_priority_sum(idx, priority_alpha)
#
#     def _set_priority_min(self, idx, priority_alpha):
#         idx += self.capacity
#         self.priority_min[idx] = priority_alpha
#         while idx >= 2:
#             idx //= 2
#             self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])
#
#     def _set_priority_sum(self, idx, priority_alpha):
#         idx += self.capacity
#         self.priority_sum[idx] = priority_alpha
#         while idx >= 2:
#             idx //= 2
#             self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]
#
#     def _sum(self):
#         return self.priority_sum[1]
#
#     def _min(self):
#         return self.priority_min[1]
#
#     def find_prefix_sum_idx(self, prefix_sum):
#         idx = 1
#         while idx < self.capacity:
#             if self.priority_sum[idx * 2] > prefix_sum:
#                 idx = 2 * idx
#             else:
#                 prefix_sum -= self.priority_sum[idx * 2]
#                 idx = 2 * idx + 1
#
#         return idx - self.capacity
#
#     def sample(self, batch_size, beta):
#
#         samples = {
#             'weights': np.zeros(shape=batch_size, dtype=np.float32),
#             'indexes': np.zeros(shape=batch_size, dtype=np.int32),
#         }
#
#         for i in range(batch_size):
#             p = random.random() * self._sum()
#             idx = self.find_prefix_sum_idx(p)
#             samples['indexes'][i] = idx
#
#         prob_min = self._min() / self._sum()
#         max_weight = (prob_min * self.size) ** (-beta)
#
#         for i in range(batch_size):
#             idx = samples['indexes'][i]
#             prob = self.priority_sum[idx + self.capacity] / self._sum()
#             weight = (prob * self.size) ** (-beta)
#             samples['weights'][i] = weight / max_weight
#
#         for k, v in self.data.items():
#             samples[k] = v[samples['indexes']]
#
#         return samples
#
#     def update_priorities(self, indexes, priorities):
#
#         for idx, priority in zip(indexes, priorities):
#             self.max_priority = max(self.max_priority, priority)
#             priority_alpha = priority ** self.alpha
#             self._set_priority_min(idx, priority_alpha)
#             self._set_priority_sum(idx, priority_alpha)
#
#     def is_full(self):
#         return self.capacity == self.size
#
#     def __len__(self):
#         return self.size


import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt

# 单独补充其中的绘制部分
def plot(data_dict):
    """该函数适用于 notebook 中画图
    # 也可以使用 wandb 进行相关操作绘制，动态显示


    :param data_dict:
    :return:
    """
    rewards = data_dict['moving_avg_rewards']
    std = data_dict['moving_std_rewards']
    frame_idx = data_dict['frame_idx']
    clear_output(True)
    plt.figure(figsize=(20, 5))

    # 建立其中的基础画布
    # 因为其中的有两个图，所以不能在一起绘制，否则 ax3d 会冲突
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    reward = np.array(rewards)
    stds = np.array(std)
    plt.fill_between(np.arange(len(reward)), reward - 0.25 * stds, reward + 0.25 * stds, color='b', alpha=0.1)
    plt.fill_between(np.arange(len(reward)), reward - 0.5 * stds, reward + 0.5 * stds, color='b', alpha=0.1)
    plt.show()

import wandb
import numpy as np

def plot_wandb(data_dict):
    rewards = data_dict['moving_avg_rewards']
    std = data_dict['moving_std_rewards']
    frame_idx = data_dict['frame_idx']
    reward = np.array(rewards)
    stds = np.array(std)

    # 初始化WandB
    wandb.init(project='your_project_name')

    # 记录rewards和stds作为指标
    wandb.log({'rewards': reward, 'stds': stds}, step=frame_idx)

    # 绘制图像
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.fill_between(np.arange(len(reward)), reward - 0.25 * stds, reward + 0.25 * stds, color='b', alpha=0.1)
    plt.fill_between(np.arange(len(reward)), reward - 0.5 * stds, reward + 0.5 * stds, color='b', alpha=0.1)

    # 保存图像到WandB
    wandb.log({"plot": wandb.Image(plt)})

    # 关闭WandB运行
    wandb.finish()

# TODO 实施管理措施部分


def plot_hairy_lines(num, ax3d):
    """头发丝绘制的部分

    Args:
        num (_type_): _description_
        ax3d (_type_): _description_
    """
    # 即 ax3d =ax
    colortop = "lime"
    colorbottom = "black"

    ays_0 = np.random.rand(num, 3)
    time = np.linspace(0, 100, 1000)

    # 自己添加的部分
    parameter_list = [(0.03, 147, 47000000000.0, 2.0, 4000000000000.0, 50, 50, 8.57e-05)]

    for i in range(num):
        x0 = ays_0[i]
        traj = odeint(ays.AYS_rescaled_rhs, x0, time, args=parameter_list[0]) # TODO 完善管理措施
        
        # 为每条曲线生成不同的颜色
        color = plt.cm.viridis(i / num)  # 使用viridis色图，根据i/num生成不同颜色
        
        # 绘制曲线
        # ax3d.plot3D(xs=traj[:, 0], ys=traj[:, 1], zs=traj[:, 2], color='purple', alpha=.08)
        ax3d.plot3D(xs=traj[:,0], ys=traj[:,1], zs=traj[:,2],
                    color=colorbottom if traj[-1,2]<0.5 else colortop, alpha=.08)
    
    # 统一起始点的颜色为紫色，停止点的颜色为红色，并调整标记点的大小
    # 加入三个固定点显示
    ax3d.scatter(*zip([0.5,0.5,0.5]), lw=1, color='pink')
    # ax3d.scatter(*zip([0.60,0.38,0]), lw=4, color='black')
    # ax3d.scatter(*zip([0,1,1]), lw=4, color='lime')