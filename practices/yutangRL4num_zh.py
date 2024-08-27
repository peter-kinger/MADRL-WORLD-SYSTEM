"""
# -*- coding: utf-8 -*-
@Author: Shiwei Yuan
@E-mail: yuansw@itpcas.ac.cn
@Filename: yutangRL.py
@CreatTime：2023/11/28, 9:18
@Used for:
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import MultipleLocator

# 开始的文件参数设置，统一内容
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14
plt.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['font.weight'] = 'normal'
mpl.rcParams['axes.labelweight'] = 'bold'


# 需要运行比较久的时间


class FishPond:
    """鱼塘类的定义

    开始定义一个鱼塘中实行的一个环境，包括鱼的数量，捕捞量，增长率等
    统一都写在一个类中
    """
    def __init__(self, initial_fish_population=1000, max_fish_population=3000, growth_rate=1.01):
        self.fish_population = initial_fish_population
        self.max_fish_population = max_fish_population
        self.growth_rate = growth_rate

    def reset(self):
        # 重置鱼塘的状态
        self.fish_population = self.fish_population = np.random.randint(5, 16) * 100
        return self.fish_population # 初始状态为当前鱼群数量和捕捉量的元组

    def step(self, action):
        #option

        # 计算捕捉的鱼量
        #fish_catch = np.random.randint(1, self.max_fish_catch + 1)*8
        # 若传入数组
        fish_amount = sum(action)
        fish_catch = min(fish_amount*3, self.fish_population) # 设置其中可以捕捉的数量，这里的 3 是通过贪婪系数进行相关定义的

        # 更新鱼群数量
        # option 1 指数增长
        #self.fish_population = int(self.fish_population * self.growth_rate) - fish_catch
        #option 2 logistic增长

        # 这是动力学演进的关键部分，鱼的数量随着时间的演进而变化
        self.fish_population = int(self.fish_population + self.fish_population * (self.growth_rate-1) * (1 - self.fish_population / self.max_fish_population)) - fish_catch

        # 防止鱼的数量为负数
        self.fish_population = max(0, self.fish_population) # 注意其中演进的顺序是按照时间顺序的，所以这个 max 是写在后面的

        # 计算奖励
        reward = fish_catch/20 # 设置每次捕鱼的奖励，这里的 20 是通过贪婪系数进行相关定义的，因为是越捕越少所以奖励也会变少

        if self.fish_population <= 100:
            reward -= 200  # 降低到某个阈值后给与惩罚

        return self.fish_population, reward # 返回环境的两个数组内容

class QLearningAgent:
    """
    
    开始定义其中的学习部分，这里的学习部分是通过Q-learning的方法进行的
    """
    def __init__(self, environment, fishman= 4, max_fish_catch=10, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

        # 注意其中可选定义的动作空间和状态空间的大小
        self.action_space_size = max_fish_catch + 1
        self.state_space_size = environment.max_fish_population // 100 + 1

        self.fishman_num = fishman
        self.q_table = np.zeros((self.fishman_num,self.state_space_size, self.action_space_size)) # 开始的 q_table 的定义其实就是一个 numpy 的量

    def choose_action(self, state):
        action = []
        for i in range(self.fishman_num):
            if np.random.rand() < self.exploration_prob: # 这里的 exploration_prob 是通过贪婪系数进行相关定义的s
                action.append(np.random.randint(0, self.action_space_size))
            else:
                action.append(np.argmax(self.q_table[i, state, :]))
                # print("q_table 被调用了")

        return action  # 选择 Q 值最大的动作，这是对其进行相关的返回

    def update_q_table(self, state, action, reward, next_state):
        for i in range(self.fishman_num):
            best_next_action = np.argmax(self.q_table[i, next_state, :])
            # 贝尔曼方程
            self.q_table[i, state, action[i]] += self.learning_rate * (
                    reward + self.discount_factor * self.q_table[i, next_state, best_next_action] - self.q_table[i, state, action[i]])

    def train(self, num_episodes, env):
        rewards = []
        fish_pop = [] # 鱼的数量
        
        
        for episode in range(num_episodes):
            fish_population = env.reset()
            state = fish_population // 100
            total_reward = 0
            round = 0

            # 一个回合的循环
            while True:
                action = self.choose_action(state)
                fish_population, reward = env.step(action)
                next_state = fish_population // 100
                total_reward += reward

                self.update_q_table(state, action, reward, next_state) # discounted reward 的关键，对未来进行相关的计算

                state = next_state
                round += 1
                if env.fish_population == 0:
                    break

                if env.fish_population >= 3500:
                    break

                if round > 100:
                    break

            rewards.append(total_reward)
            fish_pop.append(env.fish_population)

        return rewards, fish_pop

# 创建FishPond和QLearningAgent
fish_pond = FishPond(initial_fish_population=1000, max_fish_population=3000, growth_rate=1.05)
agent = QLearningAgent(fish_pond, fishman=4, max_fish_catch=8)  # 动作空间为0到10，捕鱼数量为动作空间*10



# 训练智能体
num_episodes = 5000
# 开始进行配置训练的内容
training_rewards, training_fish = agent.train(num_episodes, fish_pond) # 开始运行进行训练智能体

# # 测试智能体
# test_rewards = []
# for _ in range(10):
#     state = fish_pond.reset()
#     total_reward = 0
#     while True:
#         action = agent.choose_action(state)
#         next_state, reward = fish_pond.step(action)
#         total_reward += reward
#         state = next_state
#         if fish_pond.fish_population == 0:
#             break
#         if fish_pond.fish_population >= 1000:
#             break
#
#     test_rewards.append(total_reward)

fish_population = fish_pond.reset() # 来自开始的智能体训练结果


state = fish_population // 100
total_reward = 0
total_action = []
total_population = []
round = 0

"""
这些区别表明第一段代码可能是一个简化的测试或实验性质的代码片段，
而第二段代码则更像是一个完整的强化学习训练过程的实现。
"""

while True:

    """
    师兄上面的 train 部分没有使用，但是单独定义了一个 while True 循环的部分进行操作
    """
    # 和默认的程序一样是一个 部分
    round += 1
    # 其实主要的写法就是把环境和 agent 单独写成了两种不同的类，然后在这里进行了调用（后面考虑可以利用封装好的进行嵌套处理）
    action = agent.choose_action(state) # 其实就是把一个 agent 定义为多个 agent 了
    fish_population, reward = fish_pond.step(action) # 这是自然变化增进的过程
    next_state = fish_population // 100

    total_reward += reward

    state = next_state
    total_action.append(action) # 每次都会添加其中的 action
    total_population.append(fish_population)

    # 这是让其不断进行循环的，没有停止的条件
    if fish_population == 0:
        break
    if fish_population >= 3500:
        break
    if round > 100: # 这是定义训练的论述，也就是训练的次数
        break
    
    
action_list = np.array(total_action)
action_list = action_list.T


# 可视化训练和测试结果
plt.figure(figsize=(16, 9))
x_major_locator=MultipleLocator(10)
y_major_locator=MultipleLocator(1)

plt.subplot(2, 3, 1)
plt.ylim(-0.5,8.5)#y轴范围设置
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
ax.xaxis.set_major_locator(x_major_locator)

plt.scatter(range(len(action_list[0])), action_list[0], label='Action')
# plt.axhline(np.mean(test_rewards), color='r', linestyle='dashed', linewidth=2, label='Test Mean')
plt.xlabel('Time')
# plt.ylabel('Action of Fishman 1')
plt.title('Actions of fishman A')
# plt.legend()

plt.subplot(2, 3, 2)
plt.ylim(-0.5,8.5)#y轴范围设置
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
ax.xaxis.set_major_locator(x_major_locator)
plt.scatter(range(len(action_list[1])), action_list[1], label='Action')
# plt.axhline(np.mean(test_rewards), color='r', linestyle='dashed', linewidth=2, label='Test Mean')
plt.xlabel('Time')
# plt.ylabel('Action of Fishman 2')
plt.title('Actions of fishman B')
# plt.legend()

plt.subplot(2, 3, 3)
plt.ylim(-0.5,8.5)#y轴范围设置
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
ax.xaxis.set_major_locator(x_major_locator)
plt.scatter(range(len(action_list[2])), action_list[2], label='Action')
# plt.axhline(np.mean(test_rewards), color='r', linestyle='dashed', linewidth=2, label='Test Mean')
plt.xlabel('Time')
# plt.ylabel('Action of Fishman 3')
plt.title('Actions of fishman C')
# plt.legend()

plt.subplot(2, 3, 4)
plt.ylim(-0.5,8.5)#y轴范围设置
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
ax.xaxis.set_major_locator(x_major_locator)
plt.scatter(range(len(action_list[3])), action_list[3], label='Action')
# plt.axhline(np.mean(test_rewards), color='r', linestyle='dashed', linewidth=2, label='Test Mean')
plt.xlabel('Time')
# plt.ylabel('Action of Fishman 4')
plt.title('Actions of fishman D')
# plt.legend()



plt.subplot(2, 3, (5,6))
plt.ylim(-50,1550)#y轴范围设置
ax = plt.gca()
y_major_locator=MultipleLocator(300)
ax.yaxis.set_major_locator(y_major_locator)
ax.xaxis.set_major_locator(x_major_locator)
plt.plot(range(len(total_population)), total_population, label='Fish_population')
#plt.axhline(np.mean(test_rewards), color='r', linestyle='dashed', linewidth=2, label='Test Mean')
plt.xlabel('Time')
# plt.ylabel('State')
plt.title('Fish population dynamics')
plt.legend()



# plt.subplot(1, 2, 1)
# plt.plot(range(1, num_episodes + 1), training_rewards, label='Training')
# plt.axhline(np.mean(test_rewards), color='r', linestyle='dashed', linewidth=2, label='Test Mean')
# plt.xlabel('Episodes')
# plt.ylabel('Total Reward')
# plt.title('Fishing in a Pond with Dynamic Fish Population')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(range(1, num_episodes + 1), training_fish, label='Fish Population')
# #plt.axhline(np.mean(test_rewards), color='r', linestyle='dashed', linewidth=2, label='Test Mean')
# plt.xlabel('Episodes')
# plt.ylabel('Fish Population')
# plt.title('Fishing in a Pond with Dynamic Fish Population')
# plt.legend()
plt.tight_layout()
plt.show(block=True)


