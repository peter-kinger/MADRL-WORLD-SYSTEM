

import learn_class as lc
dqn1 = lc.Learn()
print(dqn1)

test_path = "240321_policy_net.pth"
dqn1.set_agent("DQN", pt_file_path=test_path)
print(dqn1.agent.policy_net.state_dict())


import matplotlib.pyplot as plt
fig, axes = plt.subplots(subplot_kw={'projection': '3d'})
dqn1.plot_trajectory([0.5, 0.5, 0.5], start_state=[0.5, 0.5, 0.5], axes=axes)

# 把 plot_trjectory 里面定义的 actions 和rewards 打印出来
# print(dqn1.actions)
# print(dqn1.rewards)

# dqn1.plot_trajectory([0.5, 1, 0.5], axes=axes)
# self.current_state = [0.5, 0.5, 0.5]

# dqn1.plot_trajectory([0.5, 0.5, 0.5], start_state=[2, 0.5, 0.5], axes=axes)

# 在 ax3d 上图像进行添加绘制一个线
axes.plot([0, 1], [0, 1], [0, 1], lw=4, color='red')

axes.scatter(*zip([0.5,0.5,0.5]), lw=4, color='pink')
axes.scatter(*zip([0.60,0.38,0]), lw=4, color='black')
axes.scatter(*zip([0,1,1]), lw=4, color='lime')

axes.set_title('Sparse', fontsize=30)
plt.tight_layout()

plt.legend()
# 最后再进行显示部分
plt.show()


"""解释：
为什么原文中只有 20 次这么稀少，
因为其中有个累积的结果在上面，所以只有 20 次累积 8次效果就比较好了


"""


# 绘制 actions 和 rewards 的图
plt.figure(figsize=(16, 9))
ax = plt.gca()
plt.scatter(range(len(dqn1.data_plot['actions'])), dqn1.data_plot['actions'], label='Action')

plt.show()