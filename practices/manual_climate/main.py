
from experiments import *
import seaborn as sns
sns.set_theme()

import os
os.environ['WANDB_API_KEY'] = 'a0588ba2a7092e316ecd1459c25378695144de82'

import wandb
wandb.login()

# 类的相关写法，里面包含了很多的 agent 部分
# 因为原始的 ays 中知识运行了一次，所以这里就直接单独写了，，没有加 for 循环

experiment = PB_Learn(max_frames=1e5) # experiment class: Markov_Learn, Noisy_Learn, Simple_Learn, different number of frames have different effects on plots
# experiment.set_agent("PPO") # agent name: DQN, DuelDDQN, Random, PPO, A2C, add parameters as kwargs
print("运行了")

experiment.set_agent("DQN")
# 下面的语句是通过上面的 set_agent 来进行指定选择的
print("运行了2")
experiment.train() # approx 10 minutes
print("运行了3")
# 还挺快的，我的电脑 4min 就运行完毕了


"""操作补充

增加了 wandb 改为了 True
notebook 改成了 False
配置 wandb 

# 报错，可能是因为实体的问题，
wandb.errors.CommError: It appears that you do not have permission to access the requested resource. Please reach out to the proj


关闭 notebook 的话，就会出现下面的错误

"""


