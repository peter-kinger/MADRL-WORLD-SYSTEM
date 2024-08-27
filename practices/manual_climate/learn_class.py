import random
import numpy as np
from learn import utils
from envs.AYS.AYS_Environment import *
from learn import agents as ag


import torch


# 导入自己的包
from learn import utils

class Learn:
    def __init__(self,reward_type="PB",
                max_episodes=2000, max_steps=600, max_frames=100000,
                max_epochs=50,seed=0, gamma=0.99, decay_number=0,
                save_loaccly=True): # TODO 暂时还不处理保存本地的数据
        # Initialization code here
        
        # 设置环境
        self.env = AYS_Environment(reward_type=reward_type, discount=gamma)
        self.state_dim = len(self.env.observation_space)
        self.action_dim = len(self.env.action_space)
        self.gamma = gamma

        # # 设置智能体算法
        # # 这里重新书写了
        # self.agent = ag.DQN(self.state_dim, self.action_dim, gamma=self.gamma)

        # seed 
        # 种子在最开始就设定了，seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 截止的参数
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.max_frames = max_frames
        self.max_epochs = max_epochs

        # 筛检率
        self.decay_number = decay_number

        # 保存本地
        # self.save_locally = save_locally

        # # 总数据
        # self.data = {
        #     "rewards": [],
        #     # "losses": [],
        #     # "steps": [],
        #     'moving_avg_rewards': [],
        #     'moving_std_rewards': [],
        #     'frame_idx': 0, # 都会进行等同
        #     'episodes': 0, # same as above
        #     'final_point': []
        #      }
        # run information in a dictionary
        self.data = {'rewards': [],
                     'moving_avg_rewards': [], # 蠢死了，字母拼不对啊！！！
                     'moving_std_rewards': [],
                     'frame_idx': 0,
                     'episodes': 0,
                     'final_point': []
                     }

        # 单独给绘制动作等使用的
        self.data_plot = {
            'learning_progress': [],
            'actions': [],
            'rewards': [],
        }
        
        

    def train(self, notebook=True):
        # Training code here
        print("Training...")
        
        # 已经通过 Set 设置 agent self.agent = ag.DQN(self.state_dim, self.action_dim, gamma=self.gamma)
        if str(self.agent) == "DQN":

            self.decay_number = 6
            # 计算对应的步骤是多少衰减
            # 即 ag.DQN.step_decay = int(self.max_frames / (self.decay_number + 1))
            self.agent.step_decay = int(self.max_frames / (self.decay_number + 1))
            self.learning_loop_offline(256, 32768, per_is=False, notebook=notebook)
        else:
            print("当前没有指定智能体")


    def learning_loop_offline(self, batch_size, buffer_size, per_is=False, notebook=True,
                             plotting=False, alpha=0.213, beta = 0.7389, config=None):
        """
        
        参数说明：
        batch_size: 一次性拿取的数据量
        config: 配置文件(wandb使用的配置文件)
        """
        # TODO 这里以后可以进行大规模的替换重写操作
        
        # 对应叶强中的 learning
        # Learning loop offline code here
        # 这里主要是基于 DQN 相关算法的训练
        # 经验拿取的部分，不涉及到具体的

        # 参数说明：
        # Per_is: 是更加高级的算法提升 PER_IS_ReplayBuffer class  > basic ReplayBuffer class 
        
        # 开始训练的话需要将对应的数据进行对应，尤其是如下写法
        self.data['frame_idx'] = self.data['episodes'] = 0

        # 使用基本的学习算法 ，DQN 中经验出处的一块
        # 本质上就是个拿取经验操作的部分
        # self.memory = utils.PER_IS_ReplayBuffer(buffer_size) if per_is else utils.ReplayBuffer(buffer_size)

        # initiate memory
        self.memory = utils.PER_IS_ReplayBuffer(buffer_size, alpha=alpha,
                                                state_dim=self.state_dim) if per_is else utils.ReplayBuffer(buffer_size)


        # loss = 0
        # 注意这里使用的是 episodes，里面使用的是 steps
        for episodes in range(self.max_episodes):

            # 重新设置环境为初始状态
            # reset 重写了

            state = self.env.reset()
            episode_reward = 0

            step_in_episode = 0
            # 开始循环进行相关计算
            for i in range(self.max_steps):
 
 
                action = self.agent.get_action(state)

                # step through environment
                # 是在环境中进行书写的
                next_state, reward, done, _ = self.env.step(action)

                # add reward
                # 标准的写法，可以参照动手学强化学习里面部分内容
                # 这就是 episode_reward 部分，记录每次的 reward 的相关变化，方便后面来进行显示
                episode_reward += reward

                step_in_episode += 1

                # DQN 部分
                self.memory.push(state, action, reward, next_state, done)

                if len(self.memory) > batch_size:
                    # 判断是否使用优先经验回放
                    if per_is:
                        # 补充：优先经验回放的部分
                        pass
                    else:
                        sample = self.memory.sample(batch_size)
                        loss, _ = self.agent.update(sample) # 这里是discounted_reward 的关键， targets = (rewards + self.gamma * next_state_values.squeeze(1))

                # 准备下一次迭代
                state = next_state
                
                # 作为 episode 里面大小的限制部分
                """
                因为这里的 frame 是作为里面 step 和 episode 总共运行次数的部分
                """
                self.data['frame_idx'] += 1

                # 一个停止限制的条件
                # if the episode is finished we stop there
                if done:
                    break

            # 整个运行完结果才进行加载 reward 部分
            # 添加 episode 奖励
            # 后面单独再书写添加计算数据的部分
            self.append_data(episode_reward) # 类似于 DQN 的标准写法 return_list.append(episode_return)


            # 增加写入其中的绘制奖励函数的部分
            if notebook and episodes % 10 == 0:
                utils.plot(self.data)

            if self.data['frame_idx'] >= self.max_frames:
                break

                
        # 每次运行结果都会对应一个最终状态：final_point
        # 单独使用一个量来计算成功率
        # 自动计算其中的数组储存对应量
        # 计算每次的对应计算量          
        # 注意计算，应该是总共的次数            
        success_rate = self.data['final_point'].count("Green_FP") / self.data['episodes']
        print("现在是对应成功率的结果，也就是每 episode 中达到 可持续边界的次数{0}".format(success_rate))

        # 绘制运行的结果
        # 利用 wandb 动态绘制结果
        
        if plotting:
            utils.plot_wandb(self.data)

        # 如果后续考虑使用 wandb 来显示，那么保存时候最后的部分才 wandb.finish()


    def learning_loop_rollout(self, batch_size, buffer_size, per_is=False, notebook=True):
        # Learning loop rollout code here
        # TODO 目前调用的 DQN  还不需要用到
        """_summary_

        Args:
            batch_size (_type_): _description_
            buffer_size (_type_): _description_
            per_is (bool, optional): _description_. Defaults to False.
            notebook (bool, optional): _description_. Defaults to True.
        """
        pass 

    def set_agent(self, agent_str, pt_file_path=None, second_path=None, **kwargs):
        """
        就是
        self.agent = ag.DQN(self.state_dim, self.action_dim, gamma=self.gamma)
        # 到其中来拿取相关的算法操作
        Args:
            agent_str (_type_): _description_
            pt_file_path (_type_, optional): _description_. Defaults to None.   """
        # 定义其中的衰减率
        step_decay = int(self.max_frames / (self.decay_number + 1))
        try:
            # 这里引入修改加入 DQN 的部分
            # 单纯进行评估
            # print("eval 评估前的智能体：", self.agent)
            # 没有创建之前，都会保存
            self.agent = eval("ag." + agent_str)(self.state_dim, self.action_dim,
                                                 gamma=self.gamma, step_decay=step_decay, **kwargs)
            print("eval 评估后的智能体：", self.agent)
            print("Agent set to", agent_str)
        except:
            print('Not a valid agent, try "Random", "A2C", "DQN", "PPO" or "DuelDDQN".')

        # TODO 后面重写可以加入新的智能体相关算法部分

        """
        上面经过评判 agent 对应的部分了，就可以进行agent 中网络的相关加载了
        """
        if pt_file_path is not None:
            if agent_str == "A2C":
                print("选择加载 a2c")
                pass
            elif agent_str == "DQN":   
                print("选择加载 DQN")
                # 可以使用一个来进行加载，原文中有 copy 部分
                self.agent.policy_net.load_state_dict(torch.load(pt_file_path))
                self.agent.target_net.load_state_dict(torch.load(pt_file_path))

    # def step(self, agent_str, pt_file_path=None, second_path=None, **kwargs):
    #     # Step code here
    #     # 设置智能体对应什么算法以及选用的路径文件对应
    #     # TODO 
        
    #     pass

 

    def append_data(self, episode_reward):

        """
        rewards 就是每次运行一次时间步长的内容
        
        """
        self.data['rewards'].append(episode_reward)
        # 分别计算的标准不一样
        # 'moving_avg_rewards': [],  # 蠢死了，字母拼不对啊！！！
        # 'moving_std_rewards': [],
        self.data['moving_avg_rewards'].append(np.mean(self.data['rewards'][-50:]))
        self.data['moving_std_rewards'].append(np.std(self.data['rewards'][-50:]))
        self.data['episodes'] += 1
        # 添加的是具体的名字对应
        # 说明
        #     OUT_PB = 0
        #     BLACK_FP = 1
        #     GREEN_FP = 2
        #
        #     A_PB = 3
        #     Y_SF = 4
        #     S_PB = 5
        
        # 因为存在情况我们可能没有到达边界，所以对终点进行判断分析结果
        self.data['final_point'].append(self.env.which_final_state())

        # 冗余部分就是 verbose 来控制打印输出
        print("Episode:", self.data['episodes'], "|| Reward:", round(episode_reward), "|| Final State ",
              self.env.which_final_state())

    def plot_trajectory(self, colour, start_state=None, steps=600, fname=None, axes=None, fig=None):
        """To plot trajectories of the agent
        主要基本的运行
        包含关键函数：
        1 运行部分
        2 绘制过程部分
        3 头发丝部分
        
        """
        state = self.env.reset_for_state(start_state)
        learning_progress = []
        actions = []
        rewards = []
        # 模拟的 steps 和真实的 steps 一样
        # max_steps=600
        # 类似于测试中的部分操作
        # 因为里面测试部分已经固定了，所以不涉及变动，
        # 相当于复现了一遍
        for step in range(steps):
            # 拿到当前的状态部分
            list_state = self.env.get_plot_state_list()

            # take recommended action
            action = self.agent.get_action(state, testing=True)

            # Do the new chosen action in Environment
            new_state, reward, done, _ = self.env.step(action)
            actions.append(action)
            rewards.append(reward) # 类似于其中的 episode_reward += reward
            learning_progress.append([list_state, action, reward])

            state = new_state
            if done:
                break

        # 关键的传入参数就是 learning_progress
        # 类似于之前的代码部分
            
        # fig 和 外面的 axes 都是匹配的
        self.env.plot_run(learning_progress, fig=fig, ax3d=axes, fname=fname,colour=colour )

        # TODO 头发丝部分
        utils.plot_hairy_lines(num=20, ax3d=axes)
        #
        # def action_plot():
        #     """
        #     用于获取里面的 actions 部分原件
        #
        #     :return:
        #     """
        #     print("开始返回动作等")
        #     return actions, rewards

        # 单独写在外面接受来操作
        # return actions, rewards

        self.data_plot['actions'] = actions
        self.data_plot['rewards'] = rewards


if __name__ == "__main__":
    # 需要测试满足几个函数的基本写法满足要求
    pass