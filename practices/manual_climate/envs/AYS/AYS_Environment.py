"""作为耦合 step 和 ays_model 部分的函数
目的在于方便 rl 与 ays_environment 部分的耦合

"""
from learn import utils


from scipy.integrate import odeint
from . import ays_model as ays
import numpy as np
from gym import Env

import mpl_toolkits.mplot3d as plt3d
import matplotlib.pyplot as plt


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



# 必要要继承 env 类
class AYS_Environment(Env):
    """
    动态变量有 3 个：
    A
    Y
    S

    参数：
    sim_time: 时间步长采取的步骤
    每个格点可以选择的动作，default/DG/ET/None
    """

    # 定义基本参数维度等量
    dimensions = np.array(['A', 'Y', 'S'])
    management_options = np.array(['default', 'DG', 'ET', 'LG+ET'])  # 4 种管理方式
    action_space = [(False, False), (True, False), (False, True), (True, True)]  # 4 种动作空间
    action_space_number = np.arange(len(action_space))  # 4 种动作空间的数量

    # 定义改编复杂系统的相关参数部分
    # AYS example from Kittel et al. 2017:
    tau_A = 50
    tau_S = 50
    beta = 0.03
    beta_LG = 0.015
    eps = 147
    A_offset = 600
    theta = beta / (950 - A_offset)  # beta / ( 950 - A_offset(=350) )


    rho = 2.
    sigma = 4e12
    sigma_ET = sigma * 0.5 ** (1 / rho)

    phi = 4.7e10


    def __init__(self, discount=0.99, t0=0, dt=1, reward_type="PB", max_steps=600, run_number=0, **kwargs):
        # Initialization code here

        self.gamma = discount
        self.max_steps = max_steps

        self.green_fp = [0, 1, 1]
        self.brown_fp = [0.6, 0.4, 0]
        self.final_radius = 0.05  # Attention depending on how large the radius is, the BROWN_FP can be reached!

        # Definitions from outside
        self.current_state = [0.5, 0.5, 0.5]
        # 因为开始的运行的状态和开始的状态是一致的，但是 state 会变化，而 start_state 不会变化
        self.state = self.start_state = self.current_state
        # gym 的关键对应
        self.observation_space = self.state

        # 增加最终基础装填和 reward
        self.reward = 0
        self.final_state = False


        """
        This values define the planetary boundaries of the AYS model
        """
        # 这里定义基本的边界限制
        #  后面考虑使用复杂的函数来计算，这里先简单定义相关的值操作
        # 注意无限和单独的相关计算设置
        # 直接在在本地已经换算好了
        self.A_PB = 0.5897 # # Planetary boundary: 0.5897
        self.Y_SF = 0.5 # Social foundations as boundary: 0.3636
        self.S_LIMIT = 0

        # 拼接成数组
        self.PB = np.array([self.A_PB, self.Y_SF, self.S_LIMIT]) # # array([0.5897, 0.3836, 0.    ])

        # 定义奖励部分
        self.reward_type = reward_type
        self.reward_function = self.get_reward_function(reward_type)

        # 定义其中的时间信息
        timeStart = 0
        intSteps = 10 # 这是积分的步数
        self.t = self.t0 = t0 # 上面的初始化参数有
        self.dt = dt # 这是时间步长

        
    
    def step(self, action: int):
        """单步的多元组计算
        比如考虑了特殊情况：
        - 截止条件等
        - 单独的奖励如何进行对应
        - 每次运行的轨迹也是 self.t + self.dt 这样 dt 的步幅进行变化
        
        """
        # Action code here
        # 执行了DRL 算法中执行模拟与复杂系统耦合的关键部分
        # 函数通过状态更新并且根据所选的奖励函数接受奖励

        # 每次更新相应的时间步长取决了数值模拟的精度
        next_t = self.t + self.dt  # next_t 作为一个单独的量来记录
        # 获得下一步采取什么动作
        self.state = self._perform_step(action, next_t)

        self.t = next_t 

        # 补充：原文中的三个截止条件
        # crosses a Planetary Boundary
        # reaches the vicinity of the Black Final Point
        # reaches the vicinity of the Green Final Point

        # 设置特殊条件的判断
        # reaches the vicinity of the Black Final Point
        # reaches the vicinity of the Green Final Point
        if self._arrived_at_final_state():
            self.final_state = True # 传递记录的表格
        
        reward = self.reward_function(self.state) # 获取五元组所需要的相关量耦合的部分

        # 多个条件判断是否到达边界
        # crosses a Planetary Boundary
        if not self._inside_planetary_boundaries():
            self.final_state = True

        # 计算是否到达边界辅助上未来奖励折损的相关计算内容
        if self.final_state and self.reward_type == "PB":
            """奖励改变的关键：
            事实上动作的变化是由于奖励而进行变化的，而根据原来 AYS 中的图形运行部分，
            只有有个条件发生才会进行更大的奖励的实际计算（也就是平时的 RL 不一定使用衰减奖励的思想）
            - 越过边界
            - 靠近绿点
            - 靠近黑点

            所以图中动作的改编只在 shelter 和 backwater 中进行变化

            （系统本身是会随着时间的变化进行状态改变的）
            这一点是区别于传统 RL 每次对获取奖励进行变化，比如走格子
            这个和 Go 里面的原理非常类似就·
            
            """
            reward += self.calculate_expected_reward()
        
        return self.state, reward, self.final_state, None


    #################################################
    # 以下是一些辅助函数
    def _perform_step(self, action, next_t):
        # 这就是 apply_obs
        # 通过执行动作，返回下一个状态
        # 这里是一个简单的例子
        # 可以根据具体的情况进行修改

        # 由于是根据 action 来控制参数的，还需要书写一个控制参数对应的函数
        parameter_list = self._get_parameters(action)

        # 利用微分方程进行相关求解
        # 这里详细参照 ipynb 中的教程
        #
        # 注意完整的参数输入
        # traj_one_step = odeint(ays.AYS_rescaled_rhs, self.state, [self.t, next_t], args=parameter_list[0], mxstep=50000)
        traj_one_step = odeint(ays.AYS_rescaled_rhs, self.state, [self.t, next_t], args=parameter_list[0], mxstep=50000)

        # 利用数值来对应反映轨迹
        a = traj_one_step[:, 0][-1]
        y = traj_one_step[:, 1][-1]
        s = traj_one_step[:, 2][-1] 

        return np.array([a, y, s])
    
    def _get_parameters(self, action):
        """
        这个函数控制了到底每个动作到底控制了哪个参数的量
        比如 action 1 控制了 alpha 参数， acgtion 对应了 beta 参数
        """

        # 动作必须满足动作空间长度的要求
        if action < len(self.action_space):
            action_tuple = self.action_space[action]
        else:
            raise ValueError("动作不在动作空间中")
        
        # beta = 0.03
        # beta_LG = 0.015
        # 指定了复杂系统中每个量的相关对应
        # 缺少值了
        # sigma = 4e12
        # sigma_ET = sigma * 0.5 ** (1 / rho)
        """默认参数
        beta = 0.03 # 3%/year economic output growth
        eps = 147
        phi = 4.7e10 # Fossil fuel conbustion efficiency
        rho = 2.    # Learning rate of renewable knowledge
        sigma = 4e12 
        tau_A = 50
        tau_S = 50
        theta = beta / (950 - A_offset)  # beta / ( 950 - A_offset(=350) )
        
        """
        parameter_list = [(self.beta_LG if action_tuple[0] else self.beta, self.eps, self.phi, self.rho, 
                           self.sigma_ET if action_tuple[1] else self.sigma, self.tau_A, self.tau_S, self.theta)]
           
        return parameter_list # 返回的结果应该是一个列表 [beta, eps, phi, rho, sigma] 对应具体值

    #################################################
    # 状态的相关识别 
    # 考虑增加判断边界实验模拟的相关函数部分
    # 比如 
    # self._arrived_at_final_state():
    # self._inside_planetary_boundaries():
    # self.calculate_expected_final_reward()

    # 最开始的重设状态
    def reset(self):
        # self.state=np.array(self.random_StartPoint())
        self.state = np.array(self.current_state_region_StartPoint())
        # self.state=np.array(self.current_state)
        self.final_state = False
        self.t = self.t0
        return self.state
    

    # 测试专用的重设状态部分
    def reset_for_state(self, state=None):
        """清空所有状态回到初始化
        区别与上面的状态变化
        因为它可以进行 state 的设置，比如设置初始状态为自定义的参数 state
        e.g:
        state = self.env.reset_for_state(test_states[i])

        Args:
            state (_type_): _description_
        """
        if state is None:
            self.start_state = self.state = np.array(self.current_state)
        else:
            self.start_state = self.state = np.array(state)
        self.final_state = False
        self.t = self.t0
        return self.state


    def current_state_region_StartPoint(self):

        self.state = [0, 0, 0]
        limit_start = 0.05
        while not self._inside_planetary_boundaries():
            # self.state=self.current_state + np.random.uniform(low=-limit_start, high=limit_start, size=3)
            self.state[0] = self.current_state[0] + np.random.uniform(low=-limit_start, high=limit_start)
            self.state[1] = self.current_state[1] + np.random.uniform(low=-limit_start, high=limit_start)
            self.state[2] = self.current_state[2] #+ np.random.uniform(low=-limit_start, high=limit_start)

        # print(self.state)
        return self.state

    def _arrived_at_final_state(self):
        # reaches the vicinity of the Black Final Point
        # reaches the vicinity of the Green Final Point
        a, y, s = self.state
        # self.final_radius = 0.05  # Attention depending on how large the radius is, the BROWN_FP can be reached!
        
        # self.green_fp = [0, 1, 1]
        # self.brown_fp = [0.6, 0.4, 0]

        # green 部分对应
        if np.abs(a - self.green_fp[0]) < self.final_radius and np.abs(y - self.green_fp[1]) < self.final_radius and np.abs(s - self.green_fp[2]) < self.final_radius:
            return True
        # brown 部分对应
        elif np.abs(a - self.brown_fp[0]) < self.final_radius and np.abs(
                y - self.brown_fp[1]) < self.final_radius and np.abs(s - self.brown_fp[2]) < self.final_radius:
            return True
        
        else:
            return False

    def _inside_planetary_boundaries(self):
        # crosses a Planetary Boundary
        # 设置其中的边界越过条件
        a, y, s = self.state
        is_inside = True
        if a > self.A_PB or y < self.Y_SF or s < self.S_LIMIT:
            return False
        else:
            return True
        
    # def _good_final_state(self):
    #     a, y, s = self.state
    #     if np.abs(a - self.green_fp[0]) < self.final_radius and np.abs(
    #             y - self.green_fp[1]) < self.final_radius and np.abs(s - self.green_fp[2]) < self.final_radius:
    #         return True
    #     else:
    #         return False


    def which_final_state(self):
        """
        根据值判断当前状态属于哪个最终状态
        :return:
        """

        # 这里直接显式地定义操作了
        # Basins.GREEN_FP = 2
        GREEN_FP = 2
        BLACK_FP = 1

        a, y, s = self.state
        if np.abs(a - self.green_fp[0]) < self.final_radius and np.abs(
                y - self.green_fp[1]) < self.final_radius and np.abs(s - self.green_fp[2]) < self.final_radius:
            # print("ARRIVED AT GREEN FINAL STATE WITHOUT VIOLATING PB!")
            # return Basins.GREEN_FP
            return GREEN_FP
        elif np.abs(a - self.brown_fp[0]) < self.final_radius and np.abs(
                y - self.brown_fp[1]) < self.final_radius and np.abs(s - self.brown_fp[2]) < self.final_radius:
            # return Basins.BLACK_FP
            return BLACK_FP
        else:
            # return Basins.OUT_PB
            # 超过边界则只返回当前的边界判断
            return self._which_PB()


    def _which_PB(self, ):
        """ To check which PB has been violated"""
        # 超过边界只返回当前的边界判断
        # 其中是具有两个边界的
        A_SF = 1
        A_PB = 3
        Y_SF = 4
        S_PB = 5
        OUT_OF_TIME = 6

        if self.state[0] >= self.A_PB:
            return A_PB
        elif self.state[1] <= self.Y_SF:
            return Y_SF
        elif self.state[2] <= 0:
            return S_PB
        else:
            return OUT_OF_TIME


    #################################################
    # 奖励函数的类型定义部分
        
    def get_reward_function(self, choice_of_reward):
        """包含很多对应的种类，这是实验的关键
        返回对应的奖励函数
        """

        def reward_final_state(action=0):
            pass

        def reward_type2(action=0):
            pass

        def reward_type3(action=0):
            pass
        # ......

        def reward_final_PB(action=0):
            # 利用范数来计算长度
            if self._inside_planetary_boundaries():
                reward = np.linalg.norm(self.state - self.PB)
            else:
                reward = 0.
            return reward
        
        def simple_reward(action=0):
            # 简单的奖励函数
            return 1


        if choice_of_reward == "PB":
            return reward_final_PB
        elif choice_of_reward == "Type2":
            return reward_type2
        elif choice_of_reward == "Type3":
            return reward_type3
        elif choice_of_reward == "simple":
            return simple_reward
        elif choice_of_reward == None:
            print("没有对应的奖励函数")
        else:
            raise ValueError("没有对应的奖励函数")
        
    def calculate_expected_reward(self):
        """路易师姐的提示
        为什么要单独计算未来奖励，因为虽然因为外界条件终止了，但是这些奖励对于轨迹的相关考虑还是要纳入其中
        
        """
        reward_final_state = self.reward_function()
        
        
        remaining_steps = self.max_steps - self.t

        discounted_future_reward = 0
        # 累积的折扣汇报是对未来奖励作出相关计算
        for i in range(remaining_steps):
            discounted_future_reward += self.gamma ** i * reward_final_state
            
        return discounted_future_reward


###############################################
# 绘制的部分
    
    
    def plot_run(self, learning_progress, fig, ax3d, colour, fname=None,):
        timeStart = 0
        intSteps = 2  # integration Steps
        dt = 1
        sim_time_step = np.linspace(timeStart, self.dt, intSteps)
        # if axes is None:
        #     fig, ax3d = create_figure()
        # else:
        # ax3d = axes

        start_state = learning_progress[0][0]

        for state_action in learning_progress:
            state = state_action[0]
            action = state_action[1]
            parameter_list = self._get_parameters(action)
            traj_one_step = odeint(ays.AYS_rescaled_rhs, state, sim_time_step, args=parameter_list[0])
            # Plot trajectory

            color_list=['#e41a1c','#ff7f00','#4daf4a','#377eb8','#984ea3']
            
            my_color = color_list[action]
            ax3d.plot3D(xs=traj_one_step[:, 0], ys=traj_one_step[:, 1], zs=traj_one_step[:, 2],
                        color=colour, alpha=0.3, lw=3)

        # Plot from startpoint only one management option to see if green fix point is easy to reach:
        # self.plot_current_state_trajectories(ax3d)
        # utils.plot_hairy_lines(20, ax3d)


        
        
        
    def get_plot_state_list(self):
        return self.state.tolist()[:3]

