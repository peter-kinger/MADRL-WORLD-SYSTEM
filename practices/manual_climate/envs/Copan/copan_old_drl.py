"""
This is the implementation of the c:GLOBAL Environment in the form
that it can used within the Agent-Environment interface 
in combination with the DRL-agent.

@author: Felix Strnad
"""
import sys

import numpy as np
from scipy.integrate import odeint

from gym import Env
from collections import OrderedDict
from DeepReinforcementLearning.Basins import Basins
import mpl_toolkits.mplot3d as plt3d
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnchoredText
import matplotlib.ticker as ticker


import heapq as hq
import functools as ft
import operator as op
INFTY_SIGN = u"\u221E"


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    

from inspect import currentframe, getframeinfo

def get_linenumber():
    print_debug_info()
    print("Line: ")
    cf = currentframe()
    return cf.f_back.f_lineno

def print_debug_info():
    frameinfo = getframeinfo(currentframe())
    print ("File: ", frameinfo.filename)
    
@np.vectorize
def compactification(x, x_mid):
    if x == 0:
        return 0.
    if x == np.infty:
        return 1.
    # 也就是计算其中的初始值，这个应该被反复调用
    return x / (x + x_mid)

@np.vectorize
def inv_compactification(y, x_mid):
    if y == 0:
        return 0.
    if np.allclose(y, 1):
        return np.infty
    return x_mid * y / (1 - y)



#########################################
# 其实整个就是主要是这个类的部分，其他都是相似的

class cG_LAGTPKS_Environment(Env):
    """
    This Environment describes the 7D implementation of the copan:GLOBAL model developed by Jobst Heitzig.
    The parameters are taken from Nitzbon et al. 2017. 
    The code contains implementation parts that go back to Jan Nitzbon 2016 
    Dynamic variables are :
        - terrestrial ("land") carbon L
        - excess atmospheric carbon stock A
        - geological carbon G
        - temperature T
        - population P
        - capital K
        - the renewable energy knowledge stock S
    
    Parameters (mainly Nitzbon et al. 2016 )
    ----------
        - sim_time: Timestep that will be integrated in this simulation step
          In each grid point the agent can choose between subsidy None, A, B or A and B in combination. 
        - Sigma = 1.5 * 1e8
        - CstarPI=4000
        - Cstar=5500
        - a0=0.03
        - aT=3.2*1e3
        - l0=26.4
        - lT=1.1*1e6
        - delta=0.01
        - m=1.5
        - g=0.02
        - p=0.04
        - Wp=2000
        - q0=20
        - b=5.4*1e-7
        - yE=147
        - eB=4*1e10
        - eF=4*1e10
        - i=0.25
        - k0=0.1
        - aY=0.

        - aB= 3e5 (varied, basic year 2000)
        - aF= 5e6 (varied, basic year 2000)
        - aR= 7e-18 (varied, basic year 2000)
        - sS=1./50.
        - sR=1.
    """

    # 管理选项就是对应了操作的部分
    management_options=['default', 
                        'Sub' , 'Tax','NP' ,
                        'Sub+Tax', 'Sub+NP', 'Tax+NP',
                        'Sub+Tax+NP' ]

#     management_options=['default', 
#                         'SubTax','DG' , 'NP',
#                         'SubTax+DG', 'SubTax+NP', 'DG+NP',
#                         'SubTax+DG+NP' ]
    action_space=[(False, False, False), 
                        (True, False,False), (False, True, False), (False, False, True),
                        (True, True, False), (True, False, True) , (False, True, True),
                        (True, True, True)
                        ]
    dimensions=np.array(['L', 'A', 'G', 'T', 'P', 'K', 'S'])
    
    def __init__(self, t0=0, dt=1 , reward_type=None, image_dir=None, run_number=0, plot_progress=False,
                 ics=dict(  L=2480.,  
                            A=758.0,
                            G=1125,
                            T=5.053333333333333e-6,
                            P=6e9,
                            K=6e13,
                            S=5e11
                            ) , # ics defines the initial values!
                 pars=dict( Sigma = 1.5 * 1e8,
                            Cstar=5500,
                            a0=0.03,
                            aT=3.2*1e3,
                            l0=26.4,
                            lT=  1.1*1e6,
                            delta=0.01,
                            m=1.5,
                            g=0.02,
                            p=0.04,
                            Wp=2000,
                            q0=20,
                            b=5.4*1e-7,
                            yE=147,
                            eB=4*1e10,
                            eF=4*1e10,
                            i=0.25,
                            k0=0.1,
                            aY=0.,
                            aB=3e5,
                            aF=5e6,
                            aR=7e-18,
                            sS=1./50.,
                            sR=1.,
                            ren_sub=.5,
                            carbon_tax=.5,
                            i_DG=0.1,
                            L0=0,
                            )   , # pars contains the parameters for the global model
                 
                 # 这是单独考虑了温室气体等相关效应来进行计算的
                 specs=[] # contains specifications for the global model as e.g. the INST_GREENHOUSE  
                 ):
        
        self.image_dir=image_dir
        self.run_number = run_number
        self.plot_progress=plot_progress
        # The grid defines the number of cells, hence we have 8x8 possible states
        self.final_state=False
        self.reward=0
        
        self.reward_function=self.get_reward_function(reward_type)
        
        timeStart = 0
        intSteps = 10    # integration Steps

        self.t=self.t0=t0


        self.dt=dt
        
        self.sim_time_step=np.linspace(timeStart,dt, intSteps)
        
        # 可以删除这个选项内容，
        # "INST_DIFF" 可能指的是 "瞬时扩散"，而 "INST_GH" 可能指的是 "瞬时温室效应"。
        self.specs=specs

        # 调用设置相关的参数内容对应
        self.setParams(paramDict=pars)
        # 调整初始值，一些值是通过计算得到的
        self.setInitials(iniDict=ics)
        
        
        # Definitions from outside
        self.state=self.current_state=np.array([self.iniDynVar['L'], self.iniDynVar['A'], self.iniDynVar['G'], self.iniDynVar['T'], 
                                       self.iniDynVar['P'], self.iniDynVar['K'], self.iniDynVar['S'] 
                                       ])
        self.state=self.start_state=self.current_state
        
        # 默认的自然保护是关闭的 
        self.Lprot=False
        # 观测关键的相关代替
        self.observation_space=self.state
        
        # Planetary Boundaries for A, Y, P (population!)
        self.A_PB=945
        self.A_scale=1
        self.Y_PB=self.direct_Y(self.iniDynVar['L'], self.iniDynVar['G'], self.iniDynVar['P'], self.iniDynVar['K'], self.iniDynVar['S'])
        self.P_PB=1e6
        self.W_PB= (1- self.params['i'])*self.Y_PB / (1.01*self.iniDynVar['P'])   # Economic production in year 2000 and population in year 2000
        self.W_scale=1e3

        # 对比：
        # self.PB=[self.A_PB, self.Y_SF,0]
        self.PB=np.array([self.A_PB, self.W_PB, self.P_PB])
        

        # 重新进行书写，参考 ays 中的写法，不包含这一部分内容
        self.compact_PB=compactification(self.PB, self.ini_state)    

        self.P_scale=1e9
        self.reward_type=reward_type
        
        print("Initialized c:GLOBAL environment!" ,
              "\nReward type: " + str(reward_type),
              "\nPlanetary Boundaries are: " + str(self.PB),
              "\nInitial LAGTPKS-values are: " + str(self.ini_state),
              "\nInitial derived values are: Wini:"+str(self.Wini)+"Yini: "+str(self.Yini))
        
        # 暂时不写，可以不用这个部分
        self.color_list=['orangered', 'mediumvioletred', 'darkgreen', 'midnightblue', 'yellow', 'goldenrod', 'slategrey', 'olive' ] # Contains as many numbers as management options!

    """
    This function is only basic function an Environment needs to provide
    """
    def step(self, action):
        """
        This function performs one simulation step in a RFL algorithm. 
        It updates the state and returns a reward according to the chosen reward-function.
        """

        next_t= self.t + self.dt

        # 关键区别！
        self._adjust_parameters(action)

        
        self.state=self._perform_step( next_t)
        self.t=next_t

        if self._arrived_at_final_state():
            self.final_state = True
        
        reward=self.reward_function(action)
        
        if not self._inside_planetary_boundaries():
            self.final_state = True
            #print("Left planetary boundaries!" + str(self.state))

        trafo_state=compactification(self.state, self.current_state)
        #print(self.state, trafo_state)
    #    return self.state, reward, self.final_state


    ##################
    # 区别，这里未来奖励的计算是自己单独进行定义的
    # calculate_expected_reward


        return trafo_state, reward, self.final_state
    
    
    
    def _perform_step(self, next_t):
        
        #print(parameter_list[0])
        #print(self.state)
        
        # 因为参数写到了外面
        # ays 中的写法是放在并行中操作
        # self.state=self._perform_step(action, next_t)

        traj_one_step=odeint(self.dDynVar, self.state, [self.t, next_t] , mxstep=50000)
        l = traj_one_step[:,0][-1]
        a = traj_one_step[:,1][-1]
        g = traj_one_step[:,2][-1]
        t = traj_one_step[:,3][-1]
        p = traj_one_step[:,4][-1]
        k = traj_one_step[:,5][-1]
        s = traj_one_step[:,6][-1]
        
        #l,a,g,t,p,k,s= self.state

        return np.array( (l,a,g,t,p,k,s) )
    
    """
    This functions are needed to reset the Environment to specific states
    """
    def reset(self):
        self.start_state=self.state=np.array(self.current_state_region_StartPoint())
        # 多了压缩子
        trafo_state=compactification(self.state, self.current_state)

        self.final_state=False
        self.t=self.t0
#        return self.state    
        return trafo_state    
    
    
    def reset_for_state(self, state=None):
        if state==None:
            self.start_state=self.state=self.current_state
        else:
            self.start_state=self.state=np.array(state)
        self.final_state=False
        self.t=self.t0
        # 多了压缩子
        trafo_state=compactification(self.state, self.current_state)
        
        return trafo_state
        
    
    """
    This function defines the reward the Environment returns to the player for a given action
    """
    def get_reward_function(self,choice_of_reward):
        """
        This function returns one function as a function pointer according to the reward type we chose 
        for this simulation.
        """
        def reward_final_state(action=0):
            """
            Reward in the final  green fixpoint_good 100. , else 0.
            """
            if self._good_final_state():
                reward=2.
            else:
                if self._inside_planetary_boundaries():
                    reward=1.
                else:
                    reward=0.
            return reward
        
        def reward_ren_knowledge(action=0):
            """
            We want to:
            - maximize the knowledge stock of renewables S 
            """
            l,a,g,t,p,k,s = self.state
            if self._inside_planetary_boundaries():
                reward=compactification(s, self.iniDynVar['S'])
            else:
                reward=0.
            
            return reward       
        def reward_desirable_region(action=0):
            l,a,g,t,p,k,s = self.state
            desirable_share_renewable=self.iniDynVar['S']
            reward=0.
            if s >= desirable_share_renewable:
                reward=1.
            return reward
        
        def reward_survive(action=0):
            if self._inside_planetary_boundaries():
                reward=1.
            else:
                reward=0.
            return reward
        
        def reward_survive_cost(action=0):
            cost_managment=0.03
            if self._inside_planetary_boundaries():
                reward=1.
                if self.management_options[action] != 'default':
                    reward -=cost_managment
            else:
                reward=-1e-30
            
            return reward
        
        def reward_distance_PB(action=0):
            L,A,G,T,P,K,S=  self.state
            Leff=L
            if self.Lprot:
                Leff=max(L-self.L0, 0)

            # 计算福祉时候使用了 Leff来进行计算
            W=self.direct_W(Leff, G, P, K, S)
            
            #norm=np.linalg.norm(np.array([(A - self.A_PB)/Aini , (W-self.W_PB)/Wini, (P-self.P_PB)/Pini ]))
            #norm = (self.state[0] - self.A_PB)**2 
            if self._inside_planetary_boundaries():
                # 3 维度的计算距离
                norm=np.linalg.norm( self.compact_PB -  compactification( np.array([A, W, P]), self.ini_state))
                #print("reward-function: ", norm)
                reward=norm
            else:
                reward=0.
            
            return reward
         
        if choice_of_reward=='final_state':
            return reward_final_state
        elif choice_of_reward=='ren_knowledge':
            return reward_ren_knowledge
        elif choice_of_reward=='desirable_region':
            return reward_desirable_region
        elif choice_of_reward=='PB':
            return reward_distance_PB
        elif choice_of_reward=='survive':
            return reward_survive
        elif choice_of_reward=='survive_cost':
            return reward_survive_cost
        elif choice_of_reward==None:
            print("ERROR! You have to choose a reward function!\n",
                   "Available Reward functions for this environment are: PB, rel_share, survive, desirable_region!")
        else:
            print("ERROR! The reward function you chose is not available! " + choice_of_reward)
            print_debug_info()
            sys.exit(1)
   
    
    """
    This functions define the dynamics of the copan:GLOBAL model
    """
    def dDynVar(self, LAGTPKS, t):
        #auxiliary functions

        # 比起 ays 它将过程通过函数更加显式地进行了定义操作
        #photosynthesis
        def phot(L, A, T):
            return (self.params['l0']-self.params['lT']*T)*np.sqrt(A)/np.sqrt(self.params['Sigma'])
        
        #respiration
        def resp(L, T):
            return self.params['a0']+self.params['aT']*T
        
        #diffusion atmosphere <--> ocean
        def diff(L, A, G=0.):
            return self.params['delta']*(self.params['Cstar']-L-G-(1+self.params['m'])*A)
        
        def fert(P,W):
            return 2*self.params['p']*self.params['Wp']*W/(self.params['Wp']**2+W**2) 
    
        def mort(P,W):
            return self.params['q0']/(W) + self.params['qP']*P/self.params['Sigma']
        
        
        L, A, G, T, P, K, S= LAGTPKS

        # 调整基本的参数部分，暂时不写
        #adjust to lower and upper bounds
        L=np.amin([np.amax([L, 1e-12]), self.params['Cstar']])
        A=np.amin([np.amax([A, 1e-12]), self.params['Cstar']])
        G=np.amin([np.amax([G, 1e-12]), self.params['Cstar']])
        T=np.amax([T, 1e-12])
        P=np.amax([P, 1e-12])
        K=np.amax([K, 1e-12])
        S=np.amax([S, 1e-12])

        # # calculate T and A if instantaneous processes
        # if 'INST_DIFF' in self.specs:
        #     A = (self.params['Cstar']-L-G) / (1.+self.params['m'])
        # if 'INST_GH' in self.specs:
        #     T = A/self.params['Sigma']


        #calculate auxiliary quantities
        
        # if self.Lprot:
        #     Leff=max(L-self.L0, 0)
        # else:
        #     Leff=L


        # 和下面的计算过程都是类似的过程
        Leff = L

        Xb=self.params['aB']*Leff**2.
        Xf=self.params['aF']*G**2.
        Xr=self.params['aR']*S**2. 

        X=Xb+Xf+Xr
        
        expP=2./5.
        expK=2./5.

#         # 自然保护方案的内容计算
#         if 'KproptoP' in self.specs:
# #             expP=4./5.
# #             expK=0.
#             K = P*self.iniDynVar['K']/(self.iniDynVar['P'])
#         if 'NproptoP' in self.specs:
#             expP-=1./5.   # gives in combination expP=3./5

        #######################################
        # step1

        Z=self.Z(P, K, X, expP, expK)
        
        #calculate derived variables
        # 确定了生物质 B、化石 F 和可再生能源 R
        B=self.B(Xb, Z)
        F=self.F(Xf, Z)
        R=self.R(Xr, Z)

        # Economic output i
        Y=self.Y(B, F, R)
        # wellbeing is determined by per capita
        W=self.W(Y, P, L)



        #########################################
        # 真正的动态过程内容
        """前面都是理论计算，现在才是实际对应的变量
        
        """
        #calculate derivatives of the dynamic variables


        dL = (phot(L, A, T) - resp(L, T)) * L - B
        #print(self.phot(L, A, T) *L  , self.resp(L, T)*L , B , T)
        dA = -dL + diff(L, A, G)
        dG = -F
        dT = self.params['g'] * (A/self.params['Sigma'] - T)
        dP = P * (fert(P,W)-mort(P,W))
        dK = self.params['i'] * Y - self.params['k0'] * K
        dS = self.params['sR']*R - self.params['sS']*S


        # if 'INST_DIFF' in self.specs:
        #     dA = -(dL+dG)/(1.+self.params['m'])
        # if 'INST_GH' in self.specs:
        #     dT = dA/self.params['Sigma']
        
        #print(t, self.Lprot, L,  self.L0 , Leff, B, phot(L, A, T) , resp(L, T) )
        #print(Y, K, self.params['i'], self.params['k0'], dK)
        #print(R, S, self.params['sS'], dS)
        #print(W, P, dP)
        return [dL, dA, dG, dT, dP, dK, dS]        

    def setInitials(self,iniDict): # TODO
        self.iniDynVar=OrderedDict()

        """使用例子：重新复制对应
        from collections import OrderedDict

        # 创建一个空的有序字典
        self.iniDynVar = OrderedDict()

        # 填充初始动态变量的值
        self.iniDynVar['L'] = 0.5
        self.iniDynVar['A'] = 0.3
        self.iniDynVar['G'] = 0.2
        self.iniDynVar['T'] = 0.1
        self.iniDynVar['P'] = 0.4
        self.iniDynVar['K'] = 0.6
        self.iniDynVar['S'] = 0.7
        
        """
        if 'L' in iniDict.keys():
            L = iniDict['L']
            try:
                assert 0 <= L <= self.params['Cstar'], "L must be between 0 and Cstar"
                try: assert L <= self.params['Cstar'] - self.iniDynVar['A'], "L must be <= Cstar - A"
                except: pass
            except: pass
            self.iniDynVar['L'] = L
        
        if 'A' in iniDict.keys():
            A = iniDict['A']
            try:
                assert 0 <= A <= self.params['Cstar'], "A must be between 0 and Cstar"
                try: assert A <= self.params['Cstar'] - self.iniDynVar['L'], "A must be <= Cstar - L"
                except: pass
            except: pass
            self.iniDynVar['A'] = A
        
        if 'G' in iniDict.keys():
            G = iniDict['G']
            try:
                assert 0 <= G <= self.params['Cstar'], "G must be between 0 and Cstar"
            except: pass
            self.iniDynVar['G'] = G
            
        if 'T' in iniDict.keys():
            T = iniDict['T']
            try:
                assert 0 <= T, "T must be non-negative"
            except: pass
            self.iniDynVar['T'] = T
        
        if 'P' in iniDict.keys():
            P = iniDict['P']
            try:
                assert 0 <= P, "P must be non-negative"
            except: pass
            self.iniDynVar['P'] = P
            
        if 'K' in iniDict.keys():
            K = iniDict['K']
            try:
                assert 0 <= K, "K must be non-negative"
            except: pass
            self.iniDynVar['K'] = K
        
        if 'S' in iniDict.keys():
            S = iniDict['S']
            try:
                assert 0 <= S, "S must be non-negative"
            except: pass
            self.iniDynVar['S'] = S
            
        self.Aini=self.iniDynVar['A']
        self.Pini=self.iniDynVar['P']
        
        
        Xb=self.params['aB']*self.iniDynVar['L']**2.
        Xf=self.params['aF']*self.iniDynVar['G']**2.
        Xr=self.params['aR']*self.iniDynVar['S']**2. 
        X=Xb+Xf+Xr
        
        expP=2./5.
        expK=2./5.
        Z=self.Z(self.iniDynVar['P'], self.iniDynVar['K'], X, expP, expK)
        
        #calculate derived variables
        self.Bini=self.B(Xb, Z)
        self.Fini=self.F(Xf, Z)
        self.Rini=self.R(Xr, Z)

        self.Yini=self.Y(self.Bini, self.Fini, self.Rini)
        self.Wini=self.W(self.Yini, self.Pini, self.iniDynVar['L'])
        
        # 计算得到三个值
        self.ini_state=np.array([self.Aini, self.Wini, self.Pini])
            
    def setParams(self,paramDict):
        self.params={}
        # 访问获取其中的字典中的键
        # 作用：检查值是否合格然后进行输出赋值
        """example
        paramDict = {"name": "John", "age": 30, "city": "New York"}

        keys = paramDict.keys()
        
        for key in keys:
            print(key)
        """

        if 'Cstar' in paramDict.keys():
            Cstar = paramDict['Cstar']
            assert 0 < Cstar, "Cstar must be positive"
            self.params['Cstar']=Cstar
            
        if 'Sigma' in paramDict.keys():
            Sigma = paramDict['Sigma']
            assert 0 < Sigma, "Sigma must be positive"
            self.params['Sigma'] = Sigma 
            
        if 'm' in paramDict.keys():
            m = paramDict['m']
            assert 0 < m, "m must be positive"
            self.params['m'] = m
        
        if 'a0' in paramDict.keys():
            a0 = paramDict['a0']
            assert 0 <= a0, "a0 must be non-negative"
            self.params['a0'] = a0
            
        if 'aT' in paramDict.keys():
            aT = paramDict['aT']
            assert 0 <= aT, "aT must be non-negative"
            self.params['aT'] = aT
        
        if 'l0' in paramDict.keys():
            l0 = paramDict['l0']
            assert 0 <= l0, "l0 must be non-negative"
            self.params['l0'] = l0
            
        if 'lT' in paramDict.keys():
            lT = paramDict['lT']
            assert 0 <= lT, "lT must be non-negative"
            self.params['lT'] = lT
        
        if 'delta' in paramDict.keys():
            delta = paramDict['delta']
            assert 0 < delta, "delta must be positive"
            self.params['delta'] = delta
 
        if 'g' in paramDict.keys():
            g = paramDict['g']
            assert 0 < g, "g must be positive"
            self.params['g'] = g
        
        if 'p' in paramDict.keys():
            p = paramDict['p']
            assert 0 <= p, "p must be non-negative"
            self.params['p'] = p
        
        if 'q0' in paramDict.keys():
            q0 = paramDict['q0']
            assert 0 <= q0, "p must be non-negative"
            self.params['q0'] = q0
        
        if 'qP' in paramDict.keys():
            qP = paramDict['qP']
            assert 0 <= qP, "p must be non-negative"
            self.params['qP'] = qP
        
        if 'Wp' in paramDict.keys():
            Wp = paramDict['Wp']
            assert 0 <= Wp, "p must be non-negative"
            self.params['Wp'] = Wp
            
        if 'yE' in paramDict.keys():
            yE = paramDict['yE']
            assert 0 <= yE, "p must be non-negative"
            self.params['yE'] = yE
            
        if 'wL' in paramDict.keys():
            wL = paramDict['wL']
            assert 0 <= wL, "p must be non-negative"
            self.params['wL'] = wL
            
        if 'eB' in paramDict.keys():
            eB = paramDict['eB']
            assert 0 <= eB, "eB must be non-negative"
            self.params['eB'] = eB
            
        if 'eF' in paramDict.keys():
            eF = paramDict['eF']
            assert 0 <= eF, "eF must be non-negative"
            self.params['eF'] = eF       
            
        if 'aY' in paramDict.keys():
            aY = paramDict['aY']
            assert 0 <= aY, "aY must be non-negative"
            self.params['aY'] = aY

        if 'aB' in paramDict.keys():
            aB = paramDict['aB']
            assert 0 <= aB, "aB must be non-negative"
            self.params['aB'] = aB

        if 'aF' in paramDict.keys():
            aF = paramDict['aF']
            assert 0 <= aF, "aF must be non-negative"
            self.params['aF'] = aF
        
        if 'aR' in paramDict.keys():
            aR = paramDict['aR']
            assert 0 <= aR, "aR must be non-negative"
            self.params['aR'] = aR
        
        if 'i' in paramDict.keys():
            i = paramDict['i']
            assert 0 <= i <= 1., "i must be between 0 and 1"
            self.params['i'] = i
        
        if 'k0' in paramDict.keys():
            k0 = paramDict['k0']
            assert 0 <= k0, "k0 must be non-negative"
            self.params['k0'] = k0
        
        if 'sR' in paramDict.keys():
            sR = paramDict['sR']
            assert 0 <=sR , "sR must be non-negative"
            self.params['sR']=sR
        
        if 'sS' in paramDict.keys():
            sS = paramDict['sS']
            assert 0 <=sS , "sS must be non-negative"
            self.params['sS']=sS
            
        if 'ren_sub' in paramDict.keys():
            ren_sub=paramDict['ren_sub']
        if 'carbon_tax' in paramDict.keys():
            carbon_tax=paramDict['carbon_tax']
        if 'i_DG' in paramDict.keys():
            i_DG=paramDict['i_DG']
        if 'L0' in paramDict.keys():
            L0=paramDict['L0']
            
            
        # Here default parameters before management is used
        self.aR_default=aR
        self.aB_default=aB
        self.aF_default=aF
        self.i_default=i
        
        self.L0=L0
        
        self.ren_sub=ren_sub
        self.carbon_tax=carbon_tax
        self.i_DG=i_DG


    ##########################################################
    # 下面是辅助参数作为计算的内容

    # maritime stock
    
    def M(self, L, A, G):
        return self.params['Cstar']-L-A-G


    #economic production
    def Y(self, B, F, R):
        #return self.params['y'] * ( self.params['eB']*B + self.params['eF']*F )
        # Y = y * E     E = E_B + E_F + R
        return self.params['yE'] * ( self.params['eB']*B + self.params['eF']*F + R )

    #wellbeing
    def W(self, Y, P, L):
        return (1.-self.params['i']) * Y / P + self.params['wL']*L/self.params['Sigma']


    # energy sector
    #auxiliary
    def Z(self, P, K, X, expP=2./5, expK=2./5.):
        return P**expP * K**expK / X**(4./5.)
        
    def B(self, Xb, Z):
        return Xb * Z / self.params['eB']
    def F(self, Xf, Z):
        return Xf * Z / self.params['eF']
    def R(self,Xr, Z):
        return Xr * Z 
    
    def direct_Y(self, L,G,P,K,S):
        """
        总之，这个方法根据输入的参数计算了一系列中间变量，并将它们传递给其他方法进行进一步的计算，最终返回计算得到的值 Y。
        具体的计算过程和方法的实现需要查看 self.Z、self.B、self.F 和 self.R 方法的定义和实现。
        """
        Xb=self.params['aB']*L**2.
        Xf=self.params['aF']*G**2.
        Xr=self.params['aR']*S**2.

        X=Xb+Xf+Xr
        
        expP=2./5.
        expK=2./5.

        # 默认是不在的，可以作为单独计算
#         if 'KproptoP' in self.specs:
# #             expP=4./5.
# #             expK=0.
#             K = P*self.iniDynVar['K']/(self.iniDynVar['P'])
#
#         if 'NproptoP' in self.specs:
#             expP-=1./5.   # gives in combination expP=3./5

        Z=self.Z(P, K, X, expP, expK)

        # 计算经济输出的相关指标内容
        B=self.B(Xb, Z)
        F=self.F(Xf, Z)
        R=self.R(Xr, Z)
        return self.Y(B, F, R)
    
    def direct_W(self,L,G,P,K,S):
        Y=self.direct_Y(L, G, P, K, S)

        return self.W(Y, P, L)
    
    def get_Aini(self,Lini, Gini):
        return (self.params['Cstar']-Lini-Gini)/(1.+self.params['m'])
        
    def get_Tini(self,Aini):
        return Aini/self.params['Sigma']
    
    # # 没有被调用，冗余测试使用
    # # 先删除
    # def prepare_action_set(self, state):
    #     this_state_action_set=[]
    #     L, A, G, T, P, K, S= state
    #     for idx in range(len(self.action_space)):
    #         self._adjust_parameters(idx)

    #         W=self.direct_W(L, G, P, K, S)

    #         if W > self.W_PB:
    #             this_state_action_set.append(idx)

    #     return this_state_action_set


    def _adjust_parameters(self, action_number=0):
        """
        This function is needed to adjust the parameter set for the chosen management option.
        Here the action numbers are really transformed to parameter lists, according to the chosen 
        management option.
        Parameters:
            -action: Number of the action in the actionset.
             Can be transformed into: 'default', 'subsidy' 'carbon tax' 'Nature Protection ' or possible combinations
             虽然只有 3 个选项，但是本质就是排列组合中的内容
             2^3 动作进行组合操作
        """

        # 重写参数的相关部分内容
        # 这里对比其中的 参数调整

        if action_number < len(self.action_space):
            action=self.action_space[action_number]
        else:
            print("ERROR! Management option is not available!" + str (action))
            print(get_linenumber())
            sys.exit(1)

        # 需要重写参数，仿照 ays 中的工作内容
        """
        parameter_list=[(self.beta_LG if action[0] else self.beta  ,
                         self.eps, self.phi, self.rho, 
                         self.sigma_ET if action[1] else self.sigma, 
                         self.tau_A, self.tau_S, self.theta)]
        """
        # subsidy 
        if action[0]:
            self.params['aR']=self.aR_default*(1+self.ren_sub) 
        else:
            self.params['aR']=self.aR_default
        # carbon tax
        if action[1]:
            self.params['aB']=self.aB_default*(1-self.carbon_tax)
            self.params['aF']=self.aF_default*(1-self.carbon_tax)
        else:
            self.params['aB']=self.aB_default
            self.params['aF']=self.aF_default          
        # nature protection
        if action[2]:
            self.Lprot=True
        else:
            self.Lprot=False
    
    """
    This functions are needed to define a final state and to cluster to Green or brown FP
    """
    def _inside_planetary_boundaries(self):
        L,A,G,T,P,K,S = self.state
        Leff=L

        if self.Lprot:
            Leff=max(L-self.L0, 0)

        W=self.direct_W(Leff, G, P, K, S)
        
        is_inside = True
        if A > self.A_PB or W < self.W_PB or P<self.P_PB:
            is_inside = False
            #print("Outside PB!")
        return is_inside
    
        
    """
    This functions are specific to sustainable management parameters, to decide whether we are inside/outside of planetary boundaries
    and whether the game is finished or not!
    """
    def current_state_region_StartPoint(self):
        # 写法是类似于 ays 中的相关部分
        self.state=np.ones(7)
        self._adjust_parameters(0)
        while not self._inside_planetary_boundaries(): 
            #self.state=self.current_state + np.random.uniform(low=-limit_start, high=limit_start, size=3)
            lower_limit=-.1
            upper_limit=.1

            rnd= np.random.uniform(low=lower_limit, high=upper_limit, size=(len(self.state),))

            # 进行基本的拓展内容
            self.state[0] = self.current_state[0] + 1e3*rnd[0]  #L
            self.state[1] = self.current_state[1] + 1e3*rnd[1]  #A
            self.state[2] = self.current_state[2] + 1e3*rnd[2]  #G
            
            self.state[3] = self.get_Tini(self.state[1])        # T
            self.state[4] = self.current_state[4] + 1e9*rnd[4]  # P
            self.state[5] = self.current_state[5] + 1e13*rnd[5] # K 
            self.state[6] = self.current_state[6] + 1e11*rnd[6] # S
            
        return self.state

        
    def  _arrived_at_final_state(self):
        """


        对比区别在于其中 t 时间限制更换了
        """

        # 单独对其中的状态进行获取操作了，和 ays 中是类似的
        L,A,G,T,P,K,S=self.state
        # Attention that we do not break up to early since even at large W it is still possible that A_PB is violated!

        if self.A_PB - A > 0 and self.direct_W(L, G, P, K, S) > 2.e6 and P > 1e10 and self.t>400:
            return True
        else:
            return False

        
    def _good_final_state(self):
        L,A,G,T,P,K,S=self.state
        # Good Final State. TODO find a reasonable explanation for this values (maybe something like carbon budget...)!
        if self.A_PB - A > 60 and self.direct_W(L, G, P, K, S) > 2.8e6 and P > 1e10  :
            #print('Success!')
            return True
        else:
            return False
    
    def _which_final_state(self):
        l,a,g,t,p,k,s=self.state

        if self._inside_planetary_boundaries():
            #print("ARRIVED AT GREEN FINAL STATE WITHOUT VIOLATING PB!")
            return Basins.GREEN_FP
        else:
            return Basins.OUT_PB
    
    def get_plot_state_list(self):
        trafo_state=compactification(self.state, self.current_state)

        return trafo_state.tolist()
    
    def observed_states(self):
        return self.dimensions
    
