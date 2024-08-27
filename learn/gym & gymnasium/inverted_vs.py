import gym
from gym import spaces
import numpy as np
import pygame
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class InvertedPendulumEnv(gym.Env):
    def __init__(self):
        super(InvertedPendulumEnv, self).__init__()
        
        # 定义动作空间和状态空间
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        
        # 系统参数
        self.g = 9.8
        self.l = 1.0
        self.b = 0.1
        self.m = 1.0
        self.dt = 0.01  # 时间步长
        
        # 初始化状态
        self.state = None
        self.reset()

    def reset(self):
        # 初始化角度和角速度
        self.state = np.random.uniform(low=-0.1, high=0.1, size=(2,))
        return self.state

    def step(self, action):
        theta, omega = self.state
        u = action[0]
        
        # 计算微分方程的离散化
        dtheta = omega
        domega = (self.g / self.l) * np.sin(theta) - (self.b / (self.m * self.l**2)) * omega + (u / (self.m * self.l**2))
        
        # 更新状态
        theta += dtheta * self.dt
        omega += domega * self.dt
        
        self.state = np.array([theta, omega])
        
        # 计算奖励（例如，保持摆直立）
        reward = -np.abs(theta)
        
        # 判断是否结束
        done = np.abs(theta) > np.pi / 2
        
        return self.state, reward, done, {}

    def render(self, screen, x_offset=0, mode='human'):
        screen.fill((255, 255, 255), (x_offset, 0, 400, 600))  # 清屏指定区域
        
        # 坐标变换
        theta, _ = self.state
        x = x_offset + 200 + self.l * np.sin(theta) * 100
        y = 300 - self.l * np.cos(theta) * 100
        
        # 画摆杆
        pygame.draw.line(screen, (0, 0, 0), (x_offset + 200, 300), (x, y), 5)
        # 画摆锤
        pygame.draw.circle(screen, (0, 0, 255), (int(x), int(y)), 10)
        
        pygame.display.flip()



def simulate_no_control(env, screen):
    state = env.reset()
    for _ in range(200):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # 无控制输入
        action = np.array([0.0])
        state, reward, done, _ = env.step(action)
        env.render(screen, x_offset=0)
        if done:
            break


def simulate_with_rl_control(env, model, screen):
    state = env.reset()
    for _ in range(200):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        action, _states = model.predict(state)
        state, reward, done, _ = env.step(action)
        env.render(screen, x_offset=400)
        if done:
            break



def main():
    # 初始化 Pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Inverted Pendulum - No Control vs RL Control")
    clock = pygame.time.Clock()

    # 创建环境
    env = InvertedPendulumEnv()
    check_env(env)

    # 使用 PPO 算法训练
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)

    # 保存模型
    model.save("ppo_inverted_pendulum")

    # 加载模型
    model = PPO.load("ppo_inverted_pendulum")

    # 同时展示两种情况
    while True:
        simulate_no_control(env, screen)
        simulate_with_rl_control(env, model, screen)
        clock.tick(60)

if __name__ == "__main__":
    main()
