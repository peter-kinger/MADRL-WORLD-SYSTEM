"""
代码来源于 mices_ori -> mices_simple -> mices_rec -> mices_rec_gpt 

"""

import numpy as np
from scipy.integrate import odeint
import scipy.io

def depart_test():
    # 开始计算参数的部分
    # 初始化气候模型参数
    tt = np.linspace(1800, 2500, 701)
    
    co2c0 = 290  # 初始大气CO2浓度
    tem0 = 0  # 初始温度

    # 计算Ca0
    mat = 1.8e20  # 大气中的摩尔数
    alpha = 1e6 / mat / 12 * 1e15  # ppm到Pg的转换
    Ca0 = co2c0 / alpha  # 初始大气碳量（Pg）
    
    # kappa 计算
    kappa = 10  # 海洋热量吸收参数
    kappa2 = 12.6  # 第二海洋热量吸收参数
    kdeep = 0.15  # 深海热量吸收参数
    
    # lambda 计算
    lambda_ = 3.8 / 4.8  # 敏感度参数
    
    # gamma_l 计算
    gamma_l = -95  # 陆地碳气候反馈（Pg/K）
    gamma_o = -60  # 海洋碳气候反馈（Pg/K）
    
    # rho 计算
    cino_0 = 600  # 初始浅海碳量
    rho = 290 / 600  # 浅海碳量与ppm的转换因子
    rho2 = 290 / 100000  # 深海碳量与ppm的转换因子

    cinod_0 = 100000  # 初始深海碳量
    
    beta_l = 3.5  # 陆地碳浓度反馈（Pg/ppm）
    beta_o = 2.4  # 海洋碳浓度反馈（Pg/ppm）
    beta_od = 0.5  # 深海碳浓度反馈（Pg/ppm）
    cpl = 1  # 耦合参数
    c_amp = 1.1  # 温度放大参数

    # 重新运行加载已经处理好的 f_nonco2 和 gtc_in
    f_nonco2_struct = scipy.io.loadmat("test/f_nonco2_rcp26.mat")
    gtc_in_struct = scipy.io.loadmat("test/gtc_in_rcp26.mat")

    f_nonco2 = f_nonco2_struct['f_nonco2'].flatten()
    gtc_in = gtc_in_struct['gtc_in'].flatten()

    # 确保数据类型为浮点数
    f_nonco2 = f_nonco2.astype(float)
    gtc_in = gtc_in.astype(float)

    # 使用ODE求解器模拟温度和碳循环的演变
    initial_conditions = [tem0, Ca0, cino_0, cinod_0, tem0]
    result = odeint(ode_solver, initial_conditions, tt, args=(Ca0, kappa, kappa2, kdeep, lambda_, gamma_l, gamma_o, alpha, rho, rho2, beta_l, beta_o, beta_od, f_nonco2, gtc_in, tt, cpl, c_amp))

    C_result = result

    # 计算基线温度
    out = {}
    out['clim'] = {}
    out['clim']['time'] = tt
    tim_pi = (tt > 1850) & (tt < 1900)
    out['clim']['tem_atm_lx'] = result[:, 0]  # 大气温度
    out['clim']['tem'] = result[:, 0] - np.mean(result[tim_pi, 0])  # 温度变化

    out['clim']['cina'] = result[:, 1]  # 大气碳量
    out['clim']['ppm'] = out['clim']['cina'] * alpha  # 大气碳浓度（ppm）

    out['clim']['cino'] = result[:, 2]  # 浅海碳量
    out['clim']['cinod'] = result[:, 3]  # 深海碳量
    out['clim']['tem_ocean'] = result[:, 4]  # 海洋温度

    return out, C_result

# 补充其中的函数部分
def ode_solver(C, t, Ca0, kappa, kappa2, kdeep, lambda_, gamma_l, gamma_o, alpha, rho, rho2, beta_l, beta_o, beta_od, fcg, emis, tt, cpl, c_amp):
    # 初始化输出向量
    fv = np.zeros(5)  # 1st元素是温度，2nd元素是大气碳量

    # 插值当前时间步长的排放数据和非CO2辐射强迫
    emis_t = np.interp(t, tt, emis)  # 当前时间步长的碳排放量
    fcg_t = np.interp(t, tt, fcg)  # 当前时间步长的非CO2辐射强迫

    # 计算温度的导数（基于能量平衡方程）
    fv[0] = 1/kappa * (6.3 * np.log(C[1] / Ca0) + fcg_t - lambda_ * C[0]) - kdeep * (C[0] - C[4])
    fv[4] = 1/kappa2 * kdeep * (C[0] - C[4])

    # 计算大气碳浓度的变化率
    dppm = (alpha * C[1] - rho * C[2]) / 100  # 利用开始的状态变量进行相关的计算
    dppm2 = (rho * C[2] - rho2 * C[3]) / 100  # 对应里面的相关计算部分

    # 大气碳量的导数
    fv[1] = (emis_t - (gamma_l + gamma_o) * fv[0] * (1 + C[0] * c_amp)) / (1 + alpha * beta_l) - beta_o * dppm

    # 浅海碳量的导数
    fv[2] = beta_o * dppm + gamma_o * fv[0] - beta_od * dppm2

    # 深海碳量的导数
    fv[3] = beta_od * dppm2  # 类似于方程中的前一部分操作

    return fv

# 运行函数并打印结果
out, C_result = depart_test()
print(out)

# 保存其中的 Out 部分 为 xlsx 文件
import pandas as pd
df = pd.DataFrame(out['clim'], columns=["time", "tem_atm_lx", "tem", "cina", "ppm", "cino", "cinod", "tem_ocean"])
df.to_excel("test\output_rec_mices_1.xlsx", index=False)


# 绘制运行的 out 结果，分多个图进行展示，每个图中只有一个变量
import matplotlib.pyplot as plt

fig, ax = plt.subplots(4, 1, figsize=(10, 10))
ax[0].plot(out['clim']['time'], out['clim']['tem_atm_lx'])
ax[0].set_title("tem_atm_lx")
ax[1].plot(out['clim']['time'], out['clim']['cina'])
ax[1].set_title("cina")
ax[2].plot(out['clim']['time'], out['clim']['cino'])
ax[2].set_title("cino")
ax[3].plot(out['clim']['time'], out['clim']['tem_ocean'])
ax[3].set_title("tem_ocean")

plt.show()