function fv = tsolve2(C, Ca0, kappa, kappa2, kdeep, lambda, gamma_l, gamma_o, alpha, rho, rho2, beta_l, beta_o, beta_od, fcg, emis, tt, t, cpl, c_amp)
% TSOLVE2 求解气候方程，用于计算温度和大气碳浓度的变化率
% 输入参数:
%   C - 当前状态向量
%   Ca0 - 初始大气CO2浓度
%   kappa, kappa2, kdeep - 海洋热量吸收参数
%   lambda - 温度敏感度参数
%   gamma_l, gamma_o - 碳循环参数
%   alpha - ppm到Pg的转换因子
%   rho, rho2 - 浅海和深海碳量与ppm的转换因子
%   beta_l, beta_o, beta_od - 碳循环参数
%   fcg - 非CO2辐射强迫
%   emis - 碳排放量
%   tt - 时间序列
%   t - 当前时间
%   cpl - 耦合参数
%   c_amp - 温度放大参数
% 输出:
%   fv - 状态变化率向量

% 初始化输出向量
fv = zeros(5,1); % 1st元素是温度，2nd元素是大气碳量

% 插值当前时间步长的排放数据和非CO2辐射强迫
emis_t = interp1(tt, emis, t); % 当前时间步长的碳排放量
fcg_t = interp1(tt, fcg, t); % 当前时间步长的非CO2辐射强迫

% 计算温度的导数（基于能量平衡方程）

fv(1) = 1/kappa * (6.3 * log(C(2) / Ca0) + fcg_t - lambda * C(1)) - kdeep * (C(1) - C(5));
fv(5) = 1/kappa2 * kdeep * (C(1) - C(5));

% 计算大气碳浓度的变化率
dppm = (alpha * C(2) - rho * C(3)) / 100; % 利用开始的状态变量进行相关的计算
dppm2 = (rho * C(3) - rho2 * C(4)) / 100; % 对应里面的相关计算部分

% 大气碳量的导数
fv(2) = (emis_t - (gamma_l + gamma_o) * fv(1) * (1 + C(1) * c_amp)) / (1 + alpha * beta_l) - beta_o * dppm;

% 浅海碳量的导数
fv(3) = beta_o * dppm + gamma_o * fv(1) - beta_od * dppm2;

% 深海碳量的导数
fv(4) = beta_od * dppm2; % 类似于方程中的前一部分操作
t
end

