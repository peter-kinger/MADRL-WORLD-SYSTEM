
% emis{1} = load("ode_matlab\emis_rcp26.mat");
% out = climod_ode2(emis{1},paramopt,tt,'all'); 
% 


mat = 1.8e20; % 大气中的摩尔数
alpha = 1e6 / mat / 12 * 1e15; % ppm到Pg的转换

co2c0 = 290; % co2c0 = param.clim.ppm_1850; % 初始大气CO2浓度
Ca0 = co2c0 / alpha; % 初始大气碳量（Pg）


kappa = 10;
kappa2 = 12.6000000000000;
kdeep = 0.150000000000000;

clim_sens = 4.8;
lambda = 3.8 / clim_sens;

% lambda = 3.8 / 

gamma_l = -95;
gamma_o = -60;

oco2c_1 = 290; % oco2c(1) = param.clim.ppm_1850;
cino_1 = 600;

rho = oco2c_1 / cino_1 ;
rho2 = 0.00290000000000000; 
beta_l = 3.50000000000000;
beta_o = 2.40000000000000;
beta_od = 0.500000000000000;

f_nonco2_struct = load("f_nonco2_rcp26.mat");
gtc_in_struct = load("gtc_in_rcp26.mat");

f_nonco2 = f_nonco2_struct.f_nonco2;
gtc_in = gtc_in_struct.gtc_in;

tt = linspace(1800,2500,701);

cpl = 1;
c_amp = 1.10000000000000;


tem0 = 0;
Ca0 = 626.4;
cino(1) = 600;
cinod(1) = 10000;
% tem0 = 0;



% 使用ODE求解器模拟温度和碳循环的演变
% [t, C] = ode45(@(t, C) tsolve2(C, Ca0, kappa, kappa2, kdeep, lambda, gamma_l, gamma_o, ...
%     alpha, rho, rho2, beta_l, beta_o, beta_od, f_nonco2, gtc_in, tt, t, cpl, c_amp), ...
%     tt, ...
%     [tem0, Ca0, cino(1), cinod(1), tem0]);

[t, C] = ode45(@(t, C) tsolve2(C, Ca0, kappa, kappa2, kdeep, lambda, gamma_l, gamma_o, ...
    alpha, rho, rho2, beta_l, beta_o, beta_od, f_nonco2, gtc_in, tt, t, cpl, c_amp), tt, [tem0, Ca0, cino(1), cinod(1), tem0]);



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
