

% 输入：只有两个，通过其中的参数获取来操作（其中的参数是根据年份来进行确定的）
out, C_result = depart_test();

plot_mices(out)


function [out, C_result] = depart_test()
 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 开始计算参数的部分
    % 初始化气候模型参数
    tt = linspace(1800,2500,701);
    
    co2c0 =  290 ;% param.clim.ppm_1850; % 初始大气CO2浓度
    tem0 = 0; % 初始温度

    % 计算Ca0
    mat = 1.8e20; % 大气中的摩尔数
    alpha = 1e6 / mat / 12 * 1e15; % ppm到Pg的转换
    Ca0 = co2c0 / alpha; % 初始大气碳量（Pg）
    % kappa 计算
    kappa = 10 ;% param.clim.kappa; % 海洋热量吸收参数
    kappa2 = 12.6;% param.clim.kappa2; % 第二海洋热量吸收参数
    kdeep = 0.15; % param.clim.kdeep; % 深海热量吸收参数
    % lambda 计算
    lambda = 3.8 /4.8; % lambda = 3.8 / param.clim.sens; % 敏感度参数
    % gamma_l 计算
    gamma_l = -95; % gamma_l = param.clim.gamma_l; % 陆地碳气候反馈（Pg/K）
    gamma_o = -60; % gamma_o = param.clim.gamma_o; % 海洋碳气候反馈（Pg/K）
    % rho 计算
    cino(1) = 600; %param.clim.ocn_init;
    % oco2c(1) = param.clim.ppm_1850;
    % rho = oco2c(1) / cino(1);
    rho = 290 / 600; 
    rho2 = 290 / 100000;

    cinod(1) = 100000; %param.clim.docn_init; 
    % odco2c(1) = param.clim.ppm_1850;
    % rho2 = odco2c(1) / cinod(1);
    beta_l = 3.5;% beta_l = param.clim.beta_l; % 陆地碳浓度反馈（Pg/ppm）
    beta_o = 2.4;% beta_o = param.clim.beta_o; % 海洋碳浓度反馈（Pg/ppm）
    beta_od = 0.5; % param.clim.beta_od;
    cpl = 1; % param.clim.cpl; 
    c_amp = 1.1; % param.clim.c_amp; % 碳吸收的温度放大参数

    % 重新运行加载已经处理好的 f_nonco2 和 gtc_in
    f_nonco2_struct = load("f_nonco2_input.mat");
    gtc_in_struct = load("gtc_in_input.mat");

    f_nonco2 = f_nonco2_struct.f_nonco2;
    gtc_in = gtc_in_struct.gtc_in;

    % 尤其关注这里的 f_nonco2 数值大小
    % 使用ODE求解器模拟温度和碳循环的演变
    [t, C] = ode45(@(t, C) ode_solver(C, Ca0, kappa, kappa2, kdeep, lambda, gamma_l, gamma_o, ...
        alpha, rho, rho2, beta_l, beta_o, beta_od, f_nonco2, gtc_in, tt, t, cpl, c_amp), tt, [tem0, Ca0, cino(1), cinod(1), tem0]);

    C_result = C;
    % 计算基线温度
    out.clim.time = t;
    tim_pi = t > 1850 & t < 1900;
    out.clim.tem_atm_lx = C(:, 1); % 大气温度
    
    out.clim.tem = C(:, 1) - mean(C(tim_pi, 1)); % 温度变化

    out.clim.cina = C(:, 2); % 大气碳量
    out.clim.ppm = out.clim.cina * alpha; % 大气碳浓度（ppm）

    out.clim.cino = C(:, 3); % 浅海碳量
    out.clim.cinod = C(:, 4); % 深海碳量
    out.clim.tem_ocean = C(:, 5); % 海洋温度
end 

% 补充其中的函数部分
function fv = ode_solver(C, Ca0, kappa, kappa2, kdeep, lambda, gamma_l, gamma_o, alpha, rho, rho2, beta_l, beta_o, beta_od, fcg, emis, tt, t, cpl, c_amp)
% function fv = tsolve2()
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


