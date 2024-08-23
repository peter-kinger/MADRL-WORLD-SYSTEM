% 保持原有的初始化部分不变
Ca0 = 626.4;
kappa = 10;
kappa2 = 12.6;
kdeep = 0.15;
lambda = 0.791666666666667;
gamma_l = -95;
gamma_o = -60;
alpha = 0.462962962962963;
rho = 0.462962962962963;
rho2 = 0.0029; 
beta_l = 3.5;
beta_o = 2.4;
beta_od = 0.5;

f_nonco2 = load("ode_matlab\f_nonco2_rcp26.mat");
gtc_in = load("ode_matlab\gtc_in_rcp26.mat");

tt = linspace(1800,2500,701);

cpl = 1;
c_amp = 1.1;

tem0 = 0;
Ca0 = 626.4;
cino = zeros(1, length(tt)); % 确保整个数组初始化
cinod = zeros(1, length(tt)); % 确保整个数组初始化
cino(1) = 600;
cinod(1) = 10000;

% 使用ODE求解器模拟温度和碳循环的演变
[t, C] = ode45(@(t, C) tsolve2(C, Ca0, kappa, kappa2, kdeep, lambda, gamma_l, gamma_o, ...
alpha, rho, rho2, beta_l, beta_o, beta_od, f_nonco2, gtc_in, tt, t, cpl, c_amp), ...
tt, [tem0, Ca0, cino(1), cinod(1), tem0]);

function fv = tsolve2(C, Ca0, kappa, kappa2, kdeep, lambda, gamma_l, gamma_o, alpha, rho, rho2, beta_l, beta_o, beta_od, fcg, emis, tt, t, cpl, c_amp)
    % TSOLVE2 求解气候方程，用于计算温度和大气碳浓度的变化率
    % 输入参数见上文
    % 输出: fv - 状态变化率向量
    
    fv = zeros(5,1); % 初始化输出向量
    
    % 插值当前时间步长的排放数据和非CO2辐射强迫
    emis_t = interp1(tt, emis, t); % 当前时间步长的碳排放量
    fcg_t = interp1(tt, fcg, t); % 当前时间步长的非CO2辐射强迫

    % 计算温度的导数
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
end
