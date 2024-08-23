function out = climod_simple(emis, param, tt)
% SIMPLIFIED_CLIMOD 简化的气候模型模拟，使用5维气候方程。
% 输入参数:
%   emis - 排放数据的结构体
%   param - 参数数据的结构体
%   tt - 时间序列
% 输出:
%   out - 模拟结果的结构体

% 投影和计算辐射强迫
out.CH4 = project_new(emis.CH4, param.CH4, 1, tt);
out.N2O = project_new(emis.N2O, param.N2O, 1, tt);
out.minor = project_minor(emis, out);
out.aer = project_new_aer(emis, out, param.aer);

% 初始化气候模型参数
lambda = 3.8 / param.clim.sens; % 敏感度参数
kappa = param.clim.kappa; % 海洋热量吸收参数
kappa2 = param.clim.kappa2; % 海洋热量吸收参数
kdeep = param.clim.kdeep; % 海洋热量吸收参数

co2c0 = param.clim.ppm_1850; % 初始大气CO2浓度
tem0 = 0; % 初始温度
alpha = 1e6 / 1.8e20 / 12 * 1e15; % ppm到Pg的转换
Ca0 = co2c0 / alpha; % 初始大气碳量（Pg）

% 初始化陆地和海洋碳循环参数
cinl(1) = param.clim.lnd_init;
gamma_l = param.clim.gamma_l; % Pg/K
beta_l = param.clim.beta_l; % Pg/ppm
gamma_o = param.clim.gamma_o; % Pg/K
beta_o = param.clim.beta_o; % Pg/ppm
cino(1) = param.clim.ocn_init;
cinod(1) = param.clim.docn_init;
beta_od(1) = param.clim.beta_od;

% 插值排放数据
emis_fossil = interp1(emis.FossilCO2.time, emis.FossilCO2.val, tt); % 化石燃料排放数据
emis_lu = interp1(emis.OtherCO2.time, emis.OtherCO2.val, tt); % 土地利用排放数据

% 计算非CO2辐射强迫和碳排放总量
for i = 1:numel(tt)
    f_nonco2(i) = out.CH4.RFrcp(i) + out.N2O.RFrcp(i) + out.aer.aer_f(i) + out.minor.rad(i); % 非CO2辐射强迫总量
    gtc_in(i) = emis_fossil(i) + emis_lu(i); % 人为碳排放总量（Pg）
end

% 使用ODE求解器模拟温度和碳循环的演变
[t, C] = ode45(@(t, C) tsolve_simplified(C, Ca0, kappa, kappa2, kdeep, lambda, gamma_l, gamma_o, alpha, beta_l, beta_o, beta_od, f_nonco2, gtc_in, tt, t), tt, [tem0, Ca0, cino(1), cinod(1), tem0]);

% 存储结果
out.clim.f_co2 = 6.3 * log(C(:, 2) / Ca0);
out.clim.f_nonco2 = f_nonco2;
out.clim.gtc_in = gtc_in;
out.clim.time = t;
out.clim.cina = C(:, 2);
out.clim.tem = C(:, 1) - mean(C(t < 1900, 1));
out.clim.cino = C(:, 3);
out.clim.cinod = C(:, 4);
out.clim.ppm = out.clim.cina * alpha;

% 累积碳排放总量和陆地碳储量计算
cumc(1) = 0;
for i = 2:numel(tt)
    dt = out.clim.tem(i) - out.clim.tem(i - 1);
    dppm = out.clim.ppm(i) - out.clim.ppm(i - 1);
    absl(i) = beta_l * dppm + gamma_l * dt;
    cinl(i) = cinl(i - 1) + absl(i);
    f_co2(i) = 6.3 * log(out.clim.ppm(i) / co2c0);
    cumc(i) = cumc(i - 1) + gtc_in(i);
end

% 存储累积结果
out.clim.f_co2 = f_co2;
out.clim.cinl = cinl;
out.clim.absl = absl;
out.clim.cumc = cumc;
end

function dCdt = tsolve_simplified(C, Ca0, kappa, kappa2, kdeep, lambda, gamma_l, gamma_o, alpha, beta_l, beta_o, beta_od, f_nonco2, gtc_in, tt, t)
% TSOLVE_SIMPLIFIED 简化的气候方程
% 输入参数:
%   C - 当前状态
%   Ca0 - 初始大气碳量
%   kappa, kappa2, kdeep - 海洋热量吸收参数
%   lambda - 敏感度参数
%   gamma_l, gamma_o - 碳循环参数
%   alpha - ppm到Pg的转换
%   beta_l, beta_o, beta_od - 碳循环参数
%   f_nonco2 - 非CO2辐射强迫
%   gtc_in - 碳排放总量
%   tt - 时间序列
%   t - 当前时间
% 输出:
%   dCdt - 状态的变化率

% 插值辐射强迫和碳排放
f_nonco2_interp = interp1(tt, f_nonco2, t);
gtc_in_interp = interp1(tt, gtc_in, t);

% 温度变化率
dTdt = (f_nonco2_interp + 6.3 * log(C(2) / Ca0) - lambda * C(1)) / (kappa + kappa2 * C(1) + kdeep * C(1));

% 大气碳量变化率
dCadt = gtc_in_interp - beta_l * (C(2) - Ca0) - gamma_l * C(1);

% 浅海碳量变化率
dCino = beta_o * (C(2) - Ca0) - gamma_o * C(1);

% 深海碳量变化率
dCinod = beta_od * (C(2) - Ca0);

dCdt = [dTdt; dCadt; dCino; dCinod; dTdt];
end
