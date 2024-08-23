

f_nonco2_struct = load("ode_matlab\f_nonco2_rcp26.mat");
% gtc_in = load("ode_matlab\gtc_in_rcp26.mat");

f_nonco2 = f_nonco2_struct.f_nonco2;

tt = linspace(1800,2500,701);

% aa = 2;

% 微分方程函数，状态导数
% function xdot = SunFun(t,x, f_nonco2, gtc_in )


% 插值当前时间步长的排放数据和非CO2辐射强迫
% emis_t = interp1(tt, emis, t); % 当前时间步长的碳排放量

t = 1850;

fcg_t = interp1(tt, f_nonco2, t); % 当前时间步长的非CO2辐射强迫
