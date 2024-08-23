clear;
clc;

% 程序主函数代码如下：
t0 = 1800;
tfinal = 2300;
X0 = 1;

f_nonco2_strcut = load("ode_matlab\f_nonco2_rcp26.mat");
gtc_in_strcut = load("ode_matlab\gtc_in_rcp26.mat");

tt = linspace(1800,2500,701);

aa = 2;

f_nonco2 = f_nonco2_strcut.f_nonco2;
gtc_in = gtc_in_strcut.gtc_in;

[t, Xt] = ode45(@(t,y) SunFun(t, y, f_nonco2, tt), [t0 tfinal], X0);

% 绘制结果图
plot(t,Xt)
grid



% 微分方程函数，状态导数
function xdot = SunFun(t,x, f_nonco2, gtc_in, tt )


% 插值当前时间步长的排放数据和非CO2辐射强迫
emis_t = interp1(tt, f_nonco2, t); % 当前时间步长的碳排放量
fcg_t = interp1(tt, gtc_in, t); % 当前时间步长的非CO2辐射强迫


% 导数关系式
% xdot = aa * x ;
xdot = aa * x + fcg_t + emis_t;

end

