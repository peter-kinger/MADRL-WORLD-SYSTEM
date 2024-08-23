function plot_mices(out)
% 只绘制温度

% 定义时间范围
time = [1800:2300];

% 温度绘制处理
mod_tas = interp1(out.clim.time, out.clim.tem, time); % 模型温度
mod_tas = mod_tas - mean(mod_tas(1:40)); % 去除前 40 年的均值

% CO2 浓度处理
mod_co2 = interp1(out.clim.time, out.clim.ppm, time); % 模型 CO2



% 绘图
figure(1)
clf
subplot(2,2,1)
% p1 = plot(time, obs_tas, 'k-'); % 绘制观测温度
% hold on
p2 = plot(time, mod_tas, 'r-'); % 绘制模型温度
% legend([p1,p2], 'CESM1-CAM5/RCP2.6', 'MiCES', 'location', 'SouthEast')
legend('MiCES')
xlabel('Year')
ylabel('Temperature/PI (K)')
xlim([1950, 2100])
% plot([1850, 2500], [2, 2], 'k--') % 添加参考线

subplot(2,2,2)
% p1 = plot(time, obs_co2, 'k-'); % 绘制观测 CO2
% hold on
p2 = plot(time, mod_co2, 'r-'); % 绘制模型 CO2
% legend([p1,p2], 'MAGICC6', 'MiCES')
legend('MiCES')
xlabel('Year')
ylabel('CO2 Concentration (ppmv)')
xlim([1950, 2300])

