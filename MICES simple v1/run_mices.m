%Read scenario emissions
% 读取当前场景下的参数对应
emis{1}=read_emissions('RCP26');

% 该代码中不包s含校正的这部分，使用的是默认的参数
% conc{1}=read_concentrations('RCP26');
% temp{1}=read_temp_cesm('RCP26');

param=read_all_parameter_test(); % 每次测试参数的部分，考虑其中的最优参数部分

%define output times
tt = linspace(1800,2500,701);

% %%%%%%%%%%%
% 如果 opt 为1，则进行参数优化，包括优化甲烷 (opt_methane)、氧化亚氮 (opt_n2o) 和气候 (opt_clim) 参数。
% 如果 opt 为0，则直接使用读取的参数 param。

% 思考对应的参数是多大，查看文献进行相关对应
paramopt=param;


% 输入的相关参数部分
% 前面就是优化后的参数来进行输入操作
out = climod_ode2(emis{1},paramopt,tt,'all'); % 关键的部分,就是输入排放参数部分

plot_mices(out)

