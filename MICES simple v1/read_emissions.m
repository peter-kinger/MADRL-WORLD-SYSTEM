function emis = read_emissions(scen)
    % READ_EMISSIONS 函数用于从MAGICC文本文件中读取排放数据
    %
    % 输入:
    %   scen - 字符串，表示排放数据的情景标识符
    %
    % 输出:
    %   emis - 结构体，包含各种物种的排放数据
    %
    % 该函数从指定的MAGICC文本文件中读取多个物种的排放数据，根据情景标识符确定文件。

    % 定义物种列表
    species = {'FossilCO2', 'OtherCO2', 'CH4', 'N2O', 'SOx', 'CO', 'NMVOC', 'NOx', ...
               'BC', 'OC', 'NH3', 'CF4', 'C2F6', 'C6F14', 'HFC23', 'HFC32', ...
               'HFC43_10', 'HFC125', 'HFC134a', 'HFC143a', 'HFC227ea', 'HFC245fa', ...
               'SF6', 'CFC_11', 'CFC_12', 'CFC_113', 'CFC_114', 'CFC_115', ...
               'CARB_TET', 'MCF', 'HCFC_22', 'HCFC_141B', 'HCFC_142B', ...
               'HALON1211', 'HALON1202', 'HALON1301', 'HALON2402', 'CH3BR', 'CH3CL'};
    
    % 初始化排放数据结构体
    emis = struct();
    
    % 遍历每个物种并读取排放数据
    for i = 1:numel(species)
        % 读取当前物种的排放数据
        [emis.(species{i}).time, emis.(species{i}).val, emis.(species{i}).unit] = ...
            readmagicc(scen, species{i}, 'E');
    end
    
    % 显示成功读取数据的提示信息
    fprintf('成功读取情景 %s 的排放数据\n', scen);
end
