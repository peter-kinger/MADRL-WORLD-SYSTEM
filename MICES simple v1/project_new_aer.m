function out = project_new_aer(emis, in, aparam)
    % PROJECT_NEW_AER - 计算新的气溶胶参数
    %
    % 输入:
    %   emis   - 结构体，包含不同物种的排放数据
    %   in     - 结构体，包含其他输入参数
    %   aparam - 结构体，包含气溶胶参数
    %
    % 输出:
    %   out    - 结构体，包含计算出的气溶胶参数

    % 提取时间和排放数据
    tt = in.CH4.tt; % 时间序列
    time = emis.CFC_11.time; % 排放数据的时间序列
    SO2 = emis.SOx.val; % SO2排放数据

    % 插值计算指定时间点tt的气溶胶排放量
    aer_t = interp1(emis.SOx.time, emis.SOx.val, tt);

    % 插值计算1990年的气溶胶排放量
    aer_1990 = interp1(emis.SOx.time, emis.SOx.val, 1990);

    % 计算新的气溶胶强迫
    out.aer_f = aer_t / aer_1990 * aparam.f_1990;
end
