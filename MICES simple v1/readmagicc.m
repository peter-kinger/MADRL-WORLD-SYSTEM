function [year, data, unit] = readmagicc(rcpname, species, type)
    % READMAGICC - 读取RCP情景的排放或浓度数据
    % 排放或浓度数据来自MAGICC6模型，时间范围为1750-2500年。
    %
    % 语法:  [year, data, unit] = readmagicc(rcpname, species, type)
    %
    % 输入:
    %    rcpname - RCP情景的名称，字符串类型。选项包括：RCP3PD, RCP45, RCP6, RCP85, 20THCENTURY
    %    species - 物种名称，单个字符串或字符串元胞数组。
    %              例如: 'CH4', 'N2O', 'HFC134a'
    %    type    - 'E' 表示排放数据，'C' 表示浓度数据。
    %
    % 输出:
    %    year    - 年份的列数组
    %    data    - 对应年份的排放或浓度数据列数组
    %    unit    - 数据单位的字符串元胞数组
    %
    % 示例 1: 
    %    [year, emiss, unit] = readmagicc('RCP6', {'CH4', 'N2O'}, 'E');
    %    subplot(2,1,1)
    %    plot(year, emiss(:,1)); ylabel(strcat('CH4 emission, ', unit{1}))
    %    subplot(2,1,2)
    %    plot(year, emiss(:,2)); ylabel(strcat('N2O emission, ', unit{2}))
    %
    % 示例 2: 
    %    [year, conc, unit] = readmagicc('RCP6', 'HFC134a', 'C');
    %    plot(year, conc); ylabel(strcat('HFC134a concentration, ', unit{1}))
    %
    % 其他文件要求: 无
    % 子函数: 无
    % MAT文件要求: 无
    %
    % 需要包含MAGICC6排放和中年浓度的ASCII格式文件的'input/'子目录，
    % 可从以下网址下载:
    % http://www.pik-potsdam.de/~mmalte/rcps/index.htm#Download
    %
    % 作者: Christopher Holmes
    % 地球系统科学系
    % 加州大学尔湾分校
    % 邮箱: cdholmes@uci.edu
    % 网站: http://www.ess.uci.edu/~cdholmes/
    % 2011年11月; 最后修订: 2012年2月23日

    %------------- 开始代码 --------------

    % 根据输入类型选择排放或浓度数据
    switch lower(type)
        case 'e'
            % 文件名
            file = 'input/%s_EMISSIONS.DAT';
            % 头文件行数
            headerlines = 38;
            if strcmp(rcpname, '20THCENTURY')
                headerlines = 37;
            end
        case 'c'
            % 文件名
            file = 'input/%s_MIDYEAR_CONCENTRATIONS.DAT';
            % 头文件行数
            headerlines = 39;
        otherwise
            error('未知类型: %s', type);
    end

    % 修改RCP名称以匹配文件名
    if strcmp(rcpname, 'RCP60')
        filename = sprintf(file, 'RCP6');
    elseif strcmp(rcpname, 'RCP26')
        filename = sprintf(file, 'RCP3PD');
    else
        filename = sprintf(file, rcpname);
    end

    % 文件是以空格分隔的
    DELIMITER = ' ';

    % 导入文件的ASCII格式数据
    newData = importdata(filename, DELIMITER, headerlines);

    % 获取数据文件中的单位和变量名
    t = textread(filename, '%s', 'delimiter', '\n');
    units = strread(t{headerlines - 1}, '%s', 'delimiter', ' ');
    names = newData.colheaders;

    % 找到年份列
    j = strcmp('YEARS', names);
    year = newData.data(:, j);

    % 将物种名称转换为元胞数组（如果还不是）
    species = cellstr(species);

    % 请求的物种数量
    D = length(species);

    % 遍历每个请求的物种
    for i = 1:D
        % 找到当前物种的列
        j = strcmp(strtrim(char(species{i})), names);
        % 保存当前物种的值和单位
        data(:, i) = newData.data(:, j);
        unit{i} = strtrim(char(units{j}));
    end
end
