function param = read_all_parameter_test()
    % READ_ALL_PARAMETERS_TEST 函数用于读取多个气候和气溶胶参数
    %
    % 输出:
    %   param - 结构体，包含各种气候和气溶胶参数

    % 定义气溶胶参数
    param.aer.f_1990 = -0.8; % 1990年的气溶胶强迫

    % 定义气候参数
    param.clim.sens = 4.8; % 气候敏感度
    param.clim.kappa = 10; % 气候反馈参数
    param.clim.ppm_1850 = 290; % 1850年的CO2浓度
    param.clim.c_amp = 1.1; % 气候变化振幅
    param.clim.beta_l = 3.5; % 陆地碳反馈参数
    param.clim.beta_o = 2.4; % 海洋碳反馈参数
    param.clim.ocn_init = 600; % 初始海洋碳库
    param.clim.gamma_l = -95; % 陆地碳气候反馈
    param.clim.gamma_o = -60; % 海洋碳气候反馈
    param.clim.lnd_init = 2300; % 初始陆地碳库
    param.clim.cpl = 1; % 耦合参数
    param.clim.beta_od = 0.5; % 深海碳反馈参数
    param.clim.docn_init = 1.0e+05; % 初始深海碳库
    param.clim.kappa2 = 12.6; % 第二气候反馈参数
    param.clim.kdeep = 0.15; % 深海气候反馈参数

    % 定义CH4参数
    param.CH4.Year = 2000; % 年份
    param.CH4.species = 'CH4'; % 物种名称
    param.CH4.mass = 16; % 分子质量
    param.CH4.Xair = 0.1765; % 空气中的摩尔分数
    param.CH4.cPD = 1746; % 2000年的浓度
    param.CH4.dcdt = 5.97; % 浓度变化率
    param.CH4.cPI = 681.8; % 工业化前浓度
    param.CH4.kOH = 0.0895; % OH反应速率
    param.CH4.kCl = 0.0064; % Cl反应速率
    param.CH4.kStrat = 0.0083; % 平流层反应速率
    param.CH4.kSurf = 0.0144; % 表面反应速率
    param.CH4.anPI = 1.1344; % 工业化前的年均浓度
    param.CH4.aPI = 1.226; % 工业化前的年均浓度
    param.CH4.a2100 = 0.9; % 2100年的年均浓度
    param.CH4.aPIstrat = 0.8; % 工业化前的平流层浓度
    param.CH4.a2100strat = 0.7; % 2100年的平流层浓度
    param.CH4.sOH = -0.2832; % OH反应速率变化
    param.CH4.sStrat = 0; % 平流层反应速率变化
    param.CH4.b = 2.8; % 反应速率系数
    param.CH4.RFe = 3.7e-04; % 辐射强迫效率
    param.CH4.kMCF = 0.2483; % MCF反应速率
    param.CH4.kMCFstrat = 0.0235; % 平流层MCF反应速率
    param.CH4.kMCFocean = -0.0041; % 海洋MCF反应速率
    param.CH4.fillMCF = 0.9; % MCF填充系数
    param.CH4.fill = 0.95; % 填充系数
    param.CH4.r272 = 0.5823; % 272nm反应速率
    param.CH4.r225 = 0; % 225nm反应速率

    % 定义N2O参数
    param.N2O.Year = 2000; % 年份
    param.N2O.species = 'N2O'; % 物种名称
    param.N2O.mass = 28; % 分子质量
    param.N2O.Xair = 0.1765; % 空气中的摩尔分数
    param.N2O.cPD = 316; % 2000年的浓度
    param.N2O.dcdt = 0.635; % 浓度变化率
    param.N2O.cPI = 298; % 工业化前浓度
    param.N2O.kOH = 0; % OH反应速率
    param.N2O.kCl = 0; % Cl反应速率
    param.N2O.kStrat = 0.0088; % 平流层反应速率
    param.N2O.kSurf = 0; % 表面反应速率
    param.N2O.anPI = 0.91; % 工业化前的年均浓度
    param.N2O.aPI = 2.5; % 工业化前的年均浓度
    param.N2O.a2100 = 1.09; % 2100年的年均浓度
    param.N2O.aPIstrat = 0.814; % 工业化前的平流层浓度
    param.N2O.a2100strat = 1.057; % 2100年的平流层浓度
    param.N2O.sOH = 0; % OH反应速率变化
    param.N2O.sStrat = 0.1959; % 平流层反应速率变化
    param.N2O.b = 4.79; % 反应速率系数
    param.N2O.RFe = 0.003; % 辐射强迫效率
    param.N2O.kMCF = 0; % MCF反应速率
    param.N2O.kMCFstrat = 0; % 平流层MCF反应速率
    param.N2O.kMCFocean = 0; % 海洋MCF反应速率
    param.N2O.fillMCF = 1; % MCF填充系数
    param.N2O.fill = 0.7943; % 填充系数
    param.N2O.r272 = 0; % 272nm反应速率
    param.N2O.r225 = 0; % 225nm反应速率

    % 定义HFC134a参数
    param.HFC134a.Year = 2010; % 年份
    param.HFC134a.species = 'HFC134a'; % 物种名称
    param.HFC134a.mass = 102; % 分子质量
    param.HFC134a.Xair = 0.1765; % 空气中的摩尔分数
    param.HFC134a.cPD = 0.058; % 2000年的浓度
    param.HFC134a.dcdt = 0; % 浓度变化率
    param.HFC134a.cPI = 0; % 工业化前浓度
    param.HFC134a.kOH = 0; % OH反应速率
    param.HFC134a.kCl = 0; % Cl反应速率
    param.HFC134a.kStrat = 0; % 平流层反应速率
    param.HFC134a.kSurf = 0; % 表面反应速率
    param.HFC134a.anPI = 1; % 工业化前的年均浓度
    param.HFC134a.aPI = 1; % 工业化前的年均浓度
    param.HFC134a.a2100 = 1; % 2100年的年均浓度
    param.HFC134a.aPIstrat = 1; % 工业化前的平流层浓度
    param.HFC134a.a2100strat = 1; % 2100年的平流层浓度
    param.HFC134a.sOH = 0; % OH反应速率变化
    param.HFC134a.sStrat = 0; % 平流层反应速率变化
    param.HFC134a.b = 17.5; % 反应速率系数
    param.HFC134a.RFe = 0.16; % 辐射强迫效率
    param.HFC134a.kMCF = 0; % MCF反应速率
    param.HFC134a.kMCFstrat = 0; % 平流层MCF反应速率
    param.HFC134a.kMCFocean = 0; % 海洋MCF反应速率
    param.HFC134a.fillMCF = 1; % MCF填充系数
    param.HFC134a.fill = 0.97; % 填充系数
    param.HFC134a.r272 = 0.427; % 272nm反应速率
    param.HFC134a.r225 = 0.816; % 225nm反应速率
end
