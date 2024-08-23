function out = project_new(emis, param, Nmc, tt)
    % PROJECT_NEW - 进行气候模拟的蒙特卡洛方法
    %
    % 输入参数:
    %   emis  - 排放数据的结构体
    %   param - 参数数据的结构体
    %   Nmc   - 蒙特卡洛模拟的迭代次数
    %   tt    - 时间序列
    %
    % 输出:
    %   out   - 模拟结果的结构体

    % 初始化排放数据
    tErcp_glob = emis.time;
    Ercp_glob = emis.val;
    
    % 初始化变量
    Epd_anthro = zeros(Nmc, 1); % 当前人类活动排放量，Tg/a
    Epd_nat = zeros(Nmc, 1); % 当前自然排放量，Tg/a
    Epd = zeros(Nmc, 1); % 当前总排放量，Tg/a
    Epi = zeros(Nmc, 1); % 工业前总排放量，Tg/a
    kMCFtotal = zeros(Nmc, 1);
    kOHpd = zeros(Nmc, 1);
    kStratPD = zeros(Nmc, 1);

    % 检查是否进行未来投影
    doProj = 1;
    if doProj 
        NT = numel(tt); % 时间点数
        cF = zeros(Nmc, NT); % 未来浓度，ppb
        Ercp = zeros(Nmc, NT); % 重标RCP排放量，Tg/a
        kFoh = zeros(Nmc, NT); % 未来OH损失频率，1/a
        kFstrat = zeros(Nmc, NT); % 未来平流层损失频率，1/a
        kF = zeros(Nmc, NT); % 未来总损失频率，1/a
        RFrcp = zeros(Nmc, NT); % 未来辐射强迫，W/m2
    end

    % 蒙特卡洛模拟
    for i = 1:Nmc
        % 填充因子
        fill = param.fill();

        % 当前总损失频率，1/a
        kMCFtotal(i) = param.fillMCF() * (param.kMCF() - param.kMCFstrat() - param.kMCFocean());

        % 当前OH损失频率，1/a
        kOHpd(i) = kMCFtotal(i) * param.r272() / fill;

        % 当前平流层损失频率，1/a
        kStratPD(i) = param.kStrat();

        % 其他过程的当前损失频率，1/a
        kOther = param.kCl() + param.kSurf();

        % 当前总损失频率，1/a
        kPD = kOHpd(i) + kStratPD(i) + kOther;

        % 工业前总损失频率，1/a
        kPI = param.aPI() * kOHpd(i) + param.aPIstrat() * kStratPD(i) + kOther;

        % 转换因子 ppb -> Tg
        ppb2Tg = param.Xair * param.mass * fill;

        % 当前浓度，ppb
        cPD = param.cPD();

        % 工业前浓度，ppb
        cPI = param.cPI();

        % 当前负荷，Tg
        Bpd = ppb2Tg * cPD;

        % 当前排放量，Tg/a
        Epd(i) = kPD * Bpd + ppb2Tg * param.dcdt();

        % 工业前负荷，Tg
        Bpi = ppb2Tg * cPI;

        % 工业前排放量，Tg/a
        Epi(i) = kPI * Bpi;

        % 当前自然排放量，Tg/a
        Epd_nat(i) = Epi(i) * param.anPI();

        % 当前人类活动排放量，Tg/a
        Epd_anthro(i) = Epd(i) - Epd_nat(i);

        if Epd_anthro(i) < 0
            x = 1; % 如果当前人类活动排放量小于0，设置x为1
        end

        if doProj
            % 未来RCP排放量，按2010年约束重标，Tg/a
            E = @(t) Ercp0(t) * Epd_anthro(i) / Ercp0(param.Year);

            % 未来OH损失频率变化，单位无关
            a = param.a2100();
            aF = @(t) interp1([0 param.Year 2100 2500], [1 1 a a], t);

            % 未来平流层损失频率变化，单位无关
            a = param.a2100strat();
            aFstrat = @(t) interp1([0 param.Year 2100 2500], [1 1 a a], t);

            % 当前平流层OH反馈因子，dln(OH)/dln(C) = dln(kOH)/dln(C)
            sPD = param.sOH();

            % 当前平流层反馈因子，dln(kStrat)/dln(C)
            sStratPD = param.sStrat();

            % 未来总损失频率，1/a
            kF1 = @(t, c) kOHpd(i) * exp(sPD * log(c ./ cPD)) .* aF(t);
            kF2 = @(t, c) kStratPD(i) * exp(sStratPD * log(c ./ cPD)) .* aFstrat(t);
            kFtot = @(t, c) kF1(t, c) + kF2(t, c) + kOther;

            % 定义积分函数
            dydt = @(t, c) (E(t) + Epi(i)) / ppb2Tg - kFtot(t, c) * c;

            % 积分未来负荷，ppb
            sol = ode23(dydt, [1765, 2500], cPI);

            % 未来浓度在十年间隔，ppb
            cF(i, :) = deval(sol, tt);

            % 保存重标未来RCP排放量，Tg/a
            Ercp(i, :) = E(tt);

            % 保存未来OH损失频率，1/a
            kFoh(i, :) = kF1(tt, cF(i, :));
            kFstrat(i, :) = kF2(tt, cF(i, :));
            kF(i, :) = kFtot(tt, cF(i, :));

            % 保存未来辐射强迫，W/m2
            RFrcp(i, :) = (cF(i, :) - cPI) * param.RFe();
        end
    end

    % 保存蒙特卡洛迭代结果
    out.tt = tt;
    out.Ercp = Ercp;
    out.cF = cF;
    out.RFrcp = RFrcp;
    out.kF = kF;
    out.kFoh = kFoh;
    out.kFstrat = kFstrat;
    out.kMCFtotal = kMCFtotal;

    % 内部函数定义
    function E = Ercp0(t)
        % RCP排放量，Tg/a
        E = interp1(tErcp_glob, Ercp_glob, t);
    end
end
