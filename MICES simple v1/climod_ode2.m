function out = climod_ode2(emis, param, tt, use)
    % CLIMOD_ODE2 - 进行气候模拟，包括甲烷（CH₄）和氧化亚氮（N₂O）的辐射强迫和温室气体浓度对气候的影响
    % 输入参数:
    %   emis  - 排放数据的结构体
    %   param - 参数数据的结构体
    %   tt    - 时间序列
    %   use   - 字符串，指定使用哪种气体进行模拟（例如，'N2O' 或 'CH4'）
    % 输出:
    %   out   - 模拟结果的结构体

    % 处理甲烷和氧化亚氮的辐射强迫
    if isempty(strmatch(use, 'N2O'))
        % 如果 use 中没有 'N2O'，那么调用 project_new 函数处理 CH4 的相关数据
        out.CH4 = project_new(emis.CH4, param.CH4, 1, tt);
    end
    if isempty(strmatch(use, 'CH4'))
        % 如果 use 中没有 'CH4'，则调用 project_new 函数处理 N2O 的相关数据
        out.N2O = project_new(emis.N2O, param.N2O, 1, tt);
    end

    if isempty(strmatch(use, 'CH4')) + isempty(strmatch(use, 'N2O')) == 2
        % 处理较小气体和气溶胶的辐射强迫
        out.minor = project_minor(emis, out);
        out.aer = project_new_aer(emis, out, param.aer);

        % 初始化气候模型参数
        cpl = param.clim.cpl;
        mat = 1.8e20; % 大气中的摩尔数
        lambda = 3.8 / param.clim.sens; % 敏感度参数
        kappa = param.clim.kappa; % 海洋热量吸收参数
        kappa2 = param.clim.kappa2; % 第二海洋热量吸收参数
        kdeep = param.clim.kdeep; % 深海热量吸收参数
        co2c0 = param.clim.ppm_1850; % 初始大气CO2浓度
        tem0 = 0; % 初始温度
        alpha = 1e6 / mat / 12 * 1e15; % ppm到Pg的转换
        Ca0 = co2c0 / alpha; % 初始大气碳量（Pg）

        % 初始化陆地和海洋碳循环参数
        cinl(1) = param.clim.lnd_init;
        gamma_l = param.clim.gamma_l; % 陆地碳气候反馈（Pg/K）
        beta_l = param.clim.beta_l; % 陆地碳浓度反馈（Pg/ppm）
        c_amp = param.clim.c_amp; % 碳吸收的温度放大参数
        gamma_o = param.clim.gamma_o; % 海洋碳气候反馈（Pg/K）
        beta_o = param.clim.beta_o; % 海洋碳浓度反馈（Pg/ppm）
        cino(1) = param.clim.ocn_init;
        oco2c(1) = param.clim.ppm_1850;
        rho = oco2c(1) / cino(1);
        cinod(1) = param.clim.docn_init;
        beta_od = param.clim.beta_od;
        odco2c(1) = param.clim.ppm_1850;
        rho2 = odco2c(1) / cinod(1);

        
        % 插值排放数据
        emis_fossil = interp1(emis.FossilCO2.time, emis.FossilCO2.val, tt); % 化石燃料排放数据
        emis_lu = interp1(emis.OtherCO2.time, emis.OtherCO2.val, tt); % 土地利用排放数据

        % 计算非CO2辐射强迫和碳排放总量
        for i = 1:numel(tt)
            f_nonco2(i) = out.CH4.RFrcp(i) + out.N2O.RFrcp(i) + out.aer.aer_f(i) + out.minor.rad(i); % 非CO2辐射强迫总量
            gtc_in(i) = emis_fossil(i) + emis_lu(i); % 人为碳排放总量（Pg）
        end

        % 尤其关注这里的 f_nonco2 数值大小

        % 使用ODE求解器模拟温度和碳循环的演变
        [t, C] = ode45(@(t, C) tsolve2(C, Ca0, kappa, kappa2, kdeep, lambda, gamma_l, gamma_o, ...
            alpha, rho, rho2, beta_l, beta_o, beta_od, f_nonco2, gtc_in, tt, t, cpl, c_amp), tt, [tem0, Ca0, cino(1), cinod(1), tem0]);

        % % 存储结果
        % out.clim.f_co2 = 6.3 * log(C(:, 2) / Ca0); % CO2辐射强迫
        % out.clim.f_nonco2 = f_nonco2; % 非CO2辐射强迫
        % out.clim.gtc_in = gtc_in; % 人为碳排放总量
        % out.clim.time = t; % 时间序列

        C
        % 计算基线温度
        out.clim.time = t;
        tim_pi = find(t > 1850 & t < 1900);
        out.clim.tem_atm_lx = C(:, 1); % 大气温度
        
        out.clim.tem = C(:, 1) - mean(C(tim_pi, 1)); % 温度变化

        out.clim.cina = C(:, 2); % 大气碳量
        out.clim.ppm = out.clim.cina * alpha; % 大气碳浓度（ppm）

        out.clim.cino = C(:, 3); % 浅海碳量
        out.clim.cinod = C(:, 4); % 深海碳量
        out.clim.tem_ocean = C(:, 5); % 海洋温度

 
    end
end
