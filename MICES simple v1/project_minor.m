function [out, c_out] = project_minor(emis, in)
    % PROJECT_MINOR 函数用于计算次要温室气体的浓度和辐射强迫
    %
    % 输入:
    %   emis - 结构体，包含不同物种的排放数据
    %   in   - 结构体，包含其他输入参数
    %
    % 输出:
    %   out   - 结构体，包含计算出的浓度和辐射强迫
    %   c_out - 结构体，包含其他计算出的结果（如果有）

    % 提取时间和排放数据
    tt = in.CH4.tt;
    time = emis.CFC_11.time;

    % 定义各物种的辐射效率
    effic.CF4 = 0.1;
    effic.C2F6 = 0.26;
    effic.C6F14 = 0.49;
    effic.HFC23 = 0.19;
    effic.HFC32 = 0.11;
    effic.HFC43_10 = 0.4;
    effic.HFC125 = 0.18;
    effic.HFC134a = 0.16;
    effic.HFC143a = 0.13;
    effic.HFC227ea = 0.26;
    effic.HFC245fa = 0.28;
    effic.SF6 = 0.52;
    effic.CFC_11 = 0.25;
    effic.CFC_12 = 0.28;
    effic.CFC_113 = 0.3;
    effic.CFC_114 = 0.31;
    effic.CFC_115 = 0.18;
    effic.CARB_TET = 0.13;
    effic.MCF = 0.06;
    effic.HCFC_22 = 0.2;
    effic.HCFC_141B = 0.14;
    effic.HCFC_142B = 0.2;
    effic.HALON1211 = 0.3;
    effic.HALON1301 = 0.32;
    effic.HALON2402 = 0.33;
    effic.CH3BR = 0.01;
    effic.CH3CL = 0.01;

    % 定义各物种的寿命（单位：年）
    lifetime.CF4 = 50000;
    lifetime.C2F6 = 10000;
    lifetime.C6F14 = 3200;
    lifetime.HFC23 = 270;
    lifetime.HFC32 = 4.9;
    lifetime.HFC43_10 = 15.9;
    lifetime.HFC125 = 29;
    lifetime.HFC134a = 14;
    lifetime.HFC143a = 52;
    lifetime.HFC227ea = 34.2;
    lifetime.HFC245fa = 7.6;
    lifetime.SF6 = 3200;
    lifetime.CFC_11 = 45;
    lifetime.CFC_12 = 100;
    lifetime.CFC_113 = 640;
    lifetime.CFC_114 = 300;
    lifetime.CFC_115 = 1700;
    lifetime.CARB_TET = 26;
    lifetime.MCF = 5;
    lifetime.HCFC_22 = 12;
    lifetime.HCFC_141B = 9.3;
    lifetime.HCFC_142B = 17.9;
    lifetime.HALON1211 = 16;
    lifetime.HALON1301 = 65;
    lifetime.HALON2402 = 20;
    lifetime.CH3BR = 0.7;
    lifetime.CH3CL = 1.0;

    % 定义各物种的分子质量（单位：g/mol）
    mass.CF4 = 88;
    mass.C2F6 = 138.01;
    mass.C6F14 = 338.0418;
    mass.HFC23 = 70;
    mass.HFC32 = 86.2;
    mass.HFC43_10 = 252;
    mass.HFC125 = 120.02;
    mass.HFC134a = 102.03;
    mass.HFC143a = 84.04;
    mass.HFC227ea = 170.03;
    mass.HFC245fa = 134.05;
    mass.SF6 = 146.06;
    mass.CFC_11 = 137.37;
    mass.CFC_12 = 120.91;
    mass.CFC_113 = 187.38;
    mass.CFC_114 = 170.92;
    mass.CFC_115 = 154.466;
    mass.CARB_TET = 153.82;
    mass.MCF = 133.4;
    mass.HCFC_22 = 86.47;
    mass.HCFC_141B = 116.94;
    mass.HCFC_142B = 100.49;
    mass.HALON1211 = 165.36;
    mass.HALON1301 = 148.91;
    mass.HALON2402 = 259.823013;
    mass.CH3BR = 94.94;
    mass.CH3CL = 50.49;

    % 获取所有物种的名称
    species = fields(lifetime);

    % 定义转换函数：从千吨到ppb
    kt_to_ppb = @(x) (1e9 / x) / (5e21 / 28.97) * 1e9;

    % 初始化输出
    out.rad = 0;

    % 遍历每个物种，计算其浓度和辐射强迫
    for j = 1:numel(species)
        tconc = zeros(size(time));
        for i = 2:numel(time)
            tconc(i) = tconc(i - 1) + emis.(species{j}).val(i) * ...
                       kt_to_ppb(mass.(species{j})) - tconc(i - 1) / lifetime.(species{j});
        end
        out.species.(species{j}).val = interp1(emis.CFC_11.time, tconc, tt);
        out.species.(species{j}).r_eff = effic.(species{j});
        out.species.(species{j}).radf = out.species.(species{j}).val * out.species.(species{j}).r_eff;

        % 累加辐射强迫
        out.rad = out.rad + out.species.(species{j}).val * ...
                  out.species.(species{j}).r_eff;
    end
end
