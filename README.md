# MADRL-WORLD-SYSTEM
this is a repository about  research on the domain of reinforcement learning and world system model to analyse the human activity and warming problems

### Env 环境主要参考依据于：
仓库内容是对他们文章内容理解的重现，非常感谢他们提出的优秀模型！
The repository content is a reproduction of the understanding of the content of their article, and I am very grateful to them for the excellent model they proposed!

Sanderson, Benjamin M., et al. "Community climate simulations to assess avoided impacts in 1.5 and 2 C futures." Earth System Dynamics 8.3 (2017): 827-847.

Ramanathan, Veerabhadran, Yangyang Xu, and Anthony Versaci. "Modelling human–natural systems interactions with implications for twenty-first-century warming." Nature Sustainability 5.3 (2022): 263-271.

# mices 迭代版本说明

这里每次迭代什么都写在一起进行备注操作

## MICES simple v1

### 主要运行程序

主要是原来代码冗余部分的处理操作，整理了其中的输入数据

- 其中包含了程序运行的主体部分，包含从开始 ssp-rcp 原始的预估数据获取的数据经过一系列处理之后计算得到 gtc_in(温室气体数据) 和 f_nonco2(非 co2 强迫数据)，包含了众多的部分
- 包含了 cliamte_ode2 和 tsolve2 计算的主体部分，这里是计算的核心部分
- 通过保存相关的结果然后绘图输出，simple 版本对其进行了简化操作

> 未来可以考虑作为读取不同场景数据的预处理文件，通过其中截止打断点获取其中的数据数据，这部分当然需要进行相关更改

### input 数据

里面是原始输入的不同 dat 文件部分

### xlsx数据

- 读取的输入数据：读取的 mat 数据显式地保存为 xlsx 文件以供检查移植获取，
- mices 程序完整运行的数据：mices_var_result.xlsx 文件以供对照结果偏差

### tutorial 

里面涉及了对于关键组件运行理解的部分，比如 ode 里面的相关学习

## MICES matlab ode v2

主要是考虑之前 simple 的代码里面对于跳转函数过于复杂，参数部分内容

优化:

- 函数集中化
- 参数显式化
- 插值部分的探究，
- 运行顺序的探究不同，联动之间代码书写有顺序

但是出现了问题：可以复现，但是结果增减性不对，结果数值范围有偏差

## MICES depart v3

通过比较之前的输入是正确的，运行逻辑也是对的，将原始的部分进行拆分，直接截取操作

注意运行主程序是 `run_mices.m`

优化：

- 参数计算部分是通过给值，现在考虑通过计算得到，而不是直接初始化具体数值
- 保存原有注释部分，对应知道数值如何获取起始部分

## MICES python v4

这里经过两个版本的 python 迭代，主要是其中的数值计算和精度

- 精度的改变
- python 中加载数据转换部分，精度转换为 float
- python 中插值部分对应程序包 `np.interp`

## MICES python-env-sdm v5

参数部分的阅读，单位弄懂，统一返回结果展示

尝试更换 odeint 里面统一返回接口，参照 theo 的 copan 部分

主要解决其中的 rl 部分接口重新书写，

尝试替换 iseec 里面的初始条件

## ISEEC python-env v5



# ISEEC 模型 v5 - 增强插值版

## 概述

此 Python 脚本（`iseec-model_v5_interp.py`）实现了一个综合能源与排放气候（ISEEC）模型，用于模拟碳排放、能源使用和温度动态。该模型使用微分方程来表示各种环境、经济和能源相关参数之间的相互作用。此外，它通过插值来处理外部输入数据，允许在模拟过程中为某些变量提供连续的数据输入。

## 特性

- **ODE 模拟**：使用 `scipy.integrate.odeint` 来求解代表气候相关过程的微分方程。
- **参数插值**：使用 `np.interp` 将离散数据输入转换为连续函数，提高模拟精度。
- **时间范围**：模拟环境动态从 1850 年至 2100 年。
- **输出**：将详细的输出数据（如温度、CO₂ 排放、能源使用）保存到 Excel 文件中，以便进一步分析。

## 环境要求

- Python 3.x
- 所需库：
  - `numpy`
  - `scipy`
  - `pandas`
  - `matplotlib`

可以通过以下命令安装所需库：

```
bash


复制代码
pip install numpy scipy pandas matplotlib
```

## 生成的文件

- `iseec_10_dimension_results_ACEno_revised.xlsx`：包含主要变量随时间的模拟结果。
- `iseec_10_dimension_t_ode.xlsx`：包含 ODE 求解过程中记录的时间步。(主要用来参照调试使用)
- `iseec_results_ode_withoutinterp.xlsx`：包含未插值调整的参数计算值。（ode 函数自定义求解的结果）
- `iseec_results_time_withinterp.xlsx`：包含与主时间序列匹配的插值结果。（主要参照比较的结果）

## 脚本结构

### 1. **参数初始化**（包括外界输入）

脚本初始化了多个常数和参数，包括气候和碳循环反馈参数、基准 CO₂ 排放、GDP 数据和初始能源相关值。数据源包括：

- `ISEEC-SSM/output_CO2emission_baseline18502100_LU.csv`：基准土地利用 CO₂ 排放数据。
- `ISEEC-SSM/output_nonco2_forcing_all_BAU.CSV`：非 CO₂ 强迫数据。（默认图表中计算得到的）
- `ISEEC-SSM/output_energy_MYbaseline18502100_total_formulated.csv`：总能源基准数据。
- `ISEEC-SSM/output_GDP_SSP5_this study.csv`：SSP5 场景的 GDP 数据。

### 2. **函数定义**

#### `IEM(y, t)`

定义了温度和碳动态的微分方程主函数：

- **插值输入**：对于某些参数（如 `CO2emission_baseline18502100_LU`、`additionalforcing`、`energy_MYbaseline18502100_total` 和 `GDP_formulated`），通过 `np.interp` 在每个时间步获得连续的函数值。

- 方程组成

  ：

  - ϕi\phi_iϕi：依赖于截至当前时间的累计 CO₂ 排放。
  - αi\alpha_iαi：一个基于当前 CO₂ 水平的动态参数。
  - **温度和排放计算**：利用基准和插值数据计算温度（`T_a`）、大气 CO₂（`C_a`）和海洋 CO₂ 水平。

### 3. **模拟执行**

脚本初始化初始条件，并在指定时间范围内使用 `odeint` 运行模拟。

- **初始条件**：在 `y0` 数组中设置，包含温度和 CO₂ 水平的初始值。
- **输出变量**：提取每个组件的结果（例如 `T_a`、`C_a`、`C_o`），并将它们保存到结构化的输出文件中。

### 4. **数据输出**

脚本使用 `pandas` 将模拟结果保存到各种 Excel 文件中。关键结果包括：

- 主要变量的时间序列数据（如温度、CO₂ 水平）。
- 可再生能源与总能源需求的比例。
- 基准和调整后的能源使用插值结果。

## 使用方法

确保所有必要的数据文件位于 `ISEEC-SSM` 目录中，然后执行脚本：

```
bash


复制代码
python iseec-model_v5_interp.py
```

运行完成后，脚本将输出多个包含模拟结果的 Excel 文件，可用于进一步分析、可视化或与实测数据进行验证。

## 注意事项

- **插值处理**：





