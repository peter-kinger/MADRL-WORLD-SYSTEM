# MADRL-WORLD-SYSTEM
this is a repository about  research on the domain of reinforcement learning and world system model to analyse the human activity and warming problems




# 迭代版本说明

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







