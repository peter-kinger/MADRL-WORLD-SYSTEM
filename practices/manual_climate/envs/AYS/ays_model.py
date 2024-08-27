"""最原始的模型对应的是文件，区别与耦合框架的函数

"""

# def AYS_rescaled_rhs(ays, t=0, beta=None, epsilon=None, phi=None, rho=None, sigma=None, tau_A=None, tau_S=None, theta=None):
def AYS_rescaled_rhs(ays, t=0, beta=None, epsilon=None, phi=None, rho=None, sigma=None, tau_A=None, tau_S=None, theta=None):
    a, y, s = ays
    # 小写 y 和 大写 Y 还不一样
    # A, y, s = Ays

    # 补充添加其中的参数值
    S_mid = 0.5
    W_mid = 0.5
    A_mid = 0.5


    s_inv = 1 - s
    s_inv_rho = s_inv ** rho
    
    K = s_inv_rho / (s_inv_rho + (S_mid * s / sigma) ** rho )

    a_inv = 1 - a
    w_inv = 1 - y
    Y = W_mid * y / w_inv
    A = A_mid * a / a_inv

    # 未缩放形式如下：

    adot = K / (phi * epsilon * A_mid) * a_inv * a_inv * Y - a * a_inv / tau_A
    # 只有y 进行了缩放
    ydot = y * w_inv * ( beta - theta * A )
    # 只有s 进行了缩放，只有自己的量需要进行缩放操作
    sdot = (1 - K) * s_inv * s_inv * Y / (epsilon * S_mid) - s * s_inv / tau_S

    return adot, ydot, sdot


def lx_new_knoewAYS_rescaled_rhs(ays, t=0, beta=None, epsilon=None, phi=None, rho=None, sigma=None, tau_A=None, tau_S=None, theta=None):
    """用于林鑫的相关理解


    :param ays:
    :param t:
    :param beta:
    :param epsilon:
    :param phi:
    :param rho:
    :param sigma:
    :param tau_A:
    :param tau_S:
    :param theta:
    :return:
    """
    a, y, s = ays
    # 小写 y 和 大写 Y 还不一样
    # A, y, s = Ays

    # 补充添加其中的参数值
    S_mid = 0.5
    W_mid = 0.5
    A_mid = 0.5

    # 全部替换掉其中的 s_inv 部分
    # s_inv = 1 - s
    s_inv_rho = (1 - s) ** rho

    # K = s_inv_rho / (s_inv_rho + (S_mid * s / sigma) ** rho )

    # a_inv = 1 - a
    # w_inv = 1 - y

    Y = W_mid * y / (1 - y)
    A = A_mid * a / (1 - a)

    # 未缩放形式如下：

    adot = (s_inv_rho / (s_inv_rho + (S_mid * s / sigma) ** rho )) / (phi * epsilon * A_mid) * (1 - a) * (1 - a) * Y - a * (1 - a) / tau_A
    # 只有y 进行了缩放
    ydot = y * (1 - y) * ( beta - theta * A )
    # 只有s 进行了缩放，只有自己的量需要进行缩放操作
    sdot = (1 - (s_inv_rho / (s_inv_rho + (S_mid * s / sigma) ** rho ))) * ((1 - s) ** rho) * ((1 - s) ** rho) * Y / (epsilon * S_mid) - s * ((1 - s) ** rho) / tau_S

    return adot, ydot, sdot