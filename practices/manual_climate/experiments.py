
from learn_class import Learn

import torch

# 不要书写的内容，删除
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class PB_Learn(Learn):
    def __init__(self, **kwargs):
        super(PB_Learn, self).__init__(**kwargs)


# TODO
# 增加噪音观察的部分
# 增加改变奖励函数的部分


    