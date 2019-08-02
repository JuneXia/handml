import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 生成一个2x3的矩阵，假设这是模型预测值，表示有2条预测数据，每条是一个3维的激活值
inputs_tensor = torch.FloatTensor([
[10, 3,  1],
[-1, 0, -4]
])

# 真实值
targets_tensor = torch.LongTensor([1, 2])


# 手动计算softmax
# ***********************************************************
inputs_exp = inputs_tensor.exp()
inputs_exp_sum = inputs_exp.sum(dim=1)
inputs_exp = inputs_exp.transpose(0, 1)
softmax_result = torch.div(inputs_exp, inputs_exp_sum)  # torch.div的两个输入张量必须广播一致的，而这两个张量的类型必须是一致的。
softmax_result = softmax_result.transpose(0, 1)
print(softmax_result)
# ***********************************************************


# 使用F.softmax计算softmax
softmax_result = F.softmax(inputs_tensor)
print(softmax_result)

# 使用np.log计算得到log_softmax
log_softmax_result = np.log(softmax_result.data)
print(log_softmax_result)

# 直接调用F.log_softmax计算得到log_softmax
log_softmax_result = F.log_softmax(inputs_tensor)
print(log_softmax_result)

# 手动计算交叉熵损失
# ***********************************************************
_targets_tensor = targets_tensor.view(-1, 1)
onehot = torch.zeros(2, 3).scatter_(1, _targets_tensor, 1)  # 对真实标签做one-hot编码
product = onehot*log_softmax_result
cross_entropy = -product.sum(dim=1)
cross_entropy_loss = cross_entropy.mean()
print(cross_entropy_loss)
# ***********************************************************


# 使用nn.NLLLoss()计算log_softmax得到交叉熵损失
loss = nn.NLLLoss()
cross_entropy_loss = loss(log_softmax_result, targets_tensor)
print(cross_entropy_loss)


# 直接使用nn.CrossEntropyLoss计算交叉熵损失
loss = nn.CrossEntropyLoss()
cross_entropy_loss = loss(inputs_tensor, targets_tensor)
print(cross_entropy_loss)

