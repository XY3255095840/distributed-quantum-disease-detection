import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
# from VGGNet import vgg16_net
# from QNN import QNN
# from QNN_copy import QNN
from mobilnet import MobileNetV2
from mps3 import QNN

# Classic Network: VGG16
# Quantum Network: 

# n_layers = 1 # 电路层数
# n_qubits = 4 # vgg输出等于量子比特个数
# # qml.device 指定后端模拟器和量子比特数，这里default.qubit模拟纯态qubit，也可以用default.mixed模拟混合态qubit，对应含噪电路

class QCNet(nn.Module):
    def __init__(self):
        super(QCNet, self).__init__()
        self.nclass = 3
        # 仅保留经典神经网络
        self.CModel = MobileNetV2()
        # 根据MobileNetV2输出维度直接映射到类别数
        self.fc = nn.Linear(8, self.nclass)  # MobileNetV2特征维度为1280

    def forward(self, x):
        # 前向传播：经典网络提取特征后直接分类
        x = self.CModel(x)
        x = self.fc(x)
        return x
