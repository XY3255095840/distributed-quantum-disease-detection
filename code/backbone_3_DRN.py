from matplotlib.pyplot import flag
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
# from VGGNet import vgg16_net
# from QNN import QNN
# from QNN_copy import QNN
from mobilnet_DRN import MobileNetV2
from mps3_DRN import QNN

# Classic Network: VGG16
# Quantum Network: 

# n_layers = 1 # 电路层数
# n_qubits = 4 # vgg输出等于量子比特个数
# # qml.device 指定后端模拟器和量子比特数，这里default.qubit模拟纯态qubit，也可以用default.mixed模拟混合态qubit，对应含噪电路

class QCNet(nn.Module):
    def __init__(self):
        super(QCNet, self).__init__()
        self.qubits = 8
        self.input_dim = 16 # n_qubits 的两倍
        self.nclass = 3
        self.flag=1
        # 经典神经网络
        self.CModel = MobileNetV2()
        
        # 全连接参数配置：将 MobileNetV2 的 1280 维映射到 16 维
        self.fc1 = nn.Linear(16, self.input_dim) 
        # 量子神经网络
        self.QModel = QNN()
        # 全连接层：QModel 输出为 2^n_qubits_2 = 32
        self.fc2 = nn.Linear(32, self.nclass)
        # self.fc2 =nn.Linear(self.qubits, 3)

        # self.lr1 = nn.LeakyReLU(0.1)



    def forward(self, x):
        # Forward Propagation
        x = self.CModel(x)
        if self.flag==1:
            self.flag=0
            print("MobileNetV2输出 shape:", x.shape)
        x = self.QModel(x)    # 进入量子网络进行 DRN 处理
        x = self.fc2(x)
        return x
