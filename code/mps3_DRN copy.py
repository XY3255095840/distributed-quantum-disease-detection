import pennylane as qml
import torch.nn as nn
# import torch.nn.functional as F
import torch
import numpy as np
import os

# 文件夹中第4组的方案

# 卷积电路分割后不能出现纠缠现象！！！

# 问题：参数对应问题，参数维度等########################################
# 问题：数据输入网络问题，切割点在制备过程中是否还需要inputs########################################

# 只要跟切割电路有关，是否都可以用该方法复现？将所有与切割电路有关的输出线路，全部测量，根据切割点测量值，逐一恢复
# qml.qnn.TorchLayer层的返回值类型  tensor类型
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Quantum Network Configuration
n_layers = 2      # 2层 * 8比特 = 16维输入
n_qubits = 8      # 总比特数设为8
n_measure = 5     # 测量前5个比特以获得 2^5 = 32 个输出，适配 fc2

dev = qml.device('lightning.qubit', wires=n_qubits)

def dr_mps_layer(layer_idx, n_qubits, weights, inputs, input_scalings):
    """
    单层 DRN MPS 层
    每个数据点只在层首重传一次
    """
    start_idx = (layer_idx * n_qubits) % len(inputs)
    
    # 1. 数据重传 (Data Re-uploading) - 每层每个比特只传一次
    for i in range(n_qubits):
        val = inputs[(start_idx + i) % len(inputs)]
        w = input_scalings[layer_idx, i]
        theta = weights[layer_idx, i]
        qml.RY(theta + w * val, wires=i)
    
    # 2. 纠缠层 (Entanglement) - 简单的 MPS 链式纠缠
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights, scalings):
    for layer in range(n_layers):
        dr_mps_layer(layer, n_qubits, weights, inputs, scalings)
    # 测量前 n_measure 个比特，返回 2^n_measure 维概率向量
    return qml.probs(wires=range(n_measure))

class QNN(nn.Module):
    def __init__(self):
        super(QNN, self).__init__()
        
        # 参数形状：(层数, 比特数)
        weight_shapes = {
            "weights": (n_layers, n_qubits),
            "scalings": (n_layers, n_qubits)
        }
        
        # 使用单个 TorchLayer 替换原来的多个切分层
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    def forward(self, inputs):
        # inputs shape: [batch_size, 16]
        # TorchLayer 内部处理了 batch，不需要手动循环
        return self.q_layer(inputs)

