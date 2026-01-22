import pennylane as qml
import torch.nn as nn
import torch
import numpy as np

# -----------------------------------------------------------------------------
# 1. 配置参数
# -----------------------------------------------------------------------------
n_layers = 2        # 2层 * 8比特 = 16维特征，刚好覆盖输入维度
n_qubits = 8        # 量子比特数
n_measure = 5       # 测量前5个比特，输出维度为 2^5 = 32
input_dim = 16      # 明确指定输入特征维度

# -----------------------------------------------------------------------------
# 2. 定义设备
# 【关键修复 1】必须添加 batch_obs=True，否则无法处理 n*16 的批量输入
# -----------------------------------------------------------------------------
dev = qml.device('default.qubit', wires=n_qubits)
# -----------------------------------------------------------------------------
# 3. 构建量子层 (DRN + MPS)
# -----------------------------------------------------------------------------
def dr_mps_layer(layer_idx, n_qubits, weights, inputs, input_scalings):
    """
    单层 DRN MPS 层
    """
    # 【关键修复 2】获取特征维度 (16)，而不是 batch size (n)
    # inputs.shape 在 forward 中通常是 [batch_size, feature_dim]
    feat_dim = inputs.shape[-1] 
    
    # 计算当前层负责的特征起始索引
    # Layer 0 -> start 0, Layer 1 -> start 8
    start_feat_idx = (layer_idx * n_qubits) % feat_dim
    
    # --- A. 数据重传 (Data Re-uploading) ---
    for i in range(n_qubits):
        # 确定当前比特对应的特征索引 (0-15)
        feat_idx = (start_feat_idx + i) % feat_dim
        
        # 【关键修复 3】正确提取特征列，保持 Batch 维度
        # inputs[..., feat_idx] 会取出所有样本的第 feat_idx 个特征
        # val 的形状将是 (batch_size,)
        val = inputs[..., feat_idx]
        
        # 获取可训练参数
        w = input_scalings[layer_idx, i]
        theta = weights[layer_idx, i]
        
        # 参数广播：theta(标量) + w(标量) * val(向量) = 结果(向量)
        # PennyLane 会自动为每个样本生成并行的旋转门
        qml.RY(theta + w * val, wires=i)
    
    # --- B. 纠缠层 (Entanglement) ---
    # 简单的链式纠缠
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    # 可选：闭环纠缠 (Ring)
    # qml.CNOT(wires=[n_qubits-1, 0]) 

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights, scalings):
    for layer in range(n_layers):
        dr_mps_layer(layer, n_qubits, weights, inputs, scalings)
    
    # 测量前 n_measure 个比特
    return qml.probs(wires=range(n_measure))

# -----------------------------------------------------------------------------
# 4. PyTorch 模型定义
# -----------------------------------------------------------------------------
class QNN(nn.Module):
    def __init__(self):
        super(QNN, self).__init__()
        
        # 定义参数形状
        weight_shapes = {
            "weights": (n_layers, n_qubits),
            "scalings": (n_layers, n_qubits)
        }
        
        # 初始化 TorchLayer
        # init_method 可以自定义参数初始化的分布，这里使用默认
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
    def forward(self, inputs):
        # inputs shape: [batch_size, 16]
        # TorchLayer 内部会自动处理 batch，只要 QNode 配置了 batch_obs=True
        return self.q_layer(inputs)

# -----------------------------------------------------------------------------
# 测试代码
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 模拟数据：n=10 个样本，每个样本 16 维
    n = 10
    test_input = torch.randn(n, 16)
    
    model = QNN()
    
    try:
        output = model(test_input)
        print(f"输入形状: {test_input.shape}")
        print(f"输出形状: {output.shape}")
        
        expected_shape = torch.Size([n, 2**n_measure]) # [10, 32]
        if output.shape == expected_shape:
            print("✅ 测试通过：代码已适应 n*16 输入！")
        else:
            print(f"❌ 维度错误：预期 {expected_shape}，实际 {output.shape}")
            
    except Exception as e:
        print(f"❌ 运行时错误: {e}")
        import traceback
        traceback.print_exc()