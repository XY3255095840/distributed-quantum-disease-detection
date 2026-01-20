import torch

def test_cuda():
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("CUDA不可用，测试终止。")
        return

    # 打印GPU信息
    print("检测到CUDA设备：", torch.cuda.get_device_name(0))
    print("CUDA版本：", torch.version.cuda)

    # 创建张量并移至GPU
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')

    # 执行矩阵乘法
    z = torch.matmul(x, y)

    # 同步以确保计算完成
    torch.cuda.synchronize()

    print("CUDA矩阵乘法测试通过，结果形状：", z.shape)

if __name__ == "__main__":
    test_cuda()
