import torch
import time
from unet import UNet  # 导入同一目录下的 UNet 模型

def main():
    # 创建模型实例
    model = UNet()
    model.eval()

    # 创建一个随机输入张量，假设输入尺寸为 3x512x512
    input_tensor = torch.randn(1, 3, 512, 512)

    # 测试维度的正确性
    with torch.no_grad():
        output_tensor = model(input_tensor)
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output_tensor.shape}")

    # 测试处理时间
    start_time = time.time()
    for _ in range(100):
        _ = model(input_tensor)
    end_time = time.time()
    avg_time = (end_time - start_time) / 100
    print(f"Average processing time per forward pass: {avg_time:.6f} seconds")

if __name__ == "__main__":
    main()