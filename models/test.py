import torch
import time
import matplotlib.pyplot as plt
from unet import UNet  # 导入同一目录下的 UNet 模型

def normalize(tensor):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    return (tensor - tensor_min) / (tensor_max - tensor_min)

def visualize(input_tensor, output_tensor):
    # 将张量转换为 numpy 数组
    input_image = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    output_image = output_tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()

    # 创建图像对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(input_image, vmin=0, vmax=1)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(output_image, vmin=0, vmax=1)
    axes[1].set_title('Output Image')
    axes[1].axis('off')

    plt.show()

def main():
    # 创建模型实例
    model = UNet()
    model.eval()

    # 创建一个随机输入张量，假设输入尺寸为 3x512x512，值范围在 [0, 1]
    input_tensor = torch.rand(1, 3, 512, 512)

    # 测试维度的正确性
    with torch.no_grad():
        output_tensor = model(input_tensor)
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output_tensor.shape}")

        # 对输出值进行归一化处理
        output_tensor = normalize(output_tensor)

        # 输出输入和输出像素值的范围
        print(f"Input pixel value range: [{input_tensor.min().item()}, {input_tensor.max().item()}]")
        print(f"Output pixel value range: [{output_tensor.min().item()}, {output_tensor.max().item()}]")

    # 可视化输入和输出图像
    visualize(input_tensor, output_tensor)

    # 测试处理时间
    start_time = time.time()
    model(input_tensor)
    end_time = time.time()
    avg_time = (end_time - start_time)
    print(f"Average processing time per forward pass: {avg_time:.6f} seconds")

if __name__ == "__main__":
    main()