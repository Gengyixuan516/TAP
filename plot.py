import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import scienceplots  # 可选，用于更科学的绘图风格


def plot_gaussian_logs(log_file_path, save_path=None):
    """
    读取log_gaussians.pth文件并绘制iteration vs anchors的图表

    参数:
        log_file_path: .pth文件路径
        save_path: 图片保存路径（可选）
    """
    try:
        # 加载数据
        log_data = torch.load(log_file_path, map_location='cpu')
        print(f"成功加载日志文件，共 {len(log_data)} 条记录")

        # 提取数据
        iterations = []
        anchors = []
        losses = []  # 如果需要也可以绘制loss

        for log_entry in log_data:
            iterations.append(log_entry['iteration'])
            anchors.append(log_entry['anchors'])
            losses.append(log_entry['loss'])

        # 转换为numpy数组
        iterations = np.array(iterations)
        anchors = np.array(anchors)
        losses = np.array(losses)

        # 设置绘图风格
        plt.style.use(['science', 'ieee', 'grid'])  # 使用scienceplots风格

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]})

        # 绘制anchors vs iteration
        ax1.plot(iterations, anchors,
                 linewidth=2.5,
                 color='#2E86AB',
                 marker='o',
                 markersize=4,
                 markerfacecolor='#F24236',
                 markeredgecolor='white',
                 markeredgewidth=0.5,
                 alpha=0.8)

        # 找到最大anchor数量的点和值
        max_anchor_idx = np.argmax(anchors)
        max_anchor_iter = iterations[max_anchor_idx]
        max_anchor_value = anchors[max_anchor_idx]

        # 标注最大anchor数量点
        ax1.plot(max_anchor_iter, max_anchor_value,
                 'o',
                 markersize=10,
                 markerfacecolor='red',
                 markeredgecolor='white',
                 markeredgewidth=2,
                 alpha=1.0,
                 label=f'Max: {max_anchor_value:,}')


        # 美化第一个子图
        ax1.set_xlabel('Iteration', fontsize=35, fontweight='bold')
        ax1.set_ylabel('Number of Anchors', fontsize=35, fontweight='bold')
        ax1.set_title('Evolution of Gaussian Anchors during Training',
                      fontsize=36, fontweight='bold', pad=50)


        # 设置网格
        ax1.grid(True, alpha=0.3, linestyle='--')

        # 格式化y轴，显示整数
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        # 设置刻度标签字体大小
        ax1.tick_params(axis='both', which='major', labelsize=26)  # 添加这一行

        # 添加数据标签在某些关键点
        if len(iterations) > 0:
            # 在起点和终点添加标签
            ax1.annotate(f'{anchors[0]:,}',
                         xy=(iterations[0], anchors[0]),
                         xytext=(10, 10),
                         textcoords='offset points',
                         fontsize=15,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

            ax1.annotate(f'{anchors[-1]:,}',
                         xy=(iterations[-1], anchors[-1]),
                         xytext=(-30, 10),
                         textcoords='offset points',
                         fontsize=15,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        ax1.annotate(f'Max Anchors: {max_anchor_value:}',
                     xy=(max_anchor_iter, max_anchor_value),
                     xytext=(20, 30),
                     textcoords='offset points',
                     fontsize=26,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                     arrowprops=dict(arrowstyle='->',
                                     connectionstyle='arc3,rad=0.3',
                                     color='red',
                                     lw=1.5))

        # 绘制loss vs iteration（可选）
        ax2.plot(iterations, losses,
                 linewidth=2,
                 color='#F24236',
                 alpha=0.8)

        ax2.set_xlabel('Iteration', fontsize=34, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=34, fontweight='bold')
        ax2.set_title('Training Loss', fontsize=36, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_yscale('log')  # 通常loss用对数尺度更好看
        # 设置刻度标签字体大小
        ax2.tick_params(axis='both', which='major', labelsize=26)  # 添加这一行

        # 调整布局
        plt.tight_layout()

        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"图表已保存至: {save_path}")

        # plt.show()

        # 打印一些统计信息
        print(f"\n统计信息:")
        print(f"迭代次数范围: {iterations[0]} - {iterations[-1]}")
        print(f"初始anchors数量: {anchors[0]:,}")
        print(f"最终anchors数量: {anchors[-1]:,}")
        print(f"anchors数量变化: {anchors[-1] - anchors[0]:+,}")
        print(f"最大anchors数量: {anchors.max():,}")
        print(f"最小anchors数量: {anchors.min():,}")

    except Exception as e:
        print(f"加载或绘图时出错: {e}")


def plot_simple_gaussian_logs(log_file_path, save_path=None):
    """
    简化版本的绘图函数（如果不安装scienceplots）
    """
    try:
        # 加载数据
        log_data = torch.load(log_file_path, map_location='cpu')
        print(f"成功加载日志文件，共 {len(log_data)} 条记录")

        # 提取数据
        iterations = []
        anchors = []

        for log_entry in log_data:
            iterations.append(log_entry['iteration'])
            anchors.append(log_entry['anchors'])

        # 创建图表
        plt.figure(figsize=(16, 8))

        # 绘制图表
        plt.plot(iterations, anchors,
                 linewidth=3,
                 color='steelblue',
                 marker='o',
                 markersize=6,
                 markerfacecolor='red',
                 markeredgecolor='white',
                 markeredgewidth=1,
                 alpha=0.8)

        # 美化图表
        plt.xlabel('Iteration', fontsize=34, fontweight='bold')
        plt.ylabel('Number of Anchors', fontsize=34, fontweight='bold')
        plt.title('Evolution of Gaussian Anchors during Training',
                  fontsize=36, fontweight='bold', pad=30)

        plt.grid(True, alpha=0.3, linestyle='--')

        # 格式化y轴
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

        # 添加数据标签
        if len(iterations) > 0:
            plt.annotate(f'Start: {anchors[0]:,}',
                         xy=(iterations[0], anchors[0]),
                         xytext=(20, 20),
                         textcoords='offset points',
                         fontsize=15,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
                         arrowprops=dict(arrowstyle='->', color='black'))

            plt.annotate(f'End: {anchors[-1]:,}',
                         xy=(iterations[-1], anchors[-1]),
                         xytext=(-80, 20),
                         textcoords='offset points',
                         fontsize=26,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
                         arrowprops=dict(arrowstyle='->', color='black'))

        plt.tight_layout()

        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"图表已保存至: {save_path}")

        plt.show()

    except Exception as e:
        print(f"加载或绘图时出错: {e}")


# 使用示例
if __name__ == "__main__":
    # 替换为您的实际文件路径
    log_file_path = "D:/code/Scaffold-GS-main/Scaffold-GS-main/output_ori_A_B_C/server_result0829/playroom/log_gaussians.pth"

    # 尝试使用精美版本（需要安装scienceplots）
    try:
        import scienceplots

        plot_gaussian_logs(log_file_path, save_path="gaussian_anchors_plot.png")
    except ImportError:
        print("scienceplots 未安装，使用简化版本")
        plot_simple_gaussian_logs(log_file_path, save_path="gaussian_anchors_plot_simple.png")