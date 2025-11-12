import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import scienceplots  # 可选，用于更科学的绘图风格


def plot_compare_gaussian_logs(log_file_paths, labels, colors=None, save_path=None):
    """
    读取多个log_gaussians.pth文件并在同一个图表中绘制比较

    参数:
        log_file_paths: .pth文件路径列表
        labels: 对应的标签列表
        colors: 颜色列表（可选）
        save_path: 图片保存路径（可选）
    """
    if colors is None:
        colors = ['#2E86AB', '#F24236', '#3498db', '#FF9F1C', '#5D5C61']

    if len(log_file_paths) != len(labels):
        raise ValueError("log_file_paths和labels的长度必须相同")

    try:
        # 设置绘图风格
        plt.style.use(['science', 'ieee', 'grid'])

        # 创建图表
        fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))

        max_values = []  # 存储每组数据的最大值信息

        for i, (log_file_path, label) in enumerate(zip(log_file_paths, labels)):
            # 加载数据
            log_data = torch.load(log_file_path, map_location='cpu')
            print(f"成功加载日志文件 {label}: {len(log_data)} 条记录")

            # 提取数据
            iterations = []
            anchors = []

            for log_entry in log_data:
                iterations.append(log_entry['iteration'])
                anchors.append(log_entry['anchors'])

            # 转换为numpy数组
            iterations = np.array(iterations)
            anchors = np.array(anchors)

            # 绘制anchors vs iteration
            line = ax1.plot(iterations, anchors,
                            linewidth=3,
                            color=colors[i % len(colors)],
                            marker='o',
                            markersize=5,
                            markerfacecolor=colors[i % len(colors)],
                            markeredgecolor='white',
                            markeredgewidth=1,
                            alpha=0.8,
                            label=label)

            # 找到最大anchor数量的点和值
            max_anchor_idx = np.argmax(anchors)
            max_anchor_iter = iterations[max_anchor_idx]
            max_anchor_value = anchors[max_anchor_idx]
            max_values.append((max_anchor_iter, max_anchor_value, label, colors[i % len(colors)]))

            # 标注最大点
            ax1.plot(max_anchor_iter, max_anchor_value,
                     'o',
                     markersize=10,
                     markerfacecolor=colors[i % len(colors)],
                     markeredgecolor='white',
                     markeredgewidth=2,
                     alpha=1.0)

            # 打印统计信息
            print(f"{label}统计信息:")
            print(f"  迭代次数范围: {iterations[0]} - {iterations[-1]}")
            print(f"  初始anchors数量: {anchors[0]:,}")
            print(f"  最终anchors数量: {anchors[-1]:,}")
            print(f"  最大anchors数量: {max_anchor_value:,} (迭代 {max_anchor_iter})")
            print(f"  anchors数量变化: {anchors[-1] - anchors[0]:+,}")
            print()

        # 美化图表
        ax1.set_xlabel('Iteration', fontsize=35, fontweight='bold')
        ax1.set_ylabel('Number of Anchors', fontsize=35, fontweight='bold')
        ax1.set_title('Comparison of Gaussian Anchors Evolution',
                      fontsize=36, fontweight='bold', pad=45)

        # 设置网格
        ax1.grid(True, alpha=0.3, linestyle='--')

        # 格式化y轴，显示整数
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

        # 设置刻度标签字体大小
        ax1.tick_params(axis='both', which='major', labelsize=24)  # 添加这一行

        # 添加图例
        ax1.legend(loc='best', fontsize=24)

        # 为每个最大点添加标注（智能避开重叠）
        for j, (max_iter, max_value, label, color) in enumerate(max_values):
            # 智能选择标注位置
            if j % 2 == 0:
                xytext = (30, 30)
                connection_style = 'arc3,rad=0.2'
            else:
                xytext = (-30, 30)
                connection_style = 'arc3,rad=-0.2'

            ax1.annotate(f'{label} Max: {max_value:}',
                         xy=(max_iter, max_value),
                         xytext=xytext,
                         textcoords='offset points',
                         fontsize=26,
                         fontweight='bold',
                         color=color,
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9),
                         arrowprops=dict(arrowstyle='->',
                                         connectionstyle=connection_style,
                                         color=color,
                                         lw=1.5))

        # 调整布局
        plt.tight_layout()

        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"比较图表已保存至: {save_path}")

        # plt.show()

    except Exception as e:
        print(f"加载或绘图时出错: {e}")


def plot_simple_compare(log_file_paths, labels, colors=None, save_path=None):
    """
    简化版本的比较绘图函数
    """
    if colors is None:
        colors = ['blue', 'red', 'green', 'orange', 'purple']

    try:
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))

        for i, (log_file_path, label) in enumerate(zip(log_file_paths, labels)):
            # 加载数据
            log_data = torch.load(log_file_path, map_location='cpu')

            # 提取数据
            iterations = []
            anchors = []

            for log_entry in log_data:
                iterations.append(log_entry['iteration'])
                anchors.append(log_entry['anchors'])

            # 绘制
            ax1.plot(iterations, anchors,
                     linewidth=2.5,
                     color=colors[i % len(colors)],
                     marker='o',
                     markersize=4,
                     label=label)

        # 美化图表
        ax1.set_xlabel('Iteration', fontsize=35, fontweight='bold')
        ax1.set_ylabel('Number of Anchors', fontsize=35, fontweight='bold')
        ax1.set_title('Comparison of Gaussian Anchors Evolution',
                      fontsize=36, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=26)  # 添加字体大小设置
        ax1.legend()

        # 格式化y轴
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    except Exception as e:
        print(f"加载或绘图时出错: {e}")


# 使用示例
if __name__ == "__main__":
    # 替换为您的实际文件路径
    log_file_paths = [
        "D:/code/Scaffold-GS-main/Scaffold-GS-main/output_ori_A_B_C/server_result0829/playroom/log_gaussians.pth",
        "D:/code/Scaffold-GS-main/Scaffold-GS-original/output/playroom/log_gaussians.pth"  # 第二个文件
    ]

    labels = [
        "Ours",
        "Scaffold-GS"
    ]

    colors = ['#1f77b4', '#ff7f0e']  # 可选自定义颜色

    # 尝试使用精美版本
    try:
        import scienceplots

        plot_compare_gaussian_logs(log_file_paths, labels, colors,
                                   save_path="comparison_plot.png")
    except ImportError:
        print("scienceplots 未安装，使用简化版本")
        plot_simple_compare(log_file_paths, labels, colors,
                            save_path="comparison_plot_simple.png")