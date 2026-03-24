"""
检查 keypoints.npy 文件的数据结构和值范围
"""

import numpy as np
import os

def inspect_keypoints(keypoints_path):
    """检查关键点数据文件"""
    print("=" * 60)
    print("检查关键点数据文件")
    print("=" * 60)

    # 检查文件是否存在
    if not os.path.exists(keypoints_path):
        print(f"文件不存在: {keypoints_path}")
        return

    # 加载数据
    print(f"加载文件: {keypoints_path}")
    keypoints = np.load(keypoints_path)

    # 基本信息
    print("\n基本信息:")
    print(f"  形状 (shape): {keypoints.shape}")
    print(f"  数据类型 (dtype): {keypoints.dtype}")
    print(f"  总元素数: {keypoints.size}")
    print(".2f")

    # 维度分析
    if len(keypoints.shape) == 3:
        n_samples, n_keypoints, n_coords = keypoints.shape
        print("\n维度分析:")
        print(f"  样本数量: {n_samples}")
        print(f"  关键点数量: {n_keypoints}")
        print(f"  坐标维度: {n_coords}")

        # 坐标范围分析
        print("\n坐标范围分析:")
        print(".6f")
        print(".6f")
        print(".6f")

        # 判断是否为归一化坐标
        if keypoints.max() <= 1.0 and keypoints.min() >= 0.0:
            print("  坐标范围在 [0, 1] 之间，可能是归一化坐标")
        elif keypoints.max() <= 2.0 and keypoints.min() >= 0.0:
            print("  坐标范围在 [0, 2] 之间，可能需要检查")
        else:
            print("  坐标范围超出预期，可能为像素坐标")

        # 统计信息
        print("\n统计信息:")
        print(".6f")
        print(".6f")
        print(".6f")
        print(".6f")

        # 样本查看
        print("\n样本数据预览:")
        for i in range(min(3, n_samples)):
            print(f"  样本 {i}:")
            for j in range(min(5, n_keypoints)):  # 只显示前5个关键点
                x, y = keypoints[i, j]
                print(".4f")
            if n_keypoints > 5:
                print(f"    ... (还有 {n_keypoints } 个关键点)")
            print()

    else:
        print(f"意外的维度数量: {len(keypoints.shape)}，期望3维 (N, K, 2)")

    print("=" * 60)

if __name__ == '__main__':
    # 检查当前目录下的 keypoints.npy
    keypoints_path = './keypoints.npy'
    inspect_keypoints(keypoints_path)
