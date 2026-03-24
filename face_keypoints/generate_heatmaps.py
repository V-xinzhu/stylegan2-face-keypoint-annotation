"""
将 keypoints.npy (N, 18, 2) 转换为高斯热图 heatmaps.npy (N, 18, 256, 256)
预先生成，避免训练时实时计算导致 CPU 瓶颈
"""
import numpy as np

def make_heatmap(height=256, width=256, cx=0.0, cy=0.0, sigma=10.0):
    """生成单关键点的高斯热图"""
    y_grid, x_grid = np.ogrid[:height, :width]
    dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
    hmap = np.exp(-dist_sq / (2 * sigma ** 2))
    max_v = hmap.max()
    if max_v > 0:
        hmap /= max_v
    return hmap.astype(np.float32)

if __name__ == '__main__':
    keypoints = np.load('keypoints.npy')   # (N, 18, 2)
    print(f'Loaded keypoints: {keypoints.shape}')

    N, K, _ = keypoints.shape
    H = W = 256
    sigma = 10.0

    heatmaps = np.zeros((N, K, H, W), dtype=np.float32)

    for i in range(N):
        for j in range(K):
            cx = keypoints[i, j, 0] * W   # 归一化 x → 像素 x
            cy = keypoints[i, j, 1] * H   # 归一化 y → 像素 y
            heatmaps[i, j] = make_heatmap(H, W, cx, cy, sigma)

        if (i + 1) % 4 == 0:
            print(f'  Generated {i+1}/{N}')

    np.save('heatmaps.npy', heatmaps)
    print(f'Saved heatmaps: {heatmaps.shape}  ({heatmaps.nbytes / 1024**2:.1f} MB)')