"""
关键点预测和可视化脚本
使用训练好的模型从RGB图像预测关键点，并与真实关键点进行对比
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import argparse

# 添加StyleGAN2路径
sys.path.append('../stylegan2-ada-pytorch')

torch.manual_seed(0)
np.random.seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# 模型定义 (与训练脚本相同)
# ---------------------------------------------------------------------------

class HeatmapPredictor(nn.Module):
    """基于 StyleGAN2 特征图的关键点热图预测模型"""
    def __init__(self, verbose=False):
        super().__init__()

        if verbose:
            print('构建 HeatmapPredictor 模型')

        self.step1_channel_reduce = nn.Sequential(
            nn.Conv2d(5568, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        self.step2_downsample1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.1),
        )
        self.step3_downsample2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1),
        )
        self.step4_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.step5_output = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 18, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.step1_channel_reduce(x)
        x = self.step2_downsample1(x)
        x = self.step3_downsample2(x)
        x = self.step4_upsample(x)
        x = self.step5_output(x)
        return x


# ---------------------------------------------------------------------------
# StyleGAN2 特征提取
# ---------------------------------------------------------------------------

def _build_upsamplers(G, target_dim, mode):
    """构建上采样器"""
    upsamplers = []

    for res in G.synthesis.block_resolutions:
        scale = target_dim / res
        if scale == 1.0:
            up = nn.Identity()
        else:
            if mode in ('bilinear', 'bicubic'):
                up = nn.Upsample(scale_factor=scale, mode=mode, align_corners=False)
            else:
                up = nn.Upsample(scale_factor=scale, mode=mode)

        if res == 4:
            upsamplers.append(up)          # only conv1 for the first block
        else:
            if scale == 1.0:
                up2 = nn.Identity()
            else:
                if mode in ('bilinear', 'bicubic'):
                    up2 = nn.Upsample(scale_factor=scale, mode=mode, align_corners=False)
                else:
                    up2 = nn.Upsample(scale_factor=scale, mode=mode)
            upsamplers.append(up)          # conv0
            upsamplers.append(up2)         # conv1
    return upsamplers


def load_stylegan2_model(stylegan_path):
    """加载StyleGAN2模型"""
    print(f'加载StyleGAN2模型: {stylegan_path}')
    with open(stylegan_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)
    G.eval()
    return G


def extract_features_from_latent(G, ws, target_dim=256, upsample_mode='bilinear'):
    """
    从latent向量提取特征图

    Parameters
    ----------
    G : StyleGAN2生成器
    ws : torch.Tensor, shape (batch, num_ws, 512) — W-space latent codes
    target_dim : int — 目标空间分辨率

    Returns
    -------
    features : torch.Tensor (batch, 5568, target_dim, target_dim)
    """
    upsamplers = _build_upsamplers(G, target_dim, upsample_mode)

    features = []
    hooks = []

    def make_hook():
        def _hook(module, inp, out):
            features.append(out.detach().float())
        return _hook

    # b4 has no conv0 — only hook conv1
    hooks.append(G.synthesis.b4.conv1.register_forward_hook(make_hook()))

    # All other blocks: hook conv0 then conv1
    for res in G.synthesis.block_resolutions[1:]:
        blk = getattr(G.synthesis, f'b{res}')
        hooks.append(blk.conv0.register_forward_hook(make_hook()))
        hooks.append(blk.conv1.register_forward_hook(make_hook()))

    with torch.no_grad():
        _ = G.synthesis(ws, noise_mode='const')

    for h in hooks:
        h.remove()

    # 上采样每个特征图到(target_dim × target_dim)并沿通道轴连接
    total_ch = sum(f.shape[1] for f in features)
    aff = torch.zeros(ws.shape[0], total_ch, target_dim, target_dim, device=device)
    start = 0
    for i, feat in enumerate(features):
        c = feat.shape[1]
        aff[:, start:start + c] = upsamplers[i](feat)
        start += c

    return aff


# ---------------------------------------------------------------------------
# 关键点提取
# ---------------------------------------------------------------------------

def extract_keypoints_from_heatmaps(heatmaps):
    """从热图中提取关键点坐标"""
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().cpu().numpy()
    B, K, H, W = heatmaps.shape
    keypoints = np.zeros((B, K, 2), dtype=np.float32)
    for b in range(B):
        for k in range(K):
            hmap = heatmaps[b, k]
            y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
            keypoints[b, k] = [x / W, y / H]
    return keypoints


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def visualize_keypoints_comparison(image_path, gt_keypoints, pred_keypoints, save_path):
    """
    可视化真实关键点和预测关键点的对比

    Parameters
    ----------
    image_path : str — RGB图像路径
    gt_keypoints : np.array (18, 2) — 真实关键点，归一化坐标
    pred_keypoints : np.array (18, 2) — 预测关键点，归一化坐标
    save_path : str — 保存路径
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f'无法读取图像: {image_path}')
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # 创建左右对比的图像
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 左图：真实关键点
    axes[0].imshow(img)
    axes[0].set_xlim(0, w)
    axes[0].set_ylim(h, 0)  # 图像坐标系
    axes[0].scatter(gt_keypoints[:, 0] * w, gt_keypoints[:, 1] * h,
                   c='blue', s=60, zorder=3, alpha=0.8)
    for j, (x, y) in enumerate(gt_keypoints):
        axes[0].annotate(str(j), (x * w, y * h), fontsize=8, color='navy',
                        ha='center', va='center')
    axes[0].set_title('Ground Truth Keypoints', fontsize=12)
    axes[0].axis('off')

    # 右图：预测关键点
    axes[1].imshow(img)
    axes[1].set_xlim(0, w)
    axes[1].set_ylim(h, 0)  # 图像坐标系
    axes[1].scatter(pred_keypoints[:, 0] * w, pred_keypoints[:, 1] * h,
                   c='red', s=60, zorder=3, alpha=0.8)
    for j, (x, y) in enumerate(pred_keypoints):
        axes[1].annotate(str(j), (x * w, y * h), fontsize=8, color='darkred',
                        ha='center', va='center')
    axes[1].set_title('Predicted Keypoints', fontsize=12)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'对比图像已保存: {save_path}')


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='关键点预测和可视化')
    parser.add_argument('--model_path', type=str, default='./exp_keypoint/best_model.pth',
                       help='训练好的模型权重路径')
    parser.add_argument('--stylegan_path', type=str, default='../ffhq.pkl',
                       help='StyleGAN2模型路径')
    parser.add_argument('--keypoints_path', type=str, default='./keypoints.npy',
                       help='真实关键点文件路径')
    parser.add_argument('--projected_dir', type=str, default='./projected',
                       help='projected图像目录')
    parser.add_argument('--output_dir', type=str, default='./prediction_results',
                       help='输出目录')
    parser.add_argument('--seed_ids', type=str, default='0,2,3,4,7,12,68,70,76,135,160,250,251,344,439,543',
                       help='要处理的seed ID列表，用逗号分隔')
    args = parser.parse_args()

    # 解析seed IDs
    seed_ids = [s.strip() for s in args.seed_ids.split(',')]

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 60)
    print('🚀 关键点预测和可视化')
    print('=' * 60)
    print(f'模型路径: {args.model_path}')
    print(f'StyleGAN2路径: {args.stylegan_path}')
    print(f'关键点文件: {args.keypoints_path}')
    print(f'Projected目录: {args.projected_dir}')
    print(f'输出目录: {args.output_dir}')
    print(f'处理seed IDs: {seed_ids}')
    print(f'设备: {device}')

    # 加载真实关键点
    print('\n[1/5] 加载真实关键点...')
    gt_keypoints_all = np.load(args.keypoints_path)
    print(f'真实关键点形状: {gt_keypoints_all.shape}')

    # 加载StyleGAN2模型
    print('\n[2/5] 加载StyleGAN2模型...')
    G = load_stylegan2_model(args.stylegan_path)

    # 加载训练好的模型
    print('\n[3/5] 加载训练好的关键点预测模型...')
    model = HeatmapPredictor(verbose=True).to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('模型加载完成')

    # 处理每个seed
    print('\n[4/5] 开始预测关键点...')
    all_pred_keypoints = []

    for i, seed_id in enumerate(seed_ids):
        print(f'  处理 seed {seed_id} ({i+1}/{len(seed_ids)})...')

        # 加载latent向量
        latent_path = os.path.join(args.projected_dir, f'seed{seed_id}', 'projected_w.npz')
        if not os.path.exists(latent_path):
            print(f'    警告: latent文件不存在: {latent_path}')
            continue

        latent_data = np.load(latent_path)
        ws = torch.from_numpy(latent_data['w']).float().to(device)  # (1, 18, 512)

        # 提取特征图
        features = extract_features_from_latent(G, ws, target_dim=256)

        # 预测热图
        with torch.no_grad():
            pred_heatmaps = model(features)

        # 从热图提取关键点
        pred_keypoints = extract_keypoints_from_heatmaps(pred_heatmaps)
        pred_keypoints = pred_keypoints[0]  # 去掉batch维度

        all_pred_keypoints.append(pred_keypoints)

        # 读取对应的真实关键点
        gt_keypoints = gt_keypoints_all[i]

        # 可视化对比
        image_path = os.path.join(args.projected_dir, f'seed{seed_id}', 'target.png')
        save_path = os.path.join(args.output_dir, f'keypoints_comparison_seed{seed_id}.png')

        if os.path.exists(image_path):
            visualize_keypoints_comparison(image_path, gt_keypoints, pred_keypoints, save_path)
        else:
            print(f'    警告: 图像文件不存在: {image_path}')

    # 保存所有预测结果
    all_pred_keypoints = np.array(all_pred_keypoints)
    pred_save_path = os.path.join(args.output_dir, 'predicted_keypoints.npy')
    np.save(pred_save_path, all_pred_keypoints)
    print(f'\n  所有预测关键点已保存: {pred_save_path}')

    # 计算误差统计
    print('\n[5/5] 计算预测误差...')
    errors = np.linalg.norm(all_pred_keypoints - gt_keypoints_all[:len(all_pred_keypoints)], axis=-1)
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    print(f'平均欧氏距离误差 (归一化): {mean_error:.4f}')
    print(f'误差标准差 (归一化): {std_error:.4f}')
    print(f'平均欧氏距离误差 (像素): {mean_error * 256:.2f} px')
    print(f'误差标准差 (像素): {std_error * 256:.2f} px')

    # 保存误差统计
    stats_path = os.path.join(args.output_dir, 'prediction_stats.txt')
    with open(stats_path, 'w') as f:
        f.write(f'平均欧氏距离误差 (归一化): {mean_error:.4f}\n')
        f.write(f'误差标准差 (归一化): {std_error:.4f}\n')
        f.write(f'平均欧氏距离误差 (像素): {mean_error * 256:.2f} px\n')
        f.write(f'误差标准差 (像素): {std_error * 256:.2f} px\n')
    print(f'误差统计已保存: {stats_path}')

    print(f'\n✅ 预测完成！结果保存在: {args.output_dir}')


if __name__ == '__main__':
    main()