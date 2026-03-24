"""
face_keypoints/inference.py
使用训练好的 HeatmapPredictor 对新图像进行关键点推理

特征提取方式与 train_keypoint_final.py 保持一致：
  - Hook StyleGAN2 各层卷积输出
  - 使用 F.interpolate 上采样到 256×256
  - 拼接得到 (1, 5568, 256, 256) 特征图

支持两种模式:
  1. random  : 从随机 Z 采样生成新图像并预测关键点
  2. latent  : 从已有的 projected_w.npz 文件预测关键点

执行示例:
  cd face_keypoints

  # 随机生成 4 张图像并预测关键点
  python inference.py --mode random --num_samples 4 --seed 42

  # 对已有 latent 文件预测
  python inference.py --mode latent --latent_path ./projected/seed7/projected_w.npz

  # 批量处理所有 projected 目录
  python inference.py --mode latent --latent_dir ./projected --output_dir ./inference_results

  # 同时保存热图可视化
  python inference.py --mode random --num_samples 2 --save_heatmaps
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加 StyleGAN2 路径（与 train_keypoint_final.py 保持一致）
_sg2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../stylegan2-ada-pytorch')
if _sg2_path not in sys.path:
    sys.path.insert(0, _sg2_path)

torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

FEATURE_MAP_SIZE = 256   # 与训练时保持一致

# 18 个关键点的颜色
KPT_COLORS = [
    '#FF0000', '#FF6600', '#FFCC00', '#99FF00', '#00FF66',
    '#00FFCC', '#0099FF', '#0033FF', '#6600FF', '#CC00FF',
    '#FF0099', '#FF3366', '#FF9933', '#FFFF00', '#66FF00',
    '#00FF99', '#00CCFF', '#9900FF',
]


# ---------------------------------------------------------------------------
# 模型定义（与 train_keypoint_heatmap.py 保持一致）
# ---------------------------------------------------------------------------

class HeatmapPredictor(nn.Module):
    def __init__(self):
        super().__init__()
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
# StyleGAN2 特征提取（与 train_keypoint_final.py 保持一致）
# ---------------------------------------------------------------------------

def extract_features(G, ws):
    """
    从 W+ latent 提取 StyleGAN2 特征图

    与 train_keypoint_final.py 的特征提取方式完全一致：
      - Hook 各层卷积输出
      - F.interpolate 上采样到 FEATURE_MAP_SIZE×FEATURE_MAP_SIZE
      - 拼接得到 (1, 5568, 256, 256)

    参数:
      G  : StyleGAN2 G_ema
      ws : torch.Tensor (1, 18, 512)

    返回:
      feature_map : torch.Tensor (1, 5568, 256, 256)
      img_np      : np.uint8 (H, W, 3)  生成的 RGB 图像
    """
    feat_list = []
    hooks = []

    def make_hook():
        def _hook(module, inp, out):
            feat_list.append(out.detach().float())
        return _hook

    # Hook all convolutional layers（与 train_keypoint_final.py 相同）
    hooks.append(G.synthesis.b4.conv1.register_forward_hook(make_hook()))
    for res in G.synthesis.block_resolutions[1:]:
        blk = getattr(G.synthesis, f'b{res}')
        hooks.append(blk.conv0.register_forward_hook(make_hook()))
        hooks.append(blk.conv1.register_forward_hook(make_hook()))

    # Forward pass
    with torch.no_grad():
        img_tensor = G.synthesis(ws, noise_mode='const')

    for h in hooks:
        h.remove()

    # 生成图像 → uint8
    img_np = (img_tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
    img_np = img_np[0].cpu().numpy().astype(np.uint8)

    # Upsample all features to FEATURE_MAP_SIZE×FEATURE_MAP_SIZE
    target_size = FEATURE_MAP_SIZE
    upsampled = []
    for feat in feat_list:
        if feat.shape[2] != target_size:
            up = F.interpolate(
                feat,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False,
            )
            upsampled.append(up)
        else:
            upsampled.append(feat)

    # Concatenate all features → (1, 5568, 256, 256)
    combined = torch.cat(upsampled, dim=1)

    return combined, img_np


# ---------------------------------------------------------------------------
# 关键点提取（从热图 argmax）
# ---------------------------------------------------------------------------

def heatmaps_to_keypoints(heatmaps):
    """
    从热图提取关键点坐标

    参数:
      heatmaps : torch.Tensor (1, 18, H, W)

    返回:
      keypoints : np.ndarray (18, 2)  归一化坐标 [0,1]
    """
    hm = heatmaps.detach().cpu().numpy()[0]   # (18, H, W)
    K, H, W = hm.shape
    kpts = np.zeros((K, 2), dtype=np.float32)
    for k in range(K):
        y, x = np.unravel_index(np.argmax(hm[k]), (H, W))
        kpts[k] = [x / W, y / H]
    return kpts


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def visualize_keypoints_on_image(img_np, keypoints, title='Predicted Keypoints', save_path=None):
    """将关键点叠加到图像上并保存"""
    H, W = img_np.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img_np)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    for k, (nx, ny) in enumerate(keypoints):
        px, py = nx * W, ny * H
        color = KPT_COLORS[k % len(KPT_COLORS)]
        ax.scatter(px, py, c=color, s=80, zorder=5,
                   edgecolors='white', linewidths=0.5)
        ax.annotate(str(k), (px, py), fontsize=7, color='white',
                    ha='center', va='center', fontweight='bold', zorder=6)

    ax.set_title(title, fontsize=12)
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  已保存: {save_path}')
    else:
        plt.show()
        plt.close()


def visualize_heatmaps(heatmaps, img_np, save_path=None):
    """可视化 18 个热图（网格布局）"""
    hm = heatmaps.detach().cpu().numpy()[0]   # (18, H, W)
    K = hm.shape[0]
    cols = 6
    rows = (K + cols - 1) // cols + 1

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    axes[0].imshow(img_np)
    axes[0].set_title('Generated Image', fontsize=9)
    axes[0].axis('off')

    for k in range(K):
        axes[k + 1].imshow(hm[k], cmap='hot', vmin=0, vmax=1)
        axes[k + 1].set_title(f'KPT {k}', fontsize=8)
        axes[k + 1].axis('off')

    for i in range(K + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f'  热图已保存: {save_path}')
    else:
        plt.show()
        plt.close()


# ---------------------------------------------------------------------------
# 推理核心
# ---------------------------------------------------------------------------

def run_inference(G, model, ws, output_dir, name, save_heatmaps=False):
    """
    对单个 latent 进行推理并保存结果

    返回:
      keypoints : np.ndarray (18, 2)
      img_np    : np.uint8 (H, W, 3)
    """
    # 提取特征图 + 生成图像
    features, img_np = extract_features(G, ws)   # (1, 5568, 256, 256)

    # 预测热图
    with torch.no_grad():
        pred_hm = model(features)   # (1, 18, 256, 256)

    # 提取关键点
    keypoints = heatmaps_to_keypoints(pred_hm)

    # 保存关键点叠加图
    kpt_save = os.path.join(output_dir, f'{name}_keypoints.png')
    visualize_keypoints_on_image(
        img_np, keypoints,
        title=f'{name}  —  Predicted Keypoints',
        save_path=kpt_save,
    )

    # 可选：保存热图
    if save_heatmaps:
        hm_save = os.path.join(output_dir, f'{name}_heatmaps.png')
        visualize_heatmaps(pred_hm, img_np, save_path=hm_save)

    return keypoints, img_np


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='关键点推理脚本')
    parser.add_argument('--mode',          type=str, default='random',
                        choices=['random', 'latent'],
                        help='推理模式: random=随机生成, latent=从文件加载')
    parser.add_argument('--model_path',    type=str, default='./exp_keypoint/best_model.pth',
                        help='HeatmapPredictor 权重路径')
    parser.add_argument('--stylegan_path', type=str, default='../ffhq.pkl',
                        help='StyleGAN2 模型路径 (.pkl)')
    parser.add_argument('--output_dir',    type=str, default='./inference_results',
                        help='输出目录')
    parser.add_argument('--save_heatmaps', action='store_true',
                        help='是否同时保存热图可视化')

    # random 模式参数
    parser.add_argument('--num_samples',   type=int, default=4,
                        help='[random 模式] 生成样本数量')
    parser.add_argument('--seed',          type=int, default=42,
                        help='[random 模式] 随机种子')
    parser.add_argument('--truncation',    type=float, default=0.7,
                        help='[random 模式] truncation psi')

    # latent 模式参数
    parser.add_argument('--latent_path',   type=str, default='',
                        help='[latent 模式] 单个 projected_w.npz 路径')
    parser.add_argument('--latent_dir',    type=str, default='./projected',
                        help='[latent 模式] 批量处理：projected 目录')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print('\n' + '=' * 60)
    print('🔍  关键点推理')
    print('=' * 60)
    print(f'  模式          : {args.mode}')
    print(f'  模型路径      : {args.model_path}')
    print(f'  StyleGAN2路径 : {args.stylegan_path}')
    print(f'  输出目录      : {args.output_dir}')
    print(f'  设备          : {device}')

    # ── 加载 StyleGAN2 ──────────────────────────────────────────
    print('\n[1/3] 加载 StyleGAN2 生成器...')
    with open(args.stylegan_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)
    G.eval()
    print(f'  G_ema 加载完成  (z_dim={G.z_dim}, w_dim={G.w_dim})')

    # ── 加载 HeatmapPredictor ───────────────────────────────────
    print('\n[2/3] 加载 HeatmapPredictor...')
    model = HeatmapPredictor().to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    loss_key = 'val_loss' if 'val_loss' in ckpt else 'train_loss'
    print(f'  模型加载完成  (best epoch={ckpt["epoch"]}, loss={ckpt[loss_key]:.6f})')

    # ── 推理 ────────────────────────────────────────────────────
    print(f'\n[3/3] 开始推理  (mode={args.mode})...')
    all_keypoints = []
    all_names = []

    if args.mode == 'random':
        # 随机采样 Z → mapping → W+
        rng = np.random.RandomState(args.seed)
        for i in range(args.num_samples):
            z = torch.from_numpy(
                rng.randn(1, G.z_dim).astype(np.float32)
            ).to(device)
            with torch.no_grad():
                ws = G.mapping(z, c=None, truncation_psi=args.truncation)

            name = f'random_seed{args.seed}_sample{i:03d}'
            print(f'  [{i+1}/{args.num_samples}] {name}')
            kpts, _ = run_inference(G, model, ws, args.output_dir, name,
                                    save_heatmaps=args.save_heatmaps)
            all_keypoints.append(kpts)
            all_names.append(name)

    elif args.mode == 'latent':
        if args.latent_path:
            latent_files = [args.latent_path]
            names = [os.path.splitext(os.path.basename(args.latent_path))[0]]
        else:
            latent_files, names = [], []
            for sub in sorted(os.listdir(args.latent_dir)):
                npz = os.path.join(args.latent_dir, sub, 'projected_w.npz')
                if os.path.exists(npz):
                    latent_files.append(npz)
                    names.append(sub)

        for i, (lf, name) in enumerate(zip(latent_files, names)):
            print(f'  [{i+1}/{len(latent_files)}] {name}  ←  {lf}')
            data = np.load(lf)
            ws = torch.from_numpy(data['w']).float().to(device)   # (1, 18, 512)
            kpts, _ = run_inference(G, model, ws, args.output_dir, name,
                                    save_heatmaps=args.save_heatmaps)
            all_keypoints.append(kpts)
            all_names.append(name)

    # ── 保存汇总结果 ─────────────────────────────────────────────
    all_keypoints = np.stack(all_keypoints)   # (N, 18, 2)
    np.save(os.path.join(args.output_dir, 'all_keypoints.npy'), all_keypoints)

    print(f'\n✅  推理完成！')
    print(f'  处理样本数 : {len(all_names)}')
    print(f'  关键点形状 : {all_keypoints.shape}')
    print(f'  结果目录   : {args.output_dir}')
    print(f'  关键点文件 : {args.output_dir}/all_keypoints.npy')



if __name__ == '__main__':
    main()