"""
face_keypoints/train_keypoint_heatmap.py
基于 StyleGAN2 特征图的关键点热图预测训练脚本

取消训练集/验证集划分，整个数据集都用于训练
"""

import os
import gc
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ORIGIN_PIC_SIZE = 1024.0
torch.manual_seed(0)
np.random.seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Step 1: 数据集定义
# ---------------------------------------------------------------------------

class KeypointDataset(Dataset):
    """关键点数据集"""
    def __init__(self, features_path, heatmaps_path):
        print('\n[Dataset] 加载数据...')
        self.features = np.load(features_path,  mmap_mode='r')
        self.heatmaps = np.load(heatmaps_path,  mmap_mode='r')
        assert len(self.features) == len(self.heatmaps), \
            f'features ({len(self.features)}) 与 heatmaps ({len(self.heatmaps)}) 数量不一致'
        print(f'  features  : {self.features.shape}  dtype={self.features.dtype}')
        print(f'  heatmaps  : {self.heatmaps.shape}  dtype={self.heatmaps.dtype}')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = torch.from_numpy(self.features[idx].astype(np.float32))
        hmap = torch.from_numpy(self.heatmaps[idx].astype(np.float32))
        return feat, hmap


# ---------------------------------------------------------------------------
# Step 2: 模型定义
# ---------------------------------------------------------------------------

class HeatmapPredictor(nn.Module):
    """基于 StyleGAN2 特征图的关键点热图预测模型"""
    def __init__(self, verbose=True):
        super().__init__()

        if verbose:
            print('\n' + '=' * 60)
            print('🔧  构建 HeatmapPredictor 模型')
            print('=' * 60)

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

        if verbose:
            total_p     = sum(p.numel() for p in self.parameters())
            trainable_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'\n  总参数量     : {total_p:,}')
            print(f'  可训练参数量 : {trainable_p:,}\n')

    def forward(self, x):
        x = self.step1_channel_reduce(x)
        x = self.step2_downsample1(x)
        x = self.step3_downsample2(x)
        x = self.step4_upsample(x)
        x = self.step5_output(x)
        return x


# ---------------------------------------------------------------------------
# Step 3: 关键点提取（从热图 argmax）
# ---------------------------------------------------------------------------

def extract_keypoints_from_heatmaps(heatmaps):
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
# Step 4: 可视化
# ---------------------------------------------------------------------------

def visualize_results(gt_keypoints, pred_keypoints, save_path, num_samples=4):
    n = min(num_samples, len(gt_keypoints))
    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for i in range(n):
        gt_px   = gt_keypoints[i]   * ORIGIN_PIC_SIZE
        pred_px = pred_keypoints[i] * ORIGIN_PIC_SIZE
        ax = axes[i, 0]
        ax.set_xlim(0, ORIGIN_PIC_SIZE); ax.set_ylim(ORIGIN_PIC_SIZE, 0)
        ax.scatter(gt_px[:, 0], gt_px[:, 1], c='blue', s=60, zorder=3)
        for j, (x, y) in enumerate(gt_px):
            ax.annotate(str(j), (x, y), fontsize=7, color='navy')
        ax.set_title(f'Sample {i}  —  Ground Truth', fontsize=11)
        ax.set_facecolor('#f0f0f0')

        ax = axes[i, 1]
        ax.set_xlim(0, ORIGIN_PIC_SIZE); ax.set_ylim(ORIGIN_PIC_SIZE, 0)
        ax.scatter(pred_px[:, 0], pred_px[:, 1], c='red', s=60, zorder=3)
        for j, (x, y) in enumerate(pred_px):
            ax.annotate(str(j), (x, y), fontsize=7, color='darkred')
        ax.set_title(f'Sample {i}  —  Predicted', fontsize=11)
        ax.set_facecolor('#f0f0f0')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  可视化结果已保存: {save_path}')


# ---------------------------------------------------------------------------
# Step 5: 训练主函数
# ---------------------------------------------------------------------------

def main(args):
    print('\n' + '=' * 60)
    print('🚀  关键点热图预测训练')
    print('=' * 60)
    for k, v in args.items():
        print(f'  {k}: {v}')
    print(f'  device: {device}')

    os.makedirs(args['exp_dir'], exist_ok=True)

    # ── [1/5] 准备数据集 ─────────────────────────────────────────
    print('\n[1/5] 准备数据集...')
    dataset = KeypointDataset(
        features_path = args['features_path'],
        heatmaps_path = args['heatmaps_path'],
    )
    train_loader = DataLoader(dataset, batch_size= args['batch_size'] if args['batch_size'] <7 else 6,
                              shuffle=True, num_workers=0)
    print(f'  数据集总样本数: {len(dataset)} 样本  （全部用于训练）')

    # ── [2/5] 构建模型 ───────────────────────────────────────────
    print('\n[2/5] 构建模型...')
    model = HeatmapPredictor(verbose=True).to(device)

    # ── [3/5] 损失函数 & 优化器 ──────────────────────────────────
    print('\n[3/5] 配置损失函数和优化器...')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=20, factor=0.5, verbose=True)

    # ── [4/5] 训练循环 ───────────────────────────────────────────
    print('\n[4/5] 开始训练...')
    best_train_loss = float('inf')
    patience_counter = 0
    patience = args.get('patience', 50)
    train_losses = []

    for epoch in range(args['epochs']):
        model.train()
        train_loss = 0.0
        train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['epochs']} [Train]", leave=False)
        for feats, hm in train_loader_iter:

            feats, hm = feats.to(device), hm.to(device)
            optimizer.zero_grad()
            pred = model(feats)
            loss = criterion(pred, hm)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loader_iter.set_postfix({"batch_loss": f"{loss.item():.6f}"})
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        scheduler.step(train_loss)

        print(f'Epoch {epoch+1:4d}/{args["epochs"]} | Train Loss: {train_loss:.6f} | Best Train Loss: {best_train_loss:.6f}')

        # 保存损失曲线
        if (epoch + 1) % 10 == 0:
            plt.figure(figsize=(10, 4))
            plt.plot(train_losses, label='Train Loss', color='steelblue')
            plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
            plt.title(f'Training Curve  (epoch {epoch+1})')
            plt.legend(); plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args['exp_dir'], 'loss_curve.png'), dpi=150)
            plt.close()

        # 早停 & 保存最佳模型
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss
            }, os.path.join(args['exp_dir'], 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\n  早停: epoch {epoch+1}, best_train_loss={best_train_loss:.6f}')
                break

        gc.collect()
        torch.cuda.empty_cache()

    # 保存最终损失曲线
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title('Training Curve'); plt.legend()
    plt.savefig(os.path.join(args['exp_dir'], 'loss_curve.png'), dpi=150)
    plt.close()
    print(f'  损失曲线已保存: {args["exp_dir"]}/loss_curve.png')

    # ── [5/5] 评估 ──────────────────────────────────────────────
    print('\n[5/5] 评估模型...')
    ckpt = torch.load(os.path.join(args['exp_dir'], 'best_model.pth'),
                      map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    full_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    gt_kpts_all = np.load(args['keypoints_path'])   # (N, 18, 2)
    all_pred_kpts = []

    with torch.no_grad():
        for feats, _ in full_loader:
            feats = feats.to(device)
            pred_hm = model(feats)
            pred_kpts = extract_keypoints_from_heatmaps(pred_hm)
            all_pred_kpts.append(pred_kpts)

    all_pred_kpts = np.concatenate(all_pred_kpts, axis=0)   # (N, 18, 2)
    errors     = np.linalg.norm(all_pred_kpts - gt_kpts_all, axis=-1)
    mean_error = np.mean(errors)
    print(f'  平均欧氏距离误差 (归一化) : {mean_error:.4f}')
    print(f'  平均欧氏距离误差 (像素)   : {mean_error * ORIGIN_PIC_SIZE:.2f} px')

    pred_save = os.path.join(args['exp_dir'], 'predicted_keypoints.npy')
    np.save(pred_save, all_pred_kpts)
    print(f'  预测关键点已保存: {pred_save}')

    visualize_results(
        gt_keypoints   = gt_kpts_all,
        pred_keypoints = all_pred_kpts,
        save_path      = os.path.join(args['exp_dir'], 'evaluation_results.png'),
        num_samples    = 4,
    )

    print(f'\n✅  训练完成！所有结果保存在: {args["exp_dir"]}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='关键点热图预测训练')
    parser.add_argument('--features',   type=str,   default='features.npy')
    parser.add_argument('--heatmaps',   type=str,   default='heatmaps.npy')
    parser.add_argument('--keypoints',  type=str,   default='keypoints.npy')
    parser.add_argument('--exp_dir',    type=str,   default='./exp_keypoint')
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch_size', type=int,   default=2)
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--patience',   type=int,   default=50)
    args = parser.parse_args()

    config = {
        'features_path' : args.features,
        'heatmaps_path' : args.heatmaps,
        'keypoints_path': args.keypoints,
        'exp_dir'       : args.exp_dir,
        'epochs'        : args.epochs,
        'batch_size'    : args.batch_size,
        'lr'            : args.lr,
        'patience'      : args.patience,
    }

    main(config)