"""
18 点人脸关键点热图预测训练 - 懒加载版本
按需处理特征提取，减少 RAM 占用，充分利用 VRAM
"""
import os
import sys
import json
import gc
import re
import shutil
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

# Constants
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ORIGIN_PIC_SIZE = 1024.0
FEATURE_MAP_SIZE = 256
KEYPOINT_ORDER = [
    "chin", "left_eye_inner", "left_eye_outer", "left_eye_pupil",
    "left_eyebrow_inner", "left_eyebrow_outer", "lower_lip", "mouth_left",
    "mouth_right", "nose_left", "nose_right", "nose_tip", "right_eye_inner",
    "right_eye_outer", "right_eyebrow_outer", "right_eyebrow_inner",
    "upper_lip", "right_eye_pupil",
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
np.random.seed(0)


def make_grad_scaler():
    """使用新 AMP API，并兼容仍使用 torch.cuda.amp 的 PyTorch。"""
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    return torch.cuda.amp.GradScaler(enabled=device == 'cuda')


def autocast_context():
    if device != 'cuda':
        return nullcontext()
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast('cuda')
    return torch.cuda.amp.autocast()


def natural_sort_key(name):
    # 类型标记保证数字、文字开头的文件名也可以相互比较。
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r'(\d+)', name))


class LazyDataProcessor:
    """懒加载数据处理器 - 只保存路径和关键点，不预先上采样"""

    def __init__(self, annotation_dir=REPO_ROOT / 'face_keypoint_annotation',
                 projected_dir=SCRIPT_DIR / 'projected',
                 generator_path=REPO_ROOT / 'ffhq.pkl',
                 stylegan_path=REPO_ROOT / 'stylegan2-ada-pytorch'):
        self.annotation_dir = os.fspath(Path(annotation_dir).expanduser().resolve())
        self.projected_dir = os.fspath(Path(projected_dir).expanduser().resolve())
        self.generator_path = os.fspath(Path(generator_path).expanduser().resolve())
        self.stylegan_path = os.fspath(Path(stylegan_path).expanduser().resolve())
        self.device = device
        self.G = None

    def get_w_paths_and_filenames(self):
        """获取所有 projected_w.npz 的路径和对应文件名"""
        print('[info] 收集 projected_w 路径...')
        if not os.path.isdir(self.annotation_dir):
            raise FileNotFoundError(f'标注图片目录不存在: {self.annotation_dir}')
        w_paths = []
        filenames = []
        image_names = [name for name in os.listdir(self.annotation_dir)
                       if Path(name).suffix.lower() in {'.png', '.jpg', '.jpeg'}
                       and os.path.isfile(os.path.join(self.annotation_dir, name))]
        image_names.sort(key=lambda name: (natural_sort_key(name), name))
        if not image_names:
            raise ValueError(f'未找到 PNG/JPG/JPEG 图片: {self.annotation_dir}')
        seen = set()
        for fname in image_names:
            prefix = Path(fname).stem
            if prefix.casefold() in seen:
                raise ValueError(f'重复图片前缀，无法唯一配对 JSON 和投影文件: {prefix}')
            seen.add(prefix.casefold())
            w_path = os.path.join(self.projected_dir, f'seed{prefix}', 'projected_w.npz')
            json_path = os.path.join(self.annotation_dir, 'keypoints_annotation', f'{prefix}.json')
            for required_path in (w_path, json_path):
                if not os.path.isfile(required_path):
                    raise FileNotFoundError(f'样本 {prefix} 缺少文件: {required_path}')
            w_paths.append(w_path)
            filenames.append(prefix)
        print(f'[info] 找到 {len(w_paths)} 个 projected_w 文件')
        return w_paths, filenames

    def load_keypoints_list(self, filenames):
        """加载 keypoints json (数据量小，可以全部加载)"""
        print('[info] Loading keypoints from json...')
        keypoints_list = []
        for filename in filenames:
            json_path = f'{self.annotation_dir}/keypoints_annotation/{filename}.json'
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for dimension in ('imageWidth', 'imageHeight'):
                if data.get(dimension) is not None and data[dimension] != ORIGIN_PIC_SIZE:
                    raise ValueError(f'{json_path}: {dimension} 必须为 {int(ORIGIN_PIC_SIZE)}，'
                                     f'实际为 {data[dimension]}')
            if not isinstance(data.get('shapes'), list):
                raise ValueError(f'{json_path}: 缺少 LabelMe shapes 列表')
            point_dict = {}
            for shape in data['shapes']:
                label = shape.get('label')
                if label not in KEYPOINT_ORDER:
                    continue
                if shape.get('shape_type') != 'point':
                    raise ValueError(f'{json_path}: {label} 必须使用 point 标注')
                if label in point_dict:
                    raise ValueError(f'{json_path}: 重复关键点标签 {label}')
                points = np.asarray(shape.get('points'), dtype=np.float64)
                if points.shape != (1, 2) or not np.isfinite(points).all():
                    raise ValueError(f'{json_path}: {label} 必须包含一个有限的 (x, y) 坐标')
                if np.any(points < 0) or np.any(points >= ORIGIN_PIC_SIZE):
                    raise ValueError(f'{json_path}: {label} 坐标必须位于 [0, {int(ORIGIN_PIC_SIZE)})')
                point_dict[label] = points[0] / ORIGIN_PIC_SIZE
            pts = []
            for label in KEYPOINT_ORDER:
                if label not in point_dict:
                    raise ValueError(f"{filename} Missing keypoint: {label}")
                pts.append(point_dict[label])
            keypoints_array = np.array(pts, dtype=np.float32)
            keypoints_list.append(keypoints_array)
        print(f'[info] Loaded {len(keypoints_list)} keypoints sets')
        return keypoints_list

    def load_generator(self):
        """加载 StyleGAN G (只加载一次)"""
        print('[info] Loading StyleGAN generator...')
        if not os.path.isdir(self.stylegan_path):
            raise FileNotFoundError(f'StyleGAN2 源码目录不存在: {self.stylegan_path}')
        if not os.path.isfile(self.generator_path):
            raise FileNotFoundError(f'生成器权重不存在: {self.generator_path}')
        if self.stylegan_path not in sys.path:
            sys.path.insert(0, self.stylegan_path)
        with open(self.generator_path, 'rb') as f:
            self.G = pickle.load(f)['G_ema'].to(self.device).eval().requires_grad_(False)
        if getattr(self.G, 'img_resolution', int(ORIGIN_PIC_SIZE)) != int(ORIGIN_PIC_SIZE):
            raise ValueError(f'人脸流程需要 {int(ORIGIN_PIC_SIZE)}×{int(ORIGIN_PIC_SIZE)} 生成器')
        print('[info] G loaded')
        return self.G

    def unload_generator(self):
        """卸载 StyleGAN G 释放显存"""
        if self.G is not None:
            del self.G
            self.G = None
            torch.cuda.empty_cache()
            gc.collect()
            print('[info] StyleGAN generator unloaded')


class LazyKeypointDataset(Dataset):
    """懒加载关键点数据集 - 按需提取特征和生成热图"""

    def __init__(self, w_paths, keypoints_list, G, processor):
        """
        Args:
            w_paths: projected_w.npz 文件路径列表
            keypoints_list: 关键点列表 (已加载到内存)
            G: StyleGAN 生成器 (已加载到目标 device)
            processor: LazyDataProcessor 实例，用于生成热图
        """
        print('\n[Dataset] 初始化懒加载数据集...')
        self.w_paths = w_paths
        self.keypoints_list = keypoints_list
        self.G = G
        self.processor = processor
        self.num_samples = len(w_paths)
        self.device = device
        # 预加载所有 w 数据到内存 (w 数据量相对较小)
        # 但不进行上采样，上采样在 __getitem__ 中按需进行
        print('[Dataset] 预加载 w 数据 (不上采样)...')
        self.w_data_list = []
        if not w_paths or len(w_paths) != len(keypoints_list):
            raise ValueError('投影数据和关键点必须非空且样本数一致')
        expected_shape = (G.num_ws, G.w_dim)
        for w_path in tqdm(w_paths, desc="Loading w data"):
            with np.load(w_path, allow_pickle=False) as archive:
                if 'w' not in archive:
                    raise ValueError(f'{w_path}: NPZ 中缺少 w')
                w = np.asarray(archive['w'])
            if w.ndim == 3 and w.shape[0] == 1:
                w = w[0]
            if w.shape != expected_shape:
                raise ValueError(f'{w_path}: w 形状应为 {expected_shape} 或 (1, {expected_shape[0]}, '
                                 f'{expected_shape[1]})，实际为 {w.shape}')
            if not np.issubdtype(w.dtype, np.number) or not np.isfinite(w).all():
                raise ValueError(f'{w_path}: w 必须包含有限数值')
            self.w_data_list.append(w.astype(np.float32, copy=False))

        print(f'[Dataset] 数据集大小: {self.num_samples} 样本')
        print(f'[Dataset] w 数据内存占用约: {sum(w.nbytes for w in self.w_data_list) / 1024 / 1024:.2f} MB')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        按需提取特征和生成热图
        这样可以避免一次性上采样所有数据占用大量 RAM
        """
        # 1. 从预加载的 w 数据获取
        w = self.w_data_list[idx]

        # 2. 提取特征 (上采样到 256x256)
        feature_map = self._extract_features_from_w(w)

        # 3. 生成热图
        kpts = self.keypoints_list[idx]
        heatmap = self._generate_heatmap_from_kpts(kpts)

        # 4. 转换为 tensor
        feat_tensor = torch.from_numpy(feature_map).float()
        hm_tensor = torch.from_numpy(heatmap).float()

        return feat_tensor, hm_tensor

    def _extract_features_from_w(self, w):
        """从单个 w 提取特征图 (上采样到 256x256)"""
        ws = torch.from_numpy(w).float().unsqueeze(0).to(self.device)
        features = []
        hooks = []

        def make_hook():
            def _hook(module, inp, out):
                features.append(out.detach().float())
            return _hook

        try:
            hooks.append(self.G.synthesis.b4.conv1.register_forward_hook(make_hook()))
            for res in self.G.synthesis.block_resolutions[1:]:
                blk = getattr(self.G.synthesis, f'b{res}')
                hooks.append(blk.conv0.register_forward_hook(make_hook()))
                hooks.append(blk.conv1.register_forward_hook(make_hook()))
            with torch.no_grad():
                _ = self.G.synthesis(ws, noise_mode='const')
        finally:
            for h in hooks:
                h.remove()

        upsampled = []
        for feat in features:
            if feat.shape[2] != FEATURE_MAP_SIZE:
                up = torch.nn.functional.interpolate(
                    feat, size=(FEATURE_MAP_SIZE, FEATURE_MAP_SIZE),
                    mode='bilinear', align_corners=False)
                upsampled.append(up)
            else:
                upsampled.append(feat)

        combined = torch.cat(upsampled, dim=1).squeeze(0)
        feature_map = combined.cpu().numpy()

        del ws, features, hooks, upsampled, combined
        # 注意：这里不调用 torch.cuda.empty_cache()，因为会频繁调用
        # 让 PyTorch 自己管理显存缓存

        return feature_map

    def _generate_heatmap_from_kpts(self, kpts):
        """从 keypoints 生成热图"""
        N_K, _ = kpts.shape
        H = W = FEATURE_MAP_SIZE
        sigma = 10.0
        heatmaps = np.zeros((N_K, H, W), dtype=np.float32)
        for j in range(N_K):
            cx = kpts[j, 0] * W
            cy = kpts[j, 1] * H
            heatmaps[j] = self._make_heatmap(H, W, cx, cy, sigma)
        return heatmaps

    def _make_heatmap(self, height, width, cx, cy, sigma=10.0):
        """生成单关键点高斯热图"""
        y_grid, x_grid = np.ogrid[:height, :width]
        dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
        hmap = np.exp(-dist_sq / (2 * sigma**2))
        max_v = hmap.max()
        if max_v > 0:
            hmap /= max_v
        return hmap.astype(np.float32)


class HeatmapPredictor(nn.Module):
    """动态通道关键点热图预测模型"""

    def __init__(self, in_channels=5568, num_keypoints=18, verbose=True):
        super().__init__()
        if verbose:
            print(f'\nHeatmapPredictor(in_ch={in_channels}, kpts={num_keypoints})')

        self.step1_channel_reduce = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=1),
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
            nn.Conv2d(64, num_keypoints, kernel_size=1),
            nn.Sigmoid(),
        )

        if verbose:
            total_p = sum(p.numel() for p in self.parameters())
            trainable_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'  总参数量     : {total_p:,}')
            print(f'  可训练参数量 : {trainable_p:,}\n')

    def forward(self, x):
        x = self.step1_channel_reduce(x)
        x = self.step2_downsample1(x)
        x = self.step3_downsample2(x)
        x = self.step4_upsample(x)
        x = self.step5_output(x)
        return x


def extract_keypoints_from_heatmaps(heatmaps):
    """从热图提取关键点"""
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().cpu().numpy()
    B, K, H, W = heatmaps.shape
    keypoints = np.zeros((B, K, 2), dtype=np.float16)
    for b in range(B):
        for k in range(K):
            hmap = heatmaps[b, k]
            y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
            keypoints[b, k] = [x / W, y / H]
    return keypoints


def visualize_results(gt_keypoints, pred_keypoints, save_path, num_samples=4):
    """可视化结果"""
    n = min(num_samples, len(gt_keypoints))
    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for i in range(n):
        gt_px = gt_keypoints[i] * ORIGIN_PIC_SIZE
        pred_px = pred_keypoints[i] * ORIGIN_PIC_SIZE
        ax = axes[i, 0]
        ax.set_xlim(0, ORIGIN_PIC_SIZE)
        ax.set_ylim(ORIGIN_PIC_SIZE, 0)
        ax.scatter(gt_px[:, 0], gt_px[:, 1], c='blue', s=60, zorder=3)
        for j, (x, y) in enumerate(gt_px):
            ax.annotate(str(j), (x, y), fontsize=7, color='navy')
        ax.set_title(f'Sample {i}  —  Ground Truth', fontsize=11)
        ax.set_facecolor('#f0f0f0')

        ax = axes[i, 1]
        ax.set_xlim(0, ORIGIN_PIC_SIZE)
        ax.set_ylim(ORIGIN_PIC_SIZE, 0)
        ax.scatter(pred_px[:, 0], pred_px[:, 1], c='red', s=60, zorder=3)
        for j, (x, y) in enumerate(pred_px):
            ax.annotate(str(j), (x, y), fontsize=7, color='darkred')
        ax.set_title(f'Sample {i}  —  Predicted', fontsize=11)
        ax.set_facecolor('#f0f0f0')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  可视化结果已保存: {save_path}')


class Trainer:
    """训练器类 - 懒加载版本"""

    def __init__(self, args):
        self.args = args
        self.exp_dir = args['exp_dir']
        self.epochs = args['epochs']
        self.batch_size = args['batch_size']
        self.lr = args['lr']
        self.patience = args['patience']
        self.resume_path = args.get('resume', None)

        self.start_epoch = 0
        self.best_train_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.filenames = []
        self.best_checkpoint_path = None

        self.processor = None
        self.dataset = None
        self.model = None
        self.optimizer = None
        self.scaler = make_grad_scaler()

        self.scheduler = None
        self.criterion = nn.MSELoss()

        # 用于评估时保存 keypoints
        self.keypoints_list = None

    def prepare_data(self):
        """准备数据 - 懒加载版本"""
        print('\n[1/5] 准备数据集 (懒加载模式)...')
        self.processor = LazyDataProcessor(**{
            key: self.args[key]
            for key in ('annotation_dir', 'projected_dir', 'generator_path', 'stylegan_path')
            if key in self.args
        })

        # 只获取路径，不加载全部数据
        w_paths, filenames = self.processor.get_w_paths_and_filenames()
        self.filenames = filenames

        # 加载 keypoints (数据量小)
        self.keypoints_list = self.processor.load_keypoints_list(filenames)

        # 加载 StyleGAN G (只加载一次)
        G = self.processor.load_generator()

        # 创建懒加载数据集
        self.dataset = LazyKeypointDataset(
            w_paths=w_paths,
            keypoints_list=self.keypoints_list,
            G=G,
            processor=self.processor
        )

        # 只提取第一个样本以推断实际通道数，不缓存整套特征。
        sample_feat, _ = self.dataset[0]
        ch_in = sample_feat.shape[0]
        num_kpts = self.keypoints_list[0].shape[0]
        del sample_feat

        print(f'[info] 特征通道数: {ch_in}, 关键点数: {num_kpts}')
        if ch_in != 5568:
            print('[info] 此生成器的特征通道不是 5568；旧 inference.py 需要对应修改输入通道。')

        os.makedirs(self.exp_dir, exist_ok=True)
        with open(os.path.join(self.exp_dir, 'filenames.json'), 'w', encoding='utf-8') as f:
            json.dump(self.filenames, f, ensure_ascii=False, indent=2)
        config = dict(self.args)
        config.update({key: getattr(self.processor, key) for key in
                       ('annotation_dir', 'projected_dir', 'generator_path', 'stylegan_path')})
        config.update(device=device, keypoint_order=KEYPOINT_ORDER,
                      origin_pic_size=ORIGIN_PIC_SIZE, feature_map_size=FEATURE_MAP_SIZE,
                      in_channels=ch_in, num_samples=len(self.dataset))
        with open(os.path.join(self.exp_dir, 'training_config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2, default=os.fspath)

        return ch_in, num_kpts

    def build_model(self, in_channels, num_keypoints):
        """构建模型"""
        print('\n[2/5] 构建模型...')
        self.model = HeatmapPredictor(in_channels=in_channels,
                                       num_keypoints=num_keypoints,
                                       verbose=True).to(device)

    def setup_optimizer(self):
        """设置优化器"""
        print('\n[3/5] 配置损失函数和优化器...')
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=20, factor=0.5)

    def load_checkpoint(self):
        """加载断点"""
        if self.resume_path:
            if not os.path.isfile(self.resume_path):
                raise FileNotFoundError(f'续训检查点不存在: {self.resume_path}')
            print(f'\n[Resume] 加载断点: {self.resume_path}')
            ckpt = torch.load(self.resume_path, map_location=device)
            self.start_epoch = ckpt.get('epoch', 0) + 1
            self.best_train_loss = ckpt.get('best_train_loss', ckpt.get('train_loss', float('inf')))
            self.patience_counter = ckpt.get('patience_counter', 0)
            self.train_losses = ckpt.get('train_losses', [])
            self.model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print('  优化器已恢复')
            if 'scheduler_state_dict' in ckpt:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                print('  调度器已恢复')
            if 'scaler_state_dict' in ckpt:
                self.scaler.load_state_dict(ckpt['scaler_state_dict'])
                print('  AMP scaler 已恢复')
            missing = [name for name in ('optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict')
                       if name not in ckpt]
            if missing:
                print(f'  旧检查点未包含 {", ".join(missing)}；对应状态保持初始化。')
            # 即使目标轮数已达到或后续没有改善，也保留刚恢复的最佳模型供评估。
            os.makedirs(self.exp_dir, exist_ok=True)
            destination = os.path.join(self.exp_dir, 'best_model.pth')
            if Path(self.resume_path).resolve() != Path(destination).resolve():
                shutil.copy2(self.resume_path, destination)
            self.best_checkpoint_path = destination
            print(f'  从 epoch {self.start_epoch} 续训, best_loss={self.best_train_loss:.6f}')
        else:
            print('\n[Resume] 新训练')

    def train_epoch(self, train_loader, epoch):
        """训练一个 epoch"""
        self.model.train()
        train_loss = 0.0
        train_loader_iter = tqdm(train_loader,
                                 desc=f"Epoch {epoch+1}/{self.epochs} [Train]",
                                 leave=False)
        for feats, hm in train_loader_iter:
            feats = feats.to(device, non_blocking=True)
            hm = hm.to(device, non_blocking=True)
            self.optimizer.zero_grad()
            with autocast_context():
                pred = self.model(feats)
                loss = self.criterion(pred, hm)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            train_loss += loss.item()
            train_loader_iter.set_postfix({"batch_loss": f"{loss.item():.6f}"})
        train_loss /= len(train_loader)
        return train_loss

    def save_checkpoint(self, epoch, train_loss):
        """保存断点"""
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'train_loss': train_loss,
            'best_train_loss': self.best_train_loss,
            'patience_counter': self.patience_counter,
            'train_losses': self.train_losses,
        }
        self.best_checkpoint_path = os.path.join(self.exp_dir, 'best_model.pth')
        torch.save(save_dict, self.best_checkpoint_path)
        print(f'  新最佳模型保存 (loss={train_loss:.6f})')

    def train(self):
        """训练主循环"""
        print('\n[4/5] 开始训练...')
        os.makedirs(self.exp_dir, exist_ok=True)

        # 使用较少的 workers，因为每个样本需要 GPU 计算
        train_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)
        print(f'  数据集总样本数: {len(self.dataset)} 样本')

        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            self.scheduler.step(train_loss)

            print(f'Epoch {epoch+1:4d}/{self.epochs} | Train Loss: {train_loss:.6f} | Best: {self.best_train_loss:.6f}')

            # 保存损失曲线
            if (epoch + 1) % 10 == 0:
                plt.figure(figsize=(10, 4))
                plt.plot(self.train_losses, label='Train Loss', color='steelblue')
                plt.xlabel('Epoch')
                plt.ylabel('MSE Loss')
                plt.title(f'Training Curve  (epoch {epoch+1})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.exp_dir, 'loss_curve.png'), dpi=150)
                plt.close()

            # 早停 & 保存
            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, train_loss)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f'\n  早停: epoch {epoch+1}, best_train_loss={self.best_train_loss:.6f}')
                    break

            gc.collect()
            torch.cuda.empty_cache()

        # 保存最终损失曲线
        plt.figure(figsize=(10, 4))
        plt.plot(self.train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training Curve')
        plt.legend()
        plt.savefig(os.path.join(self.exp_dir, 'loss_curve.png'), dpi=150)
        plt.close()
        print(f'  损失曲线已保存: {self.exp_dir}/loss_curve.png')

    def evaluate(self):
        """评估模型"""
        print('\n[5/5] 评估模型...')
        if self.best_checkpoint_path is None:
            raise RuntimeError('本次运行没有可评估的最佳检查点；请训练至少一轮或提供 --resume。')
        ckpt = torch.load(self.best_checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        gt_kpts_all = np.stack(self.keypoints_list)
        all_pred_kpts = []

        full_loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0)
        with torch.no_grad():
            for feats, _ in tqdm(full_loader, desc="Evaluating"):
                feats = feats.to(device)
                pred_hm = self.model(feats)
                pred_kpts = extract_keypoints_from_heatmaps(pred_hm)
                all_pred_kpts.append(pred_kpts)

        all_pred_kpts = np.concatenate(all_pred_kpts, axis=0)
        errors = np.linalg.norm(all_pred_kpts - gt_kpts_all, axis=-1)
        mean_error = np.mean(errors)
        print(f'  平均欧氏距离误差 (归一化) : {mean_error:.4f}')
        print(f'  平均欧氏距离误差 (像素)   : {mean_error * ORIGIN_PIC_SIZE:.2f} px')

        pred_save = os.path.join(self.exp_dir, 'predicted_keypoints.npy')
        np.save(pred_save, all_pred_kpts)
        print(f'  预测关键点已保存: {pred_save}')

        visualize_results(
            gt_keypoints=gt_kpts_all,
            pred_keypoints=all_pred_kpts,
            save_path=os.path.join(self.exp_dir, 'evaluation_results.png'),
            num_samples=4,
        )

        # 评估完成后卸载 StyleGAN G
        self.dataset.G = None
        self.processor.unload_generator()

        print(f'\n训练完成！所有结果保存在: {self.exp_dir}')

    def run(self):
        """运行完整训练流程"""
        print('\n' + '=' * 60)
        print('18 点人脸关键点热图预测训练 (内存优化版本)')
        print('=' * 60)
        for k, v in self.args.items():
            print(f'  {k}: {v}')
        print(f'  device: {device}')

        if self.epochs < 1 or self.batch_size < 1 or self.lr <= 0 or self.patience < 1:
            raise ValueError('epochs、batch_size、patience 必须为正整数，lr 必须大于 0')
        if self.resume_path and not os.path.isfile(self.resume_path):
            raise FileNotFoundError(f'续训检查点不存在: {self.resume_path}')
        try:
            ch_in, num_kpts = self.prepare_data()
            self.build_model(ch_in, num_kpts)
            self.setup_optimizer()
            self.load_checkpoint()
            self.train()
            self.evaluate()
        finally:
            if self.dataset is not None:
                self.dataset.G = None
            if self.processor is not None:
                self.processor.unload_generator()


def main():
    parser = argparse.ArgumentParser(description='18 点人脸关键点热图预测训练（懒加载）')
    parser.add_argument('--exp_dir', type=str, default='./exp_lazy_load')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--annotation_dir', type=str, default=str(REPO_ROOT / 'face_keypoint_annotation'),
                        help='图片目录，内部包含 keypoints_annotation/*.json')
    parser.add_argument('--projected_dir', type=str, default=str(SCRIPT_DIR / 'projected'),
                        help='包含 seed<图片前缀>/projected_w.npz 的目录')
    parser.add_argument('--generator_path', type=str, default=str(REPO_ROOT / 'ffhq.pkl'),
                        help='与投影和推理一致的 FFHQ 生成器权重')
    parser.add_argument('--stylegan_path', type=str, default=str(REPO_ROOT / 'stylegan2-ada-pytorch'),
                        help='StyleGAN2-ADA 源码目录（不是 PKL 文件）')
    args = parser.parse_args()

    config = vars(args)
    trainer = Trainer(config)
    trainer.run()


if __name__ == '__main__':
    main()
