"""
得到标注数据集的 W+ --> featuremap 的特征
"""
import os
import sys
import json
import numpy as np
import torch
import gc

# Add stylegan2-ada-pytorch to path
_sg2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../stylegan2-ada-pytorch')
if _sg2_path not in sys.path:
    sys.path.insert(0, _sg2_path)

import pickle

# 受限于显存的大小，我们将上采样修改成64，论文为256
FEATURE_MAP_SIZE = 256
ORINGIN_PIC_SIZE=1024.0


KEYPOINT_ORDER = [
    "chin",
    "left_eye_inner",
    "left_eye_outer",
    "left_eye_pupil",
    "left_eyebrow_inner",
    "left_eyebrow_outer",
    "lower_lip",
    "mouth_left",
    "mouth_right",
    "nose_left",
    "nose_right",
    "nose_tip",
    "right_eye_inner",
    "right_eye_outer",
    "right_eyebrow_outer",
    "right_eyebrow_inner",
    "upper_lip",
    "right_eye_pupil"
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def prepare_keypoint_data(featuresmap=True,keypoints=True):
    # Load FFHQ model
    if featuresmap ==True:
        with open('../ffhq.pkl', 'rb') as f:
            G = pickle.load(f)['G_ema'].to(device)
        
        # Load latents (W-space codes)
        latents = np.load('latent_ffhq.npy')  # (16, 18, 512)
        print(f'Loaded latents: {latents.shape}')
        
        # Generate GLOBAL features
        features_list = []
        
        for i, w in enumerate(latents):
            # ws = torch.from_numpy(w).float().to(device).unsqueeze(0)  # (1, 18, 512)
            ws = torch.from_numpy(w).float().to(device)
            # 自动修正 shape
            if ws.ndim == 4 and ws.shape[1] == 1:
                ws = ws.squeeze(1)
            
            print(ws.shape)
            # Hook to collect feature maps
            features = []
            hooks = []
            
            def make_hook():
                def _hook(module, inp, out):
                    features.append(out.detach().float())
                return _hook
            
            # Hook all convolutional layers
            hooks.append(G.synthesis.b4.conv1.register_forward_hook(make_hook()))
            for res in G.synthesis.block_resolutions[1:]:
                blk = getattr(G.synthesis, f'b{res}')
                hooks.append(blk.conv0.register_forward_hook(make_hook()))
                hooks.append(blk.conv1.register_forward_hook(make_hook()))
            
            # Forward pass
            with torch.no_grad():
                _ = G.synthesis(ws, noise_mode='const')
            
            # Remove hooks
            for h in hooks:
                h.remove()
            
            
            # Upsample all features to 64x64
            target_size = FEATURE_MAP_SIZE
            upsampled = []
            for feat in features:
                if feat.shape[2] != target_size:
                    up = torch.nn.functional.interpolate(
                        feat, 
                        size=(target_size, target_size),
                        mode='bilinear',
                        align_corners=False
                    )
                    upsampled.append(up)
                else:
                    upsampled.append(feat)
            
            # Concatenate all features
            combined = torch.cat(upsampled, dim=1)  # (1, 5568, 64, 64)
            
            # Global average pooling
            feature_map = combined.squeeze(0)  # (5568,64,64)
            #稳定容易出错
            features_list.append(feature_map.squeeze(0).cpu().numpy().astype(np.float16))
            # features_list.append(feature_map.cpu().numpy().astype(np.float16))


            # 🔥 重要：立即释放内存
            del ws, features, hooks, upsampled, combined, feature_map
            gc.collect()
            
            torch.cuda.empty_cache()
            
            if (i + 1) % 5 == 0:
                print(f'Processed {i+1}/16: memory usage {torch.cuda.memory_allocated()/1024**3:.2f} GB')
        
        # Save features (16, 5568)
        features_array = np.stack(features_list)  # (16, 5568)
        print(f'Saving  features: {features_array.shape}')
        np.save('features.npy', features_array)
        print(f'features already Saved : {features_array.shape}！')
        
    if keypoints==True:
    # Load keypoints
        keypoints_list = []
        filenames = ['0','2', '3', '4', '7', '12', '68', '70', '76', '135', '160', '250', '251', '344', '439', '543']
        
        for filename in filenames:
            json_path = f'../face_keypoint_annotation/keypoints_annotation/{filename}.json'
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # 错误的标签顺序
            # pts = []
            # for shape in data['shapes']:
            #     if shape['shape_type'] == 'point':
            #         x, y = shape['points'][0]
            #         pts.append([x /ORINGIN_PIC_SIZE , y / ORINGIN_PIC_SIZE])  # Normalize to [0, 1]

            # 构建 label->point 字典
            point_dict = {}

            for shape in data['shapes']:
                if shape['shape_type'] == 'point':
                    label = shape['label']
                    x, y = shape['points'][0]
                    point_dict[label] = [x / ORINGIN_PIC_SIZE, y / ORINGIN_PIC_SIZE]

            # 按固定顺序生成 keypoints
            pts = []

            for label in KEYPOINT_ORDER:
                if label not in point_dict:
                    raise ValueError(f'{filename} Missing keypoint: {label},')
                pts.append(point_dict[label])

            keypoints_array = np.array(pts)  # (18, 2) - 用户确认所有样本有18个关键点
            keypoints_list.append(keypoints_array)
        
        # Save keypoints (16, 18, 2)
        keypoints_array = np.stack(keypoints_list)  # (16, 18, 2)
        np.save('keypoints.npy', keypoints_array)
        print(f'Saved keypoints: {keypoints_array.shape}')

if __name__ == '__main__':
    prepare_keypoint_data(featuresmap=False,keypoints=True)
