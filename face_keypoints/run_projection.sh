#!/bin/bash

# ==========================================
# 1. 环境初始化 (关键步骤)
# ==========================================
# 如果在脚本中遇到 "conda: command not found" 或 "activate: No such file or directory"
# 需要指定 conda 的安装路径 (通常是 anaconda3 或 miniconda3)
# 请根据你的实际安装路径修改下面这一行

#切换到stylegan2-ada-pytorch 路径
cd Dataset_generation/stylegan2-ada-pytorch

source ~/miniconda3/etc/profile.d/conda.sh

# 激活环境
echo "Activating environment: stylegan2-ada-new"
conda  your stylegan2 environment 

# ==========================================
# 2. 参数配置
# ==========================================
# 使用变量管理路径，方便修改
INPUT_DIR="../face_keypoint_annotation"
NETWORK_PKL="ffhq.pkl"
OUT_BASE_DIR="../face_keypoints/projected"

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录 $INPUT_DIR 不存在"
    exit 1
fi

# ==========================================
# 3. 循环处理
# ==========================================
# 使用 find 命令更稳健，或者直接用 ls
count=0

for img in "$INPUT_DIR"/*.png; do
    # 检查是否真的有文件 (防止目录为空时执行一次空命令)
    if [ ! -f "$img" ]; then
        echo "没有找到 PNG 文件。"
        break
    fi

    # 提取文件名 (去除路径和后缀)
    filename=$(basename "$img" .png)
    
    # 打印进度
    echo "------------------------------------------------"
    echo "正在处理 [$((count+1))]: $filename.png"
    
    # 执行 Python 投影
    # 使用 --save-video=False (注意: 根据 projector.py 的具体实现，
    # 有的版本是 --no-video 或 --save-video false)
    python projector.py \
      --outdir="${OUT_BASE_DIR}/seed${filename}" \
      --target="$img" \
      --network="$NETWORK_PKL" \
      --num-steps=500 \
      --save-video=False
      
    count=$((count+1))
done

echo "------------------------------------------------"
echo "全部完成! 共处理了 $count 张图片。"