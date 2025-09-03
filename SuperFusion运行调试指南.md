# SuperFusion 运行调试指南

## 🚀 快速开始

### 1. 环境配置

#### 系统要求
- Python 3.7+
- CUDA 11.5+
- GPU内存 ≥ 8GB (推荐 ≥ 16GB)
- 系统内存 ≥ 32GB

#### 安装依赖
```bash
# 1. 创建虚拟环境 (推荐)
conda create -n superfusion python=3.8
conda activate superfusion

# 2. 安装PyTorch (CUDA 11.5版本)
pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115

# 3. 安装其他依赖
pip install -r requirement.txt

# 4. 验证安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available__()}'); print(f'GPU数量: {torch.cuda.device_count()}')"
```

### 2. 数据准备

#### 下载nuScenes数据集
```bash
# 1. 访问 https://www.nuscenes.org/ 注册并下载
# 2. 下载完整数据集和地图扩展包
# - Full dataset (v1.0)
# - Map expansion

# 3. 解压到指定目录
# 目录结构应该如下:
# /path/to/nuScenes/
# ├── maps/
# ├── samples/
# ├── sweeps/
# ├── v1.0-trainval/
# └── v1.0-test/
```

#### 下载预训练模型
```bash
# 1. 创建checkpoints目录
mkdir -p checkpoints

# 2. 下载DeepLabV3预训练模型
wget https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth -O checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth

# 3. 创建runs目录
mkdir -p runs

# 4. 下载SuperFusion预训练模型 (可选)
# 从Google Drive下载: https://drive.google.com/file/d/1UTgughJ71Rn0zPUDTXFo__HJS-57lwNG/view?usp=sharing
# 放置到 runs/ 目录下
```

## 🏃‍♂️ 运行代码

### 1. 训练模型

#### 基础训练命令
```bash
python train.py \
    --instance_seg \
    --direction_pred \
    --depth_sup \
    --dataroot /path/to/nuScenes/ \
    --pretrained \
    --add_depth_channel
```

#### 完整训练参数示例
```bash
python train.py \
    --dataroot /path/to/nuScenes/ \
    --version v1.0-trainval \
    --model SuperFusion \
    --logdir ./runs/experiment_1 \
    --nepochs 30 \
    --bsz 4 \
    --lr 0.1 \
    --weight_decay 1e-7 \
    --max_grad_norm 5.0 \
    --pos_weight 2.13 \
    --instance_seg \
    --direction_pred \
    --depth_sup \
    --pretrained \
    --add_depth_channel \
    --data_aug
```

#### 关键训练参数说明
- `--dataroot`: nuScenes数据集路径
- `--version`: 数据集版本 (v1.0-trainval 或 v1.0-mini)
- `--logdir`: 训练日志和模型保存目录
- `--nepochs`: 训练轮数 (默认30)
- `--bsz`: 批次大小 (默认4，根据GPU内存调整)
- `--lr`: 学习率 (默认0.1)
- `--instance_seg`: 启用实例分割任务
- `--direction_pred`: 启用方向预测任务
- `--depth_sup`: 启用深度监督
- `--pretrained`: 使用预训练的DeepLabV3模型
- `--add_depth_channel`: 添加深度通道
- `--data_aug`: 启用数据增强

### 2. 模型评估

#### IoU评估
```bash
python evaluate_iou_split.py \
    --dataroot /path/to/nuScenes/ \
    --modelf runs/model.pt \
    --instance_seg \
    --direction_pred \
    --depth_sup \
    --add_depth_channel \
    --pretrained
```

#### CD和AP评估
```bash
# 1. 导出预测结果为JSON格式
python export_pred_to_json.py \
    --dataroot /path/to/nuScenes/ \
    --modelf runs/model.pt \
    --depth_sup \
    --add_depth_channel \
    --pretrained

# 2. 计算CD和AP指标
python evaluate_json_split.py \
    --result_path output.json \
    --dataroot /path/to/nuScenes/
```

### 3. 结果可视化

#### 可视化真值
```bash
python vis_prediction_gt.py \
    --instance_seg \
    --direction_pred \
    --dataroot /path/to/nuScenes/
```

#### 可视化预测结果
```bash
python vis_prediction.py \
    --modelf runs/model.pt \
    --instance_seg \
    --direction_pred \
    --depth_sup \
    --pretrained \
    --add_depth_channel \
    --version v1.0-trainval \
    --dataroot /path/to/nuScenes/
```

## 🐛 调试指南

### 1. 常见问题及解决方案

#### 内存不足问题
```bash
# 问题: CUDA out of memory
# 解决方案:
# 1. 减小批次大小
python train.py --bsz 2  # 从默认4减少到2

# 2. 减小图像尺寸
python train.py --image_size 128 352  # 从默认256 704减少

# 3. 使用梯度累积
# 在train.py中修改accumulation_steps参数
```

#### 数据加载问题
```bash
# 问题: FileNotFoundError或数据路径错误
# 解决方案:
# 1. 检查数据路径
ls /path/to/nuScenes/

# 2. 检查数据集版本
python -c "from nuscenes.nuscenes import NuScenes; nusc = NuScenes(version='v1.0-trainval', dataroot='/path/to/nuScenes/', verbose=True)"

# 3. 验证数据完整性
python -c "from data.dataset_front import semantic_dataset; dataset = semantic_dataset('/path/to/nuScenes/', 'v1.0-trainval'); print(f'数据集大小: {len(dataset)}')"
```

#### 模型加载问题
```bash
# 问题: 预训练模型加载失败
# 解决方案:
# 1. 检查模型文件是否存在
ls -la checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth

# 2. 验证模型文件完整性
python -c "import torch; model = torch.load('checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth'); print('模型加载成功')"
```

### 2. 调试技巧

#### 启用详细日志
```bash
# 1. 设置日志级别
export PYTHONPATH=$PYTHONPATH:.
export CUDA_LAUNCH_BLOCKING=1  # 同步CUDA操作，便于调试

# 2. 使用调试模式运行
python -u train.py --bsz 1 --nepochs 1  # 单样本单轮次快速测试
```

#### 使用TensorBoard监控训练
```bash
# 1. 启动TensorBoard
tensorboard --logdir=./runs --port=6006

# 2. 在浏览器中访问
# http://localhost:6006

# 3. 监控指标:
# - 损失函数变化
# - IoU指标
# - 学习率变化
# - 梯度范数
```

#### 代码断点调试
```python
# 在关键位置添加断点
import pdb; pdb.set_trace()

# 或使用IPython调试器
import IPython; IPython.embed()

# 检查张量形状和数值
print(f"张量形状: {tensor.shape}")
print(f"张量范围: [{tensor.min():.4f}, {tensor.max():.4f}]")
print(f"是否包含NaN: {torch.isnan(tensor).any()}")
print(f"是否包含Inf: {torch.isinf(tensor).any()}")
```

### 3. 性能优化

#### 数据加载优化
```bash
# 1. 增加数据加载进程数
python train.py --nworkers 8  # 根据CPU核心数调整

# 2. 启用数据预取
# 在dataset.py中设置pin_memory=True
```

#### 训练加速
```bash
# 1. 使用混合精度训练
# 在train.py中添加:
# from torch.cuda.amp import autocast, GradScaler
# scaler = GradScaler()

# 2. 启用cudnn benchmark
# torch.backends.cudnn.benchmark = True
```

### 4. 模型验证

#### 快速验证流程
```bash
# 1. 小数据集测试
python train.py --version v1.0-mini --nepochs 1 --bsz 1

# 2. 过拟合单个样本
python train.py --bsz 1 --nepochs 100 # 验证模型容量

# 3. 检查梯度流
# 在train.py中添加梯度检查代码
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

#### 输出检查
```python
# 检查模型输出合理性
def check_model_output(output):
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print(f"输出均值: {output.mean():.4f}")
    print(f"输出标准差: {output.std():.4f}")
    
    # 检查概率分布
    if output.dim() == 4:  # [B, C, H, W]
        probs = torch.softmax(output, dim=1)
        print(f"概率和: {probs.sum(dim=1).mean():.4f} (应该接近1.0)")
```

## 📊 监控和分析

### 1. 训练监控指标
- **损失函数**: 总损失、分割损失、嵌入损失、方向损失
- **评估指标**: IoU、精确率、召回率
- **系统指标**: GPU利用率、内存使用、训练速度

### 2. 模型分析工具
```bash
# 1. 模型参数统计
python -c "from model_front import get_model; model = get_model('SuperFusion', {}); print(f'参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')"

# 2. 计算复杂度分析
# 使用thop库计算FLOPs
pip install thop
python -c "from thop import profile; # 添加FLOPs计算代码"

# 3. 推理速度测试
python -c "import time; # 添加推理时间测试代码"
```

### 3. 结果分析
```bash
# 1. 错误案例分析
python vis_prediction.py --save_errors  # 保存预测错误的案例

# 2. 不同距离范围的性能分析
python evaluate_iou_split.py --distance_ranges "[0,30]" "[30,60]" "[60,90]"

# 3. 不同场景的性能分析
python evaluate_iou_split.py --scene_types "urban" "highway" "parking"
```

## 🔧 高级配置

### 1. 自定义数据集
```python
# 在data/dataset_front.py中添加自定义数据集类
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, version):
        # 实现自定义数据加载逻辑
        pass
    
    def __getitem__(self, idx):
        # 返回样本数据
        pass
```

### 2. 模型架构修改
```python
# 在model_front/super_fusion.py中修改网络结构
# 例如：添加新的融合模块、修改特征维度等
```

### 3. 损失函数定制
```python
# 在loss.py中添加自定义损失函数
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # 实现自定义损失计算
        pass
```

这个指南涵盖了SuperFusion项目的完整运行和调试流程，从环境配置到高级定制，帮助您快速上手并解决可能遇到的问题。