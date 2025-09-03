#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperFusion 快速启动脚本
用于环境验证、数据检查和基础测试

使用方法:
python quick_start.py --check-env          # 检查环境
python quick_start.py --check-data         # 检查数据
python quick_start.py --test-model         # 测试模型
python quick_start.py --mini-train         # 迷你训练测试
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # Python版本
    python_version = sys.version_info
    print(f"✓ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # PyTorch版本
    print(f"✓ PyTorch版本: {torch.__version__}")
    
    # CUDA可用性
    if torch.cuda.is_available():
        print(f"✓ CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU数量: {torch.cuda.device_count()}")
        print(f"✓ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ CUDA不可用")
        return False
    
    # 检查关键依赖包
    required_packages = [
        'torchvision', 'kornia', 'numpy', 'tensorboardX', 
        'PIL', 'pyquaternion', 'nuscenes', 'inplace_abn', 
        'torch_scatter', 'shapely'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'nuscenes':
                from nuscenes.nuscenes import NuScenes
            else:
                __import__(package)
            print(f"✓ {package}: 已安装")
        except ImportError:
            print(f"❌ {package}: 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirement.txt")
        return False
    
    print("\n✅ 环境检查通过!")
    return True

def check_data(dataroot):
    """检查数据集"""
    print(f"\n🔍 检查数据集: {dataroot}")
    
    if not os.path.exists(dataroot):
        print(f"❌ 数据路径不存在: {dataroot}")
        return False
    
    # 检查必要的目录结构
    required_dirs = ['maps', 'samples', 'sweeps', 'v1.0-trainval']
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = os.path.join(dataroot, dir_name)
        if os.path.exists(dir_path):
            print(f"✓ {dir_name}: 存在")
        else:
            print(f"❌ {dir_name}: 不存在")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\n❌ 缺少目录: {', '.join(missing_dirs)}")
        return False
    
    # 尝试加载nuScenes数据集
    try:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
        print(f"✓ nuScenes数据集加载成功")
        print(f"✓ 场景数量: {len(nusc.scene)}")
        print(f"✓ 样本数量: {len(nusc.sample)}")
    except Exception as e:
        print(f"❌ nuScenes数据集加载失败: {e}")
        return False
    
    print("\n✅ 数据检查通过!")
    return True

def check_pretrained_models():
    """检查预训练模型"""
    print("\n🔍 检查预训练模型...")
    
    # 检查checkpoints目录
    checkpoints_dir = Path('checkpoints')
    if not checkpoints_dir.exists():
        print("❌ checkpoints目录不存在")
        print("请创建目录: mkdir checkpoints")
        return False
    
    # 检查DeepLabV3预训练模型
    deeplabv3_path = checkpoints_dir / 'deeplabv3_resnet101_coco-586e9e4e.pth'
    if deeplabv3_path.exists():
        print("✓ DeepLabV3预训练模型存在")
        try:
            torch.load(deeplabv3_path, map_location='cpu')
            print("✓ DeepLabV3模型加载成功")
        except Exception as e:
            print(f"❌ DeepLabV3模型加载失败: {e}")
            return False
    else:
        print("❌ DeepLabV3预训练模型不存在")
        print("请下载: wget https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth -O checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth")
        return False
    
    print("\n✅ 预训练模型检查通过!")
    return True

def test_model_loading():
    """测试模型加载"""
    print("\n🔍 测试模型加载...")
    
    try:
        # 导入模型
        sys.path.append('.')
        from model_front import get_model
        
        # 创建模型配置
        args = type('Args', (), {
            'instance_seg': True,
            'direction_pred': True,
            'depth_sup': True,
            'add_depth_channel': True,
            'pretrained': True,
            'camC': 64,
            'lidarC': 64,
            'downsample': 16,
            'use_lidar': True,
            'use_cam': True,
            'use_fusion': True,
            'add_fuser': True
        })()
        
        # 创建模型
        model = get_model('SuperFusion', args)
        print("✓ 模型创建成功")
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ 总参数量: {total_params/1e6:.2f}M")
        print(f"✓ 可训练参数量: {trainable_params/1e6:.2f}M")
        
        # 测试前向传播
        model.eval()
        with torch.no_grad():
            # 创建虚拟输入
            batch_size = 1
            imgs = torch.randn(batch_size, 6, 3, 256, 704)  # 6个相机
            lidar_data = torch.randn(batch_size, 5, 256, 704)  # 激光雷达BEV
            
            if torch.cuda.is_available():
                model = model.cuda()
                imgs = imgs.cuda()
                lidar_data = lidar_data.cuda()
            
            # 前向传播
            output = model(imgs, lidar_data)
            print(f"✓ 前向传播成功, 输出形状: {output.shape}")
        
        print("\n✅ 模型测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def mini_train_test(dataroot):
    """迷你训练测试"""
    print("\n🔍 迷你训练测试...")
    
    try:
        # 导入训练相关模块
        sys.path.append('.')
        from data.dataset_front import semantic_dataset
        from model_front import get_model
        from loss import SimpleLoss
        
        # 创建数据集
        dataset = semantic_dataset(dataroot, 'v1.0-trainval')
        print(f"✓ 数据集创建成功, 样本数量: {len(dataset)}")
        
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )
        
        # 获取一个样本
        sample = next(iter(dataloader))
        print(f"✓ 数据加载成功")
        print(f"  - 图像形状: {sample[0].shape}")
        print(f"  - 激光雷达形状: {sample[1].shape}")
        print(f"  - 标签形状: {sample[2].shape}")
        
        # 创建模型
        args = type('Args', (), {
            'instance_seg': False,  # 简化测试
            'direction_pred': False,
            'depth_sup': False,
            'add_depth_channel': False,
            'pretrained': True,
            'camC': 64,
            'lidarC': 64,
            'downsample': 16,
            'use_lidar': True,
            'use_cam': True,
            'use_fusion': True,
            'add_fuser': True
        })()
        
        model = get_model('SuperFusion', args)
        print("✓ 模型创建成功")
        
        # 创建损失函数
        loss_fn = SimpleLoss(pos_weight=2.13)
        
        # 创建优化器
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        if torch.cuda.is_available():
            model = model.cuda()
            loss_fn = loss_fn.cuda()
            sample = [s.cuda() if torch.is_tensor(s) else s for s in sample]
        
        # 训练一步
        model.train()
        optimizer.zero_grad()
        
        imgs, lidar_data, binimgs = sample[0], sample[1], sample[2]
        output = model(imgs, lidar_data)
        loss = loss_fn(output, binimgs)
        
        loss.backward()
        optimizer.step()
        
        print(f"✓ 训练一步成功, 损失值: {loss.item():.4f}")
        print("\n✅ 迷你训练测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 迷你训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='SuperFusion 快速启动脚本')
    parser.add_argument('--check-env', action='store_true', help='检查环境')
    parser.add_argument('--check-data', action='store_true', help='检查数据')
    parser.add_argument('--test-model', action='store_true', help='测试模型')
    parser.add_argument('--mini-train', action='store_true', help='迷你训练测试')
    parser.add_argument('--dataroot', type=str, default='/path/to/nuScenes/', 
                       help='nuScenes数据集路径')
    parser.add_argument('--all', action='store_true', help='运行所有检查')
    
    args = parser.parse_args()
    
    if not any([args.check_env, args.check_data, args.test_model, args.mini_train, args.all]):
        parser.print_help()
        return
    
    print("🚀 SuperFusion 快速启动检查")
    print("=" * 50)
    
    success = True
    
    # 环境检查
    if args.check_env or args.all:
        if not check_environment():
            success = False
        if not check_pretrained_models():
            success = False
    
    # 数据检查
    if args.check_data or args.all:
        if not check_data(args.dataroot):
            success = False
    
    # 模型测试
    if args.test_model or args.all:
        if not test_model_loading():
            success = False
    
    # 迷你训练测试
    if args.mini_train or args.all:
        if not mini_train_test(args.dataroot):
            success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 所有检查通过! 可以开始使用SuperFusion了")
        print("\n📚 下一步:")
        print("1. 训练模型: python train.py --dataroot /path/to/nuScenes/ --instance_seg --direction_pred --depth_sup --pretrained --add_depth_channel")
        print("2. 评估模型: python evaluate_iou_split.py --dataroot /path/to/nuScenes/ --modelf runs/model.pt")
        print("3. 可视化结果: python vis_prediction.py --modelf runs/model.pt --dataroot /path/to/nuScenes/")
    else:
        print("❌ 部分检查失败，请根据上述提示解决问题")
        sys.exit(1)

if __name__ == '__main__':
    main()