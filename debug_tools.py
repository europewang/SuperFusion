#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperFusion 调试工具集
提供模型分析、数据检查、错误诊断等功能

使用方法:
python debug_tools.py --analyze-model          # 分析模型结构
python debug_tools.py --check-gradients        # 检查梯度流
python debug_tools.py --profile-memory         # 内存分析
python debug_tools.py --test-dataloader        # 测试数据加载器
python debug_tools.py --visualize-features     # 可视化特征
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import psutil
import gc

def analyze_model(model_name='SuperFusion'):
    """分析模型结构和参数"""
    print(f"\n🔍 分析模型: {model_name}")
    
    try:
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
        
        model = get_model(model_name, args)
        
        print("\n📊 模型统计信息:")
        
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"冻结参数: {frozen_params:,} ({frozen_params/1e6:.2f}M)")
        
        # 模型大小估算
        param_size = total_params * 4 / (1024 ** 2)  # 假设float32
        print(f"模型大小 (参数): {param_size:.2f} MB")
        
        # 层级分析
        print("\n🏗️ 模型结构分析:")
        layer_count = {}
        for name, module in model.named_modules():
            module_type = type(module).__name__
            layer_count[module_type] = layer_count.get(module_type, 0) + 1
        
        for layer_type, count in sorted(layer_count.items()):
            if count > 1:
                print(f"{layer_type}: {count}")
        
        # 各模块参数分布
        print("\n📈 各模块参数分布:")
        module_params = {}
        for name, param in model.named_parameters():
            module_name = name.split('.')[0]
            if module_name not in module_params:
                module_params[module_name] = 0
            module_params[module_name] += param.numel()
        
        for module_name, param_count in sorted(module_params.items(), key=lambda x: x[1], reverse=True):
            print(f"{module_name}: {param_count:,} ({param_count/1e6:.2f}M)")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_gradients(model_path=None, dataroot='/path/to/nuScenes/'):
    """检查梯度流"""
    print("\n🔍 检查梯度流...")
    
    try:
        sys.path.append('.')
        from model_front import get_model
        from data.dataset_front import semantic_dataset
        from loss import SimpleLoss
        
        # 创建模型
        args = type('Args', (), {
            'instance_seg': False,
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
        
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"✓ 加载模型: {model_path}")
        
        # 创建数据
        dataset = semantic_dataset(dataroot, 'v1.0-trainval')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        sample = next(iter(dataloader))
        
        # 创建损失函数
        loss_fn = SimpleLoss(pos_weight=2.13)
        
        if torch.cuda.is_available():
            model = model.cuda()
            loss_fn = loss_fn.cuda()
            sample = [s.cuda() if torch.is_tensor(s) else s for s in sample]
        
        # 前向传播
        model.train()
        imgs, lidar_data, binimgs = sample[0], sample[1], sample[2]
        output = model(imgs, lidar_data)
        loss = loss_fn(output, binimgs)
        
        # 反向传播
        loss.backward()
        
        print(f"\n📊 梯度统计 (损失值: {loss.item():.4f}):")
        
        # 检查梯度
        grad_stats = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                grad_stats.append((name, grad_norm, grad_mean, grad_std))
        
        # 按梯度范数排序
        grad_stats.sort(key=lambda x: x[1], reverse=True)
        
        print("\n🔝 梯度范数最大的10个参数:")
        for i, (name, norm, mean, std) in enumerate(grad_stats[:10]):
            print(f"{i+1:2d}. {name:30s} | 范数: {norm:8.4f} | 均值: {mean:8.4f} | 标准差: {std:8.4f}")
        
        print("\n🔻 梯度范数最小的10个参数:")
        for i, (name, norm, mean, std) in enumerate(grad_stats[-10:]):
            print(f"{i+1:2d}. {name:30s} | 范数: {norm:8.4f} | 均值: {mean:8.4f} | 标准差: {std:8.4f}")
        
        # 检查梯度消失/爆炸
        zero_grads = sum(1 for _, norm, _, _ in grad_stats if norm < 1e-7)
        large_grads = sum(1 for _, norm, _, _ in grad_stats if norm > 1.0)
        
        print(f"\n⚠️  梯度诊断:")
        print(f"零梯度参数: {zero_grads}/{len(grad_stats)} ({zero_grads/len(grad_stats)*100:.1f}%)")
        print(f"大梯度参数: {large_grads}/{len(grad_stats)} ({large_grads/len(grad_stats)*100:.1f}%)")
        
        if zero_grads > len(grad_stats) * 0.5:
            print("❌ 可能存在梯度消失问题")
        elif large_grads > len(grad_stats) * 0.1:
            print("❌ 可能存在梯度爆炸问题")
        else:
            print("✅ 梯度流正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 梯度检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def profile_memory(model_name='SuperFusion', batch_size=1):
    """内存使用分析"""
    print(f"\n🔍 内存使用分析 (批次大小: {batch_size})...")
    
    try:
        sys.path.append('.')
        from model_front import get_model
        
        # 记录初始内存
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        print(f"初始CPU内存: {initial_memory:.1f} MB")
        if torch.cuda.is_available():
            print(f"初始GPU内存: {initial_gpu_memory:.1f} MB")
        
        # 创建模型
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
        
        model = get_model(model_name, args)
        
        # 模型加载后内存
        model_memory = process.memory_info().rss / 1024 / 1024
        print(f"模型加载后CPU内存: {model_memory:.1f} MB (+{model_memory-initial_memory:.1f} MB)")
        
        if torch.cuda.is_available():
            model = model.cuda()
            model_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"模型加载到GPU后内存: {model_gpu_memory:.1f} MB (+{model_gpu_memory-initial_gpu_memory:.1f} MB)")
        
        # 创建输入数据
        imgs = torch.randn(batch_size, 6, 3, 256, 704)
        lidar_data = torch.randn(batch_size, 5, 256, 704)
        
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            lidar_data = lidar_data.cuda()
        
        # 输入数据内存
        input_memory = process.memory_info().rss / 1024 / 1024
        if torch.cuda.is_available():
            input_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"输入数据加载后GPU内存: {input_gpu_memory:.1f} MB (+{input_gpu_memory-model_gpu_memory:.1f} MB)")
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(imgs, lidar_data)
        
        # 前向传播后内存
        forward_memory = process.memory_info().rss / 1024 / 1024
        if torch.cuda.is_available():
            forward_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"前向传播后GPU内存: {forward_gpu_memory:.1f} MB (+{forward_gpu_memory-input_gpu_memory:.1f} MB)")
        
        # 反向传播测试
        model.train()
        output = model(imgs, lidar_data)
        loss = output.mean()  # 简单损失
        loss.backward()
        
        # 反向传播后内存
        backward_memory = process.memory_info().rss / 1024 / 1024
        if torch.cuda.is_available():
            backward_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"反向传播后GPU内存: {backward_gpu_memory:.1f} MB (+{backward_gpu_memory-forward_gpu_memory:.1f} MB)")
        
        print(f"\n📊 内存使用总结:")
        print(f"模型参数内存: {model_memory-initial_memory:.1f} MB")
        if torch.cuda.is_available():
            print(f"GPU总内存使用: {backward_gpu_memory:.1f} MB")
            print(f"GPU可用内存: {torch.cuda.get_device_properties(0).total_memory/1024/1024 - backward_gpu_memory:.1f} MB")
        
        # 内存建议
        if torch.cuda.is_available():
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            memory_usage_ratio = backward_gpu_memory / total_gpu_memory
            
            print(f"\n💡 内存使用建议:")
            if memory_usage_ratio > 0.9:
                print("❌ GPU内存使用过高，建议减小批次大小")
                suggested_batch_size = max(1, int(batch_size * 0.8 / memory_usage_ratio))
                print(f"建议批次大小: {suggested_batch_size}")
            elif memory_usage_ratio < 0.5:
                print("✅ GPU内存使用较低，可以增大批次大小")
                suggested_batch_size = int(batch_size * 0.8 / memory_usage_ratio)
                print(f"建议批次大小: {suggested_batch_size}")
            else:
                print("✅ GPU内存使用合理")
        
        return True
        
    except Exception as e:
        print(f"❌ 内存分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader(dataroot, batch_size=4, num_workers=4):
    """测试数据加载器性能"""
    print(f"\n🔍 测试数据加载器 (批次大小: {batch_size}, 工作进程: {num_workers})...")
    
    try:
        sys.path.append('.')
        from data.dataset_front import semantic_dataset
        
        # 创建数据集
        dataset = semantic_dataset(dataroot, 'v1.0-trainval')
        print(f"✓ 数据集大小: {len(dataset)}")
        
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"✓ 数据加载器创建成功")
        print(f"✓ 批次数量: {len(dataloader)}")
        
        # 测试数据加载速度
        print("\n⏱️ 测试数据加载速度...")
        
        start_time = time.time()
        batch_times = []
        
        for i, batch in enumerate(dataloader):
            batch_start = time.time()
            
            # 检查数据形状
            if i == 0:
                imgs, lidar_data, binimgs = batch[0], batch[1], batch[2]
                print(f"✓ 图像形状: {imgs.shape}")
                print(f"✓ 激光雷达形状: {lidar_data.shape}")
                print(f"✓ 标签形状: {binimgs.shape}")
                
                # 检查数据范围
                print(f"✓ 图像范围: [{imgs.min():.3f}, {imgs.max():.3f}]")
                print(f"✓ 激光雷达范围: [{lidar_data.min():.3f}, {lidar_data.max():.3f}]")
                print(f"✓ 标签范围: [{binimgs.min():.3f}, {binimgs.max():.3f}]")
            
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
            
            if i >= 10:  # 只测试前10个批次
                break
        
        total_time = time.time() - start_time
        avg_batch_time = np.mean(batch_times)
        
        print(f"\n📊 数据加载性能:")
        print(f"总时间: {total_time:.2f}s")
        print(f"平均每批次时间: {avg_batch_time:.3f}s")
        print(f"数据加载速度: {batch_size/avg_batch_time:.1f} 样本/秒")
        
        # 性能建议
        if avg_batch_time > 1.0:
            print("\n💡 性能建议:")
            print("❌ 数据加载较慢，建议:")
            print("  1. 增加num_workers数量")
            print("  2. 使用SSD存储数据")
            print("  3. 启用pin_memory")
        else:
            print("\n✅ 数据加载速度正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def visualize_features(model_path=None, dataroot='/path/to/nuScenes/', save_dir='debug_vis'):
    """可视化模型特征"""
    print(f"\n🔍 可视化模型特征...")
    
    try:
        sys.path.append('.')
        from model_front import get_model
        from data.dataset_front import semantic_dataset
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建模型
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
        
        model = get_model('SuperFusion', args)
        
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"✓ 加载模型: {model_path}")
        
        # 创建数据
        dataset = semantic_dataset(dataroot, 'v1.0-trainval')
        sample = dataset[0]
        
        imgs = sample[0].unsqueeze(0)  # 添加batch维度
        lidar_data = sample[1].unsqueeze(0)
        binimgs = sample[2].unsqueeze(0)
        
        if torch.cuda.is_available():
            model = model.cuda()
            imgs = imgs.cuda()
            lidar_data = lidar_data.cuda()
        
        # 注册hook来捕获中间特征
        features = {}
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    features[name] = output.detach().cpu()
            return hook
        
        # 为关键层注册hook
        hooks = []
        for name, module in model.named_modules():
            if any(layer_type in name for layer_type in ['conv', 'bn', 'relu', 'pool']):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(imgs, lidar_data)
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        print(f"✓ 捕获到 {len(features)} 个特征图")
        
        # 可视化特征图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        feature_names = list(features.keys())[:6]  # 只显示前6个
        
        for i, name in enumerate(feature_names):
            feature = features[name]
            if feature.dim() == 4:  # [B, C, H, W]
                # 显示第一个通道
                feature_map = feature[0, 0].numpy()
                axes[i].imshow(feature_map, cmap='viridis')
                axes[i].set_title(f'{name}\n{feature.shape}')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_maps.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 可视化输入和输出
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 输入图像 (第一个相机)
        img = imgs[0, 0].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())  # 归一化到[0,1]
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('输入图像 (相机0)')
        axes[0, 0].axis('off')
        
        # 激光雷达BEV
        lidar_bev = lidar_data[0, 0].cpu().numpy()
        axes[0, 1].imshow(lidar_bev, cmap='viridis')
        axes[0, 1].set_title('激光雷达BEV')
        axes[0, 1].axis('off')
        
        # 预测输出
        pred = torch.sigmoid(output[0, 0]).cpu().numpy()
        axes[1, 0].imshow(pred, cmap='viridis')
        axes[1, 0].set_title('预测输出')
        axes[1, 0].axis('off')
        
        # 真值标签
        gt = binimgs[0, 0].cpu().numpy()
        axes[1, 1].imshow(gt, cmap='viridis')
        axes[1, 1].set_title('真值标签')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'input_output.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 特征可视化保存到: {save_dir}/")
        print(f"  - feature_maps.png: 中间特征图")
        print(f"  - input_output.png: 输入输出对比")
        
        return True
        
    except Exception as e:
        print(f"❌ 特征可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='SuperFusion 调试工具集')
    parser.add_argument('--analyze-model', action='store_true', help='分析模型结构')
    parser.add_argument('--check-gradients', action='store_true', help='检查梯度流')
    parser.add_argument('--profile-memory', action='store_true', help='内存分析')
    parser.add_argument('--test-dataloader', action='store_true', help='测试数据加载器')
    parser.add_argument('--visualize-features', action='store_true', help='可视化特征')
    parser.add_argument('--model-path', type=str, help='模型文件路径')
    parser.add_argument('--dataroot', type=str, default='/path/to/nuScenes/', help='数据集路径')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载进程数')
    parser.add_argument('--save-dir', type=str, default='debug_vis', help='可视化保存目录')
    parser.add_argument('--all', action='store_true', help='运行所有调试工具')
    
    args = parser.parse_args()
    
    if not any([args.analyze_model, args.check_gradients, args.profile_memory, 
                args.test_dataloader, args.visualize_features, args.all]):
        parser.print_help()
        return
    
    print("🔧 SuperFusion 调试工具集")
    print("=" * 50)
    
    # 模型分析
    if args.analyze_model or args.all:
        analyze_model()
    
    # 梯度检查
    if args.check_gradients or args.all:
        check_gradients(args.model_path, args.dataroot)
    
    # 内存分析
    if args.profile_memory or args.all:
        profile_memory(batch_size=args.batch_size)
    
    # 数据加载器测试
    if args.test_dataloader or args.all:
        test_dataloader(args.dataroot, args.batch_size, args.num_workers)
    
    # 特征可视化
    if args.visualize_features or args.all:
        visualize_features(args.model_path, args.dataroot, args.save_dir)
    
    print("\n" + "=" * 50)
    print("🎉 调试工具运行完成!")

if __name__ == '__main__':
    main()