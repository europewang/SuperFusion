#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperFusion 推理示例
展示如何使用预训练模型进行车道线检测

使用方法:
python inference_example.py --model_path runs/model.pt --data_sample sample_001
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import json

# 添加项目路径
sys.path.append('.')
from inference import SuperFusionInference

def create_sample_data(output_dir='sample_data'):
    """
    创建示例输入数据
    
    Args:
        output_dir: 输出目录
    """
    print(f"\n📁 创建示例数据到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建示例图像数据 (模拟6个相机)
    img_size = (256, 704, 3)  # H, W, C
    camera_names = ['front', 'front_right', 'front_left', 'back', 'back_left', 'back_right']
    
    image_paths = []
    for i, cam_name in enumerate(camera_names):
        # 创建模拟图像数据
        img_data = np.random.randint(0, 255, img_size, dtype=np.uint8)
        
        # 添加一些模拟的道路和车道线特征
        # 道路区域 (下半部分)
        img_data[img_size[0]//2:, :] = [100, 100, 100]  # 灰色道路
        
        # 车道线 (白色线条)
        for lane_x in [img_size[1]//4, img_size[1]//2, 3*img_size[1]//4]:
            img_data[img_size[0]//2:, lane_x-2:lane_x+2] = [255, 255, 255]
        
        # 保存图像
        from PIL import Image
        img_path = os.path.join(output_dir, f'{cam_name}.jpg')
        Image.fromarray(img_data).save(img_path)
        image_paths.append(img_path)
        
        print(f"  ✓ 创建 {cam_name} 图像: {img_path}")
    
    # 创建示例激光雷达BEV数据
    bev_size = (200, 200, 5)  # H, W, C (5个特征通道)
    lidar_data = np.random.randn(*bev_size).astype(np.float32)
    
    # 添加一些模拟的车道线特征
    for lane_y in [50, 100, 150]:
        lidar_data[lane_y-2:lane_y+2, :, 0] = 2.0  # 强化第一个通道的车道线信号
    
    lidar_path = os.path.join(output_dir, 'lidar_bev.npy')
    np.save(lidar_path, lidar_data)
    print(f"  ✓ 创建激光雷达BEV数据: {lidar_path}")
    
    return image_paths, lidar_path

def load_nuscenes_sample(dataroot, sample_token):
    """
    从nuScenes数据集加载真实样本
    
    Args:
        dataroot: nuScenes数据集路径
        sample_token: 样本token
        
    Returns:
        tuple: (image_paths, lidar_path)
    """
    try:
        from nuscenes.nuscenes import NuScenes
        from data.dataset_front import semantic_dataset
        
        print(f"\n📁 从nuScenes加载样本: {sample_token}")
        
        # 初始化nuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
        
        # 获取样本
        sample = nusc.get('sample', sample_token)
        
        # 获取相机数据
        camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                       'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        
        image_paths = []
        for cam_name in camera_names:
            cam_token = sample['data'][cam_name]
            cam_data = nusc.get('sample_data', cam_token)
            img_path = os.path.join(dataroot, cam_data['filename'])
            image_paths.append(img_path)
            print(f"  ✓ {cam_name}: {img_path}")
        
        # 获取激光雷达数据 (这里需要预处理为BEV格式)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        lidar_path = os.path.join(dataroot, lidar_data['filename'])
        
        print(f"  ✓ LIDAR: {lidar_path}")
        print(f"  ⚠️ 注意: 激光雷达数据需要预处理为BEV格式")
        
        return image_paths, lidar_path
        
    except Exception as e:
        print(f"❌ 加载nuScenes样本失败: {e}")
        return None, None

def run_inference_example(model_path, image_paths, lidar_path, output_dir='inference_results'):
    """
    运行推理示例
    
    Args:
        model_path: 模型路径
        image_paths: 图像路径列表
        lidar_path: 激光雷达数据路径
        output_dir: 输出目录
    """
    print(f"\n🚀 开始推理示例")
    print(f"模型路径: {model_path}")
    print(f"输出目录: {output_dir}")
    
    try:
        # 初始化推理器
        inferencer = SuperFusionInference(model_path)
        
        # 执行推理
        print("\n🔍 执行推理...")
        results = inferencer.predict(image_paths, lidar_path)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存原始结果
        result_file = os.path.join(output_dir, 'raw_results.npz')
        save_data = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    save_data[f"{key}_{sub_key}"] = sub_value
            else:
                save_data[key] = value
        
        np.savez(result_file, **save_data)
        print(f"✓ 原始结果保存到: {result_file}")
        
        # 生成可视化
        print("\n🎨 生成可视化...")
        vis_path = os.path.join(output_dir, 'lane_detection.png')
        inferencer.visualize_results(results, vis_path)
        
        # 矢量化车道线
        print("\n📐 矢量化车道线...")
        lanes = inferencer.vectorize_lanes(results)
        
        # 保存矢量结果
        vector_file = os.path.join(output_dir, 'detected_lanes.json')
        lane_data = {
            'lanes': lanes,
            'num_lanes': len(lanes),
            'bev_params': inferencer.bev_params,
            'metadata': {
                'model_path': model_path,
                'input_images': image_paths,
                'input_lidar': lidar_path
            }
        }
        
        with open(vector_file, 'w') as f:
            json.dump(lane_data, f, indent=2)
        
        print(f"✓ 检测到 {len(lanes)} 条车道线")
        print(f"✓ 矢量结果保存到: {vector_file}")
        
        # 打印统计信息
        seg_shape = results['segmentation']['shape']
        seg_binary = results['segmentation']['binary']
        
        print(f"\n📊 检测统计:")
        print(f"分割图尺寸: {seg_shape}")
        print(f"车道线像素数: {np.sum(seg_binary)}")
        print(f"车道线覆盖率: {np.sum(seg_binary) / np.prod(seg_shape[1:]) * 100:.2f}%")
        
        # 如果有多个输出，显示额外信息
        if 'embedding' in results:
            print(f"嵌入特征维度: {results['embedding'].shape}")
        if 'direction' in results:
            print(f"方向预测维度: {results['direction'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='SuperFusion推理示例')
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--data_sample', type=str, help='样本数据目录或nuScenes样本token')
    parser.add_argument('--dataroot', type=str, help='nuScenes数据集路径 (如果使用真实数据)')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='输出目录')
    parser.add_argument('--create_sample', action='store_true', help='创建示例数据')
    
    args = parser.parse_args()
    
    print("🚀 SuperFusion推理示例")
    print("=" * 50)
    
    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"❌ 模型文件不存在: {args.model_path}")
        print("\n💡 获取预训练模型的方法:")
        print("1. 从官方下载: https://drive.google.com/file/d/1UTgughJ71Rn0zPUDTXFo__HJS-57lwNG/view")
        print("2. 自己训练: python train.py --dataroot /path/to/nuScenes/ --instance_seg --direction_pred --depth_sup --pretrained")
        return 1
    
    # 准备输入数据
    image_paths = None
    lidar_path = None
    
    if args.create_sample or not args.data_sample:
        # 创建示例数据
        sample_dir = args.data_sample or 'sample_data'
        image_paths, lidar_path = create_sample_data(sample_dir)
        
    elif args.dataroot and len(args.data_sample) == 32:  # nuScenes样本token长度
        # 从nuScenes加载真实数据
        image_paths, lidar_path = load_nuscenes_sample(args.dataroot, args.data_sample)
        if image_paths is None:
            print("❌ 无法加载nuScenes数据，将创建示例数据")
            image_paths, lidar_path = create_sample_data('sample_data')
    
    elif os.path.isdir(args.data_sample):
        # 从指定目录加载数据
        print(f"\n📁 从目录加载数据: {args.data_sample}")
        
        # 查找图像文件
        camera_names = ['front', 'front_right', 'front_left', 'back', 'back_left', 'back_right']
        image_paths = []
        
        for cam_name in camera_names:
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = os.path.join(args.data_sample, f'{cam_name}{ext}')
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    break
            else:
                print(f"❌ 找不到 {cam_name} 图像")
                return 1
        
        # 查找激光雷达文件
        lidar_path = os.path.join(args.data_sample, 'lidar_bev.npy')
        if not os.path.exists(lidar_path):
            print(f"❌ 找不到激光雷达文件: {lidar_path}")
            return 1
    
    else:
        print(f"❌ 无效的数据样本: {args.data_sample}")
        return 1
    
    # 验证输入数据
    if len(image_paths) != 6:
        print(f"❌ 需要6个相机图像，但找到{len(image_paths)}个")
        return 1
    
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"❌ 图像文件不存在: {img_path}")
            return 1
    
    if not os.path.exists(lidar_path):
        print(f"❌ 激光雷达文件不存在: {lidar_path}")
        return 1
    
    # 运行推理
    success = run_inference_example(args.model_path, image_paths, lidar_path, args.output_dir)
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 推理示例完成!")
        print(f"📁 结果保存在: {args.output_dir}/")
        print("\n📋 输出文件:")
        print(f"  - raw_results.npz: 原始预测结果")
        print(f"  - lane_detection.png: 可视化图像")
        print(f"  - detected_lanes.json: 矢量化车道线")
        
        print("\n💡 使用建议:")
        print("1. 查看可视化图像了解检测效果")
        print("2. 检查JSON文件中的车道线坐标")
        print("3. 根据需要调整后处理参数")
        
        return 0
    else:
        return 1

if __name__ == '__main__':
    exit(main())