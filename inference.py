#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperFusion 推理脚本
直接调用预训练模型，输入点云+图片，输出车道线

使用方法:
python inference.py --model_path runs/model.pt --input_data sample_data/ --output_dir results/
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
import json
from PIL import Image
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append('.')

class SuperFusionInference:
    """SuperFusion推理类"""
    
    def __init__(self, model_path, device='cuda'):
        """
        初始化推理器
        
        Args:
            model_path: 预训练模型路径
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # BEV参数 (与训练时保持一致)
        self.bev_params = {
            'xbound': [-50.0, 50.0, 0.5],
            'ybound': [-50.0, 50.0, 0.5],
            'zbound': [-10.0, 10.0, 20.0],
            'dbound': [4.0, 45.0, 1.0]
        }
        
        # 图像参数
        self.img_size = (256, 704)  # (H, W)
        
        print(f"✓ SuperFusion推理器初始化完成")
        print(f"✓ 设备: {self.device}")
        print(f"✓ 模型参数量: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")
    
    def _load_model(self, model_path):
        """加载预训练模型"""
        from model_front import get_model
        
        # 创建模型配置 (与训练时保持一致)
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
        
        # 加载权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✓ 加载模型权重: {model_path}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        return model.to(self.device)
    
    def preprocess_images(self, image_paths):
        """
        预处理图像数据
        
        Args:
            image_paths: 6个相机图像路径列表
            
        Returns:
            torch.Tensor: 预处理后的图像张量 [1, 6, 3, H, W]
        """
        if len(image_paths) != 6:
            raise ValueError(f"需要6个相机图像，但提供了{len(image_paths)}个")
        
        images = []
        for img_path in image_paths:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"图像文件不存在: {img_path}")
            
            # 读取图像
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.img_size[1], self.img_size[0]))  # (W, H)
            
            # 转换为numpy数组并归一化
            img = np.array(img).astype(np.float32) / 255.0
            
            # 标准化 (ImageNet统计)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img - mean) / std
            
            # 转换为CHW格式
            img = img.transpose(2, 0, 1)
            images.append(img)
        
        # 堆叠为张量
        images = np.stack(images, axis=0)  # [6, 3, H, W]
        images = torch.from_numpy(images).unsqueeze(0)  # [1, 6, 3, H, W]
        
        return images.to(self.device)
    
    def preprocess_lidar(self, lidar_path):
        """
        预处理激光雷达数据
        
        Args:
            lidar_path: 激光雷达BEV特征文件路径 (.npy格式)
            
        Returns:
            torch.Tensor: 预处理后的激光雷达张量 [1, C, H, W]
        """
        if not os.path.exists(lidar_path):
            raise FileNotFoundError(f"激光雷达文件不存在: {lidar_path}")
        
        # 加载激光雷达BEV特征
        if lidar_path.endswith('.npy'):
            lidar_data = np.load(lidar_path)
        else:
            raise ValueError(f"不支持的激光雷达文件格式: {lidar_path}")
        
        # 确保数据格式正确
        if lidar_data.ndim == 3:  # [C, H, W]
            lidar_data = lidar_data[np.newaxis, ...]  # [1, C, H, W]
        elif lidar_data.ndim == 2:  # [H, W]
            lidar_data = lidar_data[np.newaxis, np.newaxis, ...]  # [1, 1, H, W]
        
        # 转换为张量
        lidar_tensor = torch.from_numpy(lidar_data.astype(np.float32))
        
        return lidar_tensor.to(self.device)
    
    def predict(self, image_paths, lidar_path):
        """
        执行推理预测
        
        Args:
            image_paths: 6个相机图像路径列表
            lidar_path: 激光雷达BEV特征文件路径
            
        Returns:
            dict: 预测结果字典
        """
        # 预处理输入数据
        images = self.preprocess_images(image_paths)
        lidar_data = self.preprocess_lidar(lidar_path)
        
        print(f"✓ 图像数据形状: {images.shape}")
        print(f"✓ 激光雷达数据形状: {lidar_data.shape}")
        
        # 模型推理
        with torch.no_grad():
            output = self.model(images, lidar_data)
        
        # 后处理预测结果
        results = self._postprocess_output(output)
        
        return results
    
    def _postprocess_output(self, output):
        """
        后处理模型输出
        
        Args:
            output: 模型原始输出
            
        Returns:
            dict: 处理后的结果
        """
        results = {}
        
        # 语义分割结果
        if isinstance(output, (list, tuple)):
            seg_output = output[0]
        else:
            seg_output = output
        
        # 应用sigmoid激活
        seg_prob = torch.sigmoid(seg_output)
        
        # 转换为numpy数组
        seg_prob_np = seg_prob.cpu().numpy()[0]  # [C, H, W]
        
        # 二值化 (阈值0.5)
        seg_binary = (seg_prob_np > 0.5).astype(np.uint8)
        
        results['segmentation'] = {
            'probability': seg_prob_np,
            'binary': seg_binary,
            'shape': seg_prob_np.shape
        }
        
        # 如果有多个输出，处理其他任务
        if isinstance(output, (list, tuple)) and len(output) > 1:
            # 实例分割
            if len(output) > 1:
                embedding = output[1].cpu().numpy()[0]
                results['embedding'] = embedding
            
            # 方向预测
            if len(output) > 2:
                direction = output[2].cpu().numpy()[0]
                results['direction'] = direction
        
        return results
    
    def visualize_results(self, results, save_path=None):
        """
        可视化预测结果
        
        Args:
            results: 预测结果字典
            save_path: 保存路径
        """
        seg_data = results['segmentation']
        
        # 创建可视化图像
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 概率图
        if seg_data['probability'].shape[0] > 1:
            # 多类别，显示第一个类别
            prob_map = seg_data['probability'][0]
        else:
            prob_map = seg_data['probability'][0]
        
        axes[0].imshow(prob_map, cmap='viridis')
        axes[0].set_title('车道线概率图')
        axes[0].axis('off')
        
        # 二值化结果
        if seg_data['binary'].shape[0] > 1:
            binary_map = seg_data['binary'][0]
        else:
            binary_map = seg_data['binary'][0]
        
        axes[1].imshow(binary_map, cmap='gray')
        axes[1].set_title('车道线检测结果')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 可视化结果保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def vectorize_lanes(self, results):
        """
        将分割结果矢量化为车道线
        
        Args:
            results: 预测结果字典
            
        Returns:
            list: 车道线矢量列表
        """
        try:
            from postprocess.vectorize import vectorize
            
            seg_binary = results['segmentation']['binary']
            
            # 如果有嵌入特征，使用完整的矢量化
            if 'embedding' in results:
                embedding = results['embedding']
                direction = results.get('direction', None)
                
                # 调用矢量化函数
                vectors = vectorize(
                    seg_binary[0] if seg_binary.shape[0] > 1 else seg_binary,
                    embedding,
                    direction
                )
            else:
                # 简单的轮廓提取
                vectors = self._simple_vectorize(seg_binary[0] if seg_binary.shape[0] > 1 else seg_binary)
            
            return vectors
            
        except ImportError:
            print("⚠️ 矢量化模块不可用，使用简单轮廓提取")
            return self._simple_vectorize(seg_binary[0] if seg_binary.shape[0] > 1 else seg_binary)
    
    def _simple_vectorize(self, binary_mask):
        """
        简单的轮廓提取矢量化
        
        Args:
            binary_mask: 二值化掩码
            
        Returns:
            list: 轮廓点列表
        """
        import cv2
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 转换为车道线格式
        lanes = []
        for contour in contours:
            if len(contour) > 10:  # 过滤太短的轮廓
                # 简化轮廓
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # 转换为BEV坐标
                lane_points = []
                for point in approx:
                    x, y = point[0]
                    # 转换为实际坐标 (根据BEV参数)
                    real_x = (x / binary_mask.shape[1]) * (self.bev_params['xbound'][1] - self.bev_params['xbound'][0]) + self.bev_params['xbound'][0]
                    real_y = (y / binary_mask.shape[0]) * (self.bev_params['ybound'][1] - self.bev_params['ybound'][0]) + self.bev_params['ybound'][0]
                    lane_points.append([real_x, real_y])
                
                lanes.append(lane_points)
        
        return lanes

def main():
    parser = argparse.ArgumentParser(description='SuperFusion推理脚本')
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--images', type=str, nargs=6, required=True, 
                       help='6个相机图像路径 (按顺序: front, front_right, front_left, back, back_left, back_right)')
    parser.add_argument('--lidar', type=str, required=True, help='激光雷达BEV特征文件路径')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='输出目录')
    parser.add_argument('--visualize', action='store_true', help='可视化结果')
    parser.add_argument('--vectorize', action='store_true', help='矢量化车道线')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='计算设备')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 SuperFusion推理开始")
    print("=" * 50)
    
    try:
        # 初始化推理器
        inferencer = SuperFusionInference(args.model_path, args.device)
        
        # 执行推理
        print("\n🔍 执行推理...")
        results = inferencer.predict(args.images, args.lidar)
        
        print(f"✓ 推理完成")
        print(f"✓ 分割结果形状: {results['segmentation']['shape']}")
        
        # 保存结果
        result_file = os.path.join(args.output_dir, 'prediction_results.npz')
        np.savez(result_file, **results['segmentation'])
        print(f"✓ 结果保存到: {result_file}")
        
        # 可视化
        if args.visualize:
            print("\n🎨 生成可视化...")
            vis_path = os.path.join(args.output_dir, 'visualization.png')
            inferencer.visualize_results(results, vis_path)
        
        # 矢量化
        if args.vectorize:
            print("\n📐 矢量化车道线...")
            lanes = inferencer.vectorize_lanes(results)
            
            # 保存矢量结果
            vector_file = os.path.join(args.output_dir, 'lane_vectors.json')
            with open(vector_file, 'w') as f:
                json.dump({
                    'lanes': lanes,
                    'bev_params': inferencer.bev_params,
                    'num_lanes': len(lanes)
                }, f, indent=2)
            
            print(f"✓ 检测到 {len(lanes)} 条车道线")
            print(f"✓ 矢量结果保存到: {vector_file}")
        
        print("\n" + "=" * 50)
        print("🎉 推理完成!")
        print(f"📁 结果保存在: {args.output_dir}/")
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())