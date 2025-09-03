#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperFusion 真实标注可视化脚本

该脚本用于可视化nuScenes数据集中的真实车道线标注（Ground Truth）。
主要功能：
1. 加载nuScenes数据集
2. 提取真实车道线向量标注
3. 在BEV（鸟瞰图）视角下可视化车道线
4. 保存可视化结果为图片

作者：SuperFusion团队
"""

import argparse
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import tqdm
import torch
import math
from data.dataset_front import semantic_dataset
from data.const import NUM_CLASSES
from model_front import get_model
from postprocess.vectorize import vectorize



def vis_vector(model, val_loader, angle_class):
    """
    可视化车道线向量的主要函数
    
    Args:
        model: 模型对象（在此脚本中未使用，仅用于可视化真实标注）
        val_loader: 验证数据加载器
        angle_class: 角度分类数量
    """
    # model.eval()  # 此脚本仅可视化真实标注，不需要模型推理
    
    # 加载车辆图标，用于在BEV图中标识车辆位置
    car_img = Image.open('pics/car.png')
    
    # 定义不同类型车道线的颜色：红色、蓝色、绿色
    colors_plt = ['r', 'b', 'g']

    with torch.no_grad():
        # 遍历验证数据集中的每个批次
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, segmentation_gt, instance_gt, direction_gt, final_depth_map_bin, final_depth_map_bin_enc, projected_depth, vectors, rec) in enumerate(val_loader):
            """
            数据加载器返回的数据说明：
            - imgs: 多视角相机图像
            - trans, rots: 相机外参（平移和旋转）
            - intrins: 相机内参
            - post_trans, post_rots: 图像预处理后的变换参数
            - lidar_data, lidar_mask: 激光雷达点云数据和掩码
            - car_trans, yaw_pitch_roll: 车辆位置和姿态信息
            - segmentation_gt: 语义分割真实标注
            - instance_gt: 实例分割真实标注
            - direction_gt: 方向预测真实标注
            - final_depth_map_bin, final_depth_map_bin_enc: 深度图相关数据
            - projected_depth: 投影深度图
            - vectors: 车道线向量真实标注（重点数据）
            - rec: 数据记录信息，包含文件路径等元数据
            """
            # 只处理批次中的第一个样本（si=0）
            for si in range(1):
                # 创建matplotlib图形，设置BEV视角的显示范围
                plt.figure(figsize=(4, 2))  # 图形尺寸：宽4英寸，高2英寸
                plt.xlim(0, 90)   # X轴范围：0-90米（车辆前方距离）
                plt.ylim(-15, 15) # Y轴范围：-15到15米（车辆左右距离）
                plt.axis('off')   # 关闭坐标轴显示

                # 遍历当前样本中的所有车道线向量
                for vector in vectors:
                    # 提取车道线向量的关键信息
                    pts = vector['pts']        # 车道线的控制点坐标
                    pts_num = vector['pts_num'] # 有效控制点的数量
                    line_type = vector['type']  # 车道线类型（0,1,2对应不同颜色）
                    
                    # 转换为numpy数组并提取有效的控制点
                    pts = pts[:pts_num].cpu().detach().numpy()
                    pts = pts[0, :]  # 取第一个样本的点
                    
                    # 分离X和Y坐标
                    x = np.array([pt[0] for pt in pts])  # X坐标（前后方向）
                    y = np.array([pt[1] for pt in pts])  # Y坐标（左右方向）
                    
                    # 根据车道线类型选择颜色并绘制
                    plt.plot(x, y, color=colors_plt[line_type])

                # 在BEV图中心位置显示车辆图标
                # extent参数定义车辆图标在BEV坐标系中的位置和大小
                plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
                
                # 打印当前处理的前视相机数据文件名
                print("rec: ", rec['data']['CAM_FRONT'])
                
                # 构建保存路径，包含批次编号和相机数据文件名
                map_path = 'results/' + args.saveroot + \
                    f'/eval_{batchi:04}_' + str(rec['data']['CAM_FRONT']) + '_gt.jpg'
                print('saving', map_path)
                
                # 保存可视化结果
                # bbox_inches='tight': 紧凑布局，去除多余空白
                # dpi=400: 高分辨率输出
                plt.savefig(map_path, bbox_inches='tight', dpi=400)
                plt.close()  # 关闭图形以释放内存


def main(args):
    """
    主函数：配置数据加载器并启动可视化过程
    
    Args:
        args: 命令行参数对象
    """
    # 数据配置字典，定义BEV网格和图像处理参数
    data_conf = {
        'num_channels': NUM_CLASSES + 1,           # 输出通道数（语义类别数+1）
        'image_size': args.image_size,             # 输入图像尺寸 [H, W]
        'depth_image_size': args.depth_image_size, # 深度图像尺寸 [H, W]
        'xbound': args.xbound,                     # BEV X轴边界 [min, max, resolution]
        'ybound': args.ybound,                     # BEV Y轴边界 [min, max, resolution]
        'zbound': args.zbound,                     # BEV Z轴边界 [min, max, resolution]
        'dbound': args.dbound,                     # 深度边界 [min, max, resolution]
        'thickness': args.thickness,               # 车道线厚度
        'angle_class': args.angle_class,           # 角度分类数量
    }

    # 创建数据加载器
    # visual=True 表示用于可视化，会加载向量标注数据
    train_loader, val_loader = semantic_dataset(
        args.version,                              # nuScenes数据集版本
        args.dataroot,                            # 数据集根目录
        data_conf,                                # 数据配置
        args.bsz,                                 # 批次大小
        args.nworkers,                            # 数据加载工作进程数
        depth_downsample_factor=args.depth_downsample_factor,  # 深度图下采样因子
        depth_sup=args.depth_sup,                 # 是否使用深度监督
        use_depth_enc=args.use_depth_enc,         # 是否使用深度编码
        use_depth_enc_bin=args.use_depth_enc_bin, # 是否使用二进制深度编码
        add_depth_channel=args.add_depth_channel, # 是否添加深度通道
        use_lidar_10=args.use_lidar_10,           # 是否使用10线激光雷达
        visual=True                               # 可视化模式，加载向量数据
    )
    
    # 此脚本仅用于可视化真实标注，不需要加载模型
    model = None
    
    # 开始可视化过程
    vis_vector(model, val_loader, args.angle_class)


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='SuperFusion真实标注可视化工具')
    
    # ==================== nuScenes数据集配置 ====================
    parser.add_argument('--dataroot', type=str,
                        default='/media/hao/HaoData/dataset/nuScenes/',
                        help='nuScenes数据集根目录路径')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        choices=['v1.0-trainval', 'v1.0-mini'],
                        help='nuScenes数据集版本选择')

    # ==================== 模型配置 ====================
    parser.add_argument("--model", type=str, default='SuperFusion',
                        help='模型名称（此脚本中未使用）')
    
    # ==================== 数据加载配置 ====================
    parser.add_argument("--bsz", type=int, default=1,
                        help='批次大小（建议设为1以便逐个可视化）')
    parser.add_argument("--nworkers", type=int, default=10,
                        help='数据加载工作进程数')
    
    parser.add_argument('--modelf', type=str, default=None,
                        help='模型文件路径（此脚本中未使用）')

    # ==================== 数据处理配置 ====================
    parser.add_argument("--thickness", type=int, default=5,
                        help='车道线渲染厚度（像素）')
    parser.add_argument("--depth_downsample_factor", type=int, default=4,
                        help='深度图下采样因子')
    parser.add_argument("--image_size", nargs=2, type=int, default=[256, 704],
                        help='输入图像尺寸 [高度, 宽度]')
    parser.add_argument("--depth_image_size", nargs=2, type=int, default=[256, 704],
                        help='深度图像尺寸 [高度, 宽度]')
    
    # ==================== BEV网格配置 ====================
    parser.add_argument("--xbound", nargs=3, type=float,
                        default=[-90.0, 90.0, 0.15],
                        help='BEV X轴边界 [最小值, 最大值, 分辨率(米)]')
    parser.add_argument("--ybound", nargs=3, type=float,
                        default=[-15.0, 15.0, 0.15],
                        help='BEV Y轴边界 [最小值, 最大值, 分辨率(米)]')
    parser.add_argument("--zbound", nargs=3, type=float,
                        default=[-10.0, 10.0, 20.0],
                        help='BEV Z轴边界 [最小值, 最大值, 分辨率(米)]')
    parser.add_argument("--dbound", nargs=3, type=float,
                        default=[2.0, 90.0, 1.0],
                        help='深度边界 [最小值, 最大值, 分辨率(米)]')

    # ==================== 实例分割配置 ====================
    parser.add_argument('--instance_seg', action='store_true',
                        help='是否启用实例分割')
    parser.add_argument("--embedding_dim", type=int, default=16,
                        help='实例嵌入维度')
    
    # ==================== 方向预测配置 ====================
    parser.add_argument('--direction_pred', action='store_true',
                        help='是否启用方向预测')
    parser.add_argument('--angle_class', type=int, default=36,
                        help='角度分类数量（360度/36=10度每类）')
    
    # ==================== 输出配置 ====================
    parser.add_argument('--saveroot', type=str, default='SuperFusion',
                        help='结果保存根目录名称')

    # ==================== 深度和激光雷达配置 ====================
    parser.add_argument('--depth_sup', action='store_true',
                        help='是否使用深度监督学习')
    parser.add_argument('--use_depth_enc', action='store_true',
                        help='是否使用深度编码器')
    parser.add_argument('--pretrained', action='store_true',
                        help='是否使用预训练模型')
    parser.add_argument('--use_depth_enc_bin', action='store_true',
                        help='是否使用二进制深度编码')
    parser.add_argument('--add_depth_channel', action='store_true',
                        help='是否添加深度通道到输入')
    parser.add_argument('--use_lidar_10', action='store_true',
                        help='是否使用10线激光雷达数据')
    
    # 解析命令行参数并运行主函数
    args = parser.parse_args()
    main(args)
