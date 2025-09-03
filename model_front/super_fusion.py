import torch
from torch import nn
import math

from .pointpillar import PointPillarEncoder
from .base import BevEncode
from data.utils import gen_dx_bx

from .ddn_deeplabv3 import DDNDeepLabV3
import torch.nn.functional as F
from inplace_abn import InPlaceABNSync

"""
SuperFusion: 多级激光雷达-相机融合网络
论文: SuperFusion：用于长距离高清地图生成的多级激光雷达-相机融合技术

本文件实现了论文中的核心模块：
1. 深度感知的相机到BEV变换模块 (LiftSplatShoot)
2. 图像引导的激光雷达BEV预测模块 (LidarPred)
3. BEV对齐与融合模块 (AlignFAnew)
4. 多级融合架构 (SuperFusion)
"""

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class BasicBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Initializes convolutional block for channel reduce
        Args:
            out_channels [int]: Number of output channels of convolutional block
            **kwargs [Dict]: Extra arguments for nn.Conv2d
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        """
        Applies convolutional block
        Args:
            features [torch.Tensor(B, C_in, H, W)]: Input features
        Returns:
            x [torch.Tensor(B, C_out, H, W)]: Output features
        """
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CamEncode(nn.Module):
    """
    相机编码器模块 - 实现论文3.1节的深度感知相机到BEV变换
    
    该模块包含两个分支：
    1. 第一个分支提取2D图像特征 F ∈ R^{W_F × H_F × C_F}
    2. 第二个分支连接深度预测网络，估计类别深度分布 D ∈ R^{W_F × H_F × D}
    
    输入: RGB图像 I 与稀疏深度图 D_sparse (通过激光雷达投影生成)
    输出: 图像特征 F 和深度分布 D
    """
    def __init__(self, D, C, use_depth_enc=False, pretrained=True, add_depth_channel=False):
        super(CamEncode, self).__init__()
        self.D = D  # 离散深度区间数量 (论文中设置为2.0-90.0米，间隔1.0米)
        self.C = C  # 特征通道数
        self.use_depth_enc = use_depth_enc  # 是否使用深度编码器
        self.add_depth_channel = add_depth_channel  # 是否添加深度通道

        # 使用预训练的DeepLabV3作为骨干网络，支持深度编码输入
        if pretrained:
            print("use pretrain")
            self.ddn = DDNDeepLabV3(
                num_classes=self.D + 1,
                backbone_name="ResNet101",
                feat_extract_layer="layer1",
                pretrained_path="checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth",
                use_depth_enc = use_depth_enc,
                add_depth_channel = add_depth_channel
            )
        else:
            print("no pretrain")
            self.ddn = DDNDeepLabV3(
                num_classes=self.D + 1,
                backbone_name="ResNet101",
                feat_extract_layer="layer1",
                #pretrained_path="checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth",
                use_depth_enc = use_depth_enc,
                add_depth_channel = add_depth_channel
            )
        
        # 通道降维模块，将256维特征降至64维
        self.channel_reduce = BasicBlock2D(256, 64)
        
        # 深度编码器：将稀疏深度图编码为高维特征
        if self.use_depth_enc:
            self.depth_enc = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

    def get_depth_dist(self, x, eps=1e-20):
        """获取深度概率分布，使用softmax归一化"""
        return x.softmax(dim=1)

    def get_depth_feat(self, x, depth, projected_depth):
        """
        提取深度特征并生成深度分布
        
        Args:
            x: RGB图像特征
            depth: 稀疏深度图 (用于深度编码)
            projected_depth: 投影深度图 (用于深度通道)
        
        Returns:
            depth_logits: 深度预测logits
            frustum_features: 视锥特征 (图像特征与深度分布的乘积)
            image_features_out: 输出的图像特征
        """
        # 根据不同配置处理深度信息
        if self.use_depth_enc:
            # 使用深度编码器处理稀疏深度图
            encoded_depth = self.depth_enc(depth)
            ddn_result = self.ddn(x, encoded_depth)
        elif self.add_depth_channel:
            # 直接添加深度通道
            ddn_result = self.ddn(x, projected_depth=projected_depth)
        else:
            # 仅使用RGB图像
            ddn_result = self.ddn(x)
            
        image_features = ddn_result["features"]  # 图像特征 F
        depth_logits = ddn_result["logits"]      # 深度预测logits
        image_features_out = ddn_result["image_features_out"]
        
        # 通道降维
        if self.channel_reduce is not None:
            image_features = self.channel_reduce(image_features)

        # 创建视锥特征：将图像特征与深度分布相乘
        frustum_features = self.create_frustum_features(image_features=image_features,
                                                        depth_logits=depth_logits)

        return depth_logits, frustum_features, image_features_out

    def create_frustum_features(self, image_features, depth_logits):
        """
        创建图像深度特征体积 - 实现论文中的视锥特征生成
        
        通过将图像特征与深度分类分数相乘来创建3D视锥特征：
        F_frustum = F_image ⊗ P_depth
        
        Args:
            image_features [torch.Tensor(N, C, H, W)]: 图像特征 F
            depth_logits [torch.Tensor(N, D, H, W)]: 深度分类logits
        Returns:
            frustum_features [torch.Tensor(N, C, D, H, W)]: 视锥特征体积
        """
        channel_dim = 1
        depth_dim = 2

        # 调整维度以匹配广播
        image_features = image_features.unsqueeze(depth_dim)  # (N, C, 1, H, W)
        depth_logits = depth_logits.unsqueeze(channel_dim)    # (N, 1, D, H, W)

        # 沿深度轴应用softmax并移除最后一个深度类别 (> 最大范围)
        depth_probs = F.softmax(depth_logits, dim=depth_dim)  # 深度概率分布
        depth_probs = depth_probs[:, :, :-1]  # 移除超出范围的深度类别

        # 相乘形成图像深度特征体积 (视锥特征)
        frustum_features = depth_probs * image_features
        return frustum_features

    def forward(self, x, depth_enc, projected_depth):
        """
        前向传播：同时输出图像特征和深度分布
        
        Args:
            x: RGB图像输入
            depth_enc: 用于深度编码的稀疏深度图
            projected_depth: 投影深度图
            
        Returns:
            x: 视锥特征 (图像特征与深度分布的融合)
            depth: 深度分布预测
            image_features_out: 输出的图像特征 (用于后续跨模态注意力)
        """
        depth, x, image_features_out = self.get_depth_feat(x, depth_enc, projected_depth)

        return x, depth, image_features_out


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class LiftSplatShoot(nn.Module):
    """
    Lift-Splat-Shoot模块 - 实现论文3.1节的深度感知相机到BEV变换
    
    该模块实现了从相机透视图到鸟瞰图(BEV)的变换，包括：
    1. 创建视锥体几何结构
    2. 将图像特征提升到3D空间
    3. 通过体素池化投影到BEV空间
    
    对应论文公式(1): M(u,v) = D(u,v) ⊗ F(u,v)
    其中 M ∈ R^{W_F × H_F × D × C_F} 是视锥体特征网格
    """
    def __init__(self, grid_conf, camC=64, downsample=16, use_depth_enc=False, pretrained=True, add_depth_channel=False):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf

        # 调整x边界，确保从0开始
        self.grid_conf['xbound'] = [
            0.0, self.grid_conf['xbound'][1], self.grid_conf['xbound'][2]]

        # 生成BEV网格参数：dx(分辨率), bx(边界), nx(网格数量)
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)  # BEV网格分辨率
        self.bx = nn.Parameter(bx, requires_grad=False)  # BEV网格边界
        self.nx = nn.Parameter(nx, requires_grad=False)  # BEV网格数量

        self.downsample = downsample  # 图像下采样倍数
        self.camC = camC  # 相机特征通道数
        # 创建视锥体几何结构
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape  # 深度区间数量
        # 相机编码器：提取图像特征和深度分布
        self.camencode = CamEncode(self.D, self.camC, use_depth_enc=use_depth_enc, pretrained=pretrained, add_depth_channel=add_depth_channel)

        # 选择使用QuickCumsum优化或标准autograd
        self.use_quickcumsum = True

    def create_frustum(self):
        """
        创建视锥体几何结构 - 为每个像素生成3D射线
        
        在图像平面创建网格，为每个像素和深度区间生成3D坐标。
        这是实现相机到BEV变换的几何基础。
        
        Returns:
            frustum: 视锥体坐标 [D, H, W, 3] (x, y, depth)
        """
        # 获取图像尺寸
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        
        # 创建深度网格：从dbound[0]到dbound[1]，步长为dbound[2]
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        
        # 创建x坐标网格 (图像宽度方向)
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        # 创建y坐标网格 (图像高度方向)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # 组合成视锥体坐标 [D, H, W, 3]
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """
        确定点云中每个点在ego坐标系中的(x,y,z)位置
        
        实现从相机坐标系到ego坐标系的几何变换，包括：
        1. 撤销图像后处理变换
        2. 从图像坐标转换到相机坐标
        3. 从相机坐标转换到ego坐标
        
        Args:
            rots: 旋转矩阵 [B, N, 3, 3]
            trans: 平移向量 [B, N, 3]
            intrins: 相机内参矩阵 [B, N, 3, 3]
            post_rots: 后处理旋转矩阵 [B, N, 3, 3]
            post_trans: 后处理平移向量 [B, N, 3]
            
        Returns:
            points: ego坐标系中的3D点 [B, N, D, H, W, 3]
        """
        B, N, _ = trans.shape

        # 1. 撤销后处理变换 (数据增强的逆变换)
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # 2. 从图像坐标转换到相机坐标系
        # 将2D像素坐标(u,v)和深度d转换为3D相机坐标(x,y,z)
        # x = (u - cx) * d / fx, y = (v - cy) * d / fy, z = d
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        
        # 3. 从相机坐标系转换到ego坐标系
        # 应用外参变换: P_ego = R * K^(-1) * P_cam + T
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x, depth_enc, projected_depth):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x, depth, image_features_out = self.camencode(x, depth_enc, projected_depth)
        x = x.view(B, N, self.camC, self.D, imH //
                   self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x, depth, image_features_out

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros(
            (B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2],
              geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans, depth_enc, projected_depth):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x, depth, image_features_out = self.get_cam_feats(x, depth_enc, projected_depth)

        x = self.voxel_pooling(geom, x)

        return x, depth, image_features_out

    def forward(self, x, rots, trans, intrins, post_rots, post_trans, depth_enc, projected_depth):
        x, depth, image_features_out = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans, depth_enc, projected_depth)
        return x.transpose(3, 2), depth, image_features_out

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(
        pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :,
        :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class LidarPred(nn.Module):
    """
    图像引导的激光雷达BEV预测模块 - 实现论文3.2节
    
    该模块通过编码器-解码器架构和交叉注意力机制，
    利用图像特征引导激光雷达BEV特征的长距离预测。
    
    对应论文公式(2): A = Attention(Q, K, V) = softmax(QK^T/√d_k)V
    其中 Q来自激光雷达BEV特征，K和V来自图像特征
    """
    def __init__(self, use_cross=False, num_heads=1, pos_emd=True, neck_dim=256, cross_dim=256):
        super(LidarPred, self).__init__()
        self.use_cross = use_cross  # 是否使用交叉注意力机制
        self.pos_emd = pos_emd      # 是否使用位置编码
        
        # 编码器：将激光雷达BEV特征逐步压缩
        self.conv11 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 第一层编码
        self.bn11 = nn.BatchNorm2d(256)

        self.conv21 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # 第二层编码
        self.bn21 = nn.BatchNorm2d(256)

        self.conv41 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # 瓶颈层
        self.bn41 = nn.BatchNorm2d(256)

        # 解码器：从瓶颈特征重建完整BEV特征
        self.conv41d = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # 瓶颈解码
        self.bn41d = nn.BatchNorm2d(256)

        self.conv21d = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # 第二层解码
        self.bn21d = nn.BatchNorm2d(256)

        self.conv11d = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # 输出层

        # 池化和反池化：保持空间结构信息
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # 交叉注意力机制：实现图像特征对激光雷达特征的引导
        if self.use_cross:
            # 图像特征降维：从2048维降至neck_dim维
            self.conv_reduce = nn.Conv2d(2048, neck_dim, kernel_size=1)
            
            # 位置编码：为激光雷达和图像特征添加空间位置信息
            if pos_emd:
                self.pe_lidar = positionalencoding2d(
                    neck_dim, 25*2, 75*2).cuda()  # 激光雷达BEV位置编码
                self.pe_img = positionalencoding2d(
                    neck_dim, 32, 88).cuda()      # 图像特征位置编码
            
            # 多头注意力：实现论文公式(2)的交叉注意力计算
            self.multihead_attn = nn.MultiheadAttention(
                    neck_dim, num_heads, dropout=0.3, batch_first=True)
            
            # 特征融合卷积：将注意力输出与原始特征融合
            self.conv = nn.Sequential(
                        nn.Conv2d(neck_dim+cross_dim,
                                  neck_dim, kernel_size=3, padding=1),
                        nn.BatchNorm2d(neck_dim),
                        nn.ReLU(inplace=True)
                    )
            
            # 交叉注意力输出变换
            self.conv_cross = nn.Sequential(
                nn.Conv2d(neck_dim,
                            cross_dim, kernel_size=1),
                nn.BatchNorm2d(cross_dim),
                nn.ReLU(inplace=True)
            )
            
    def cross_attention(self, x, img_feature):
        """
        交叉注意力机制 - 实现论文公式(2)的图像引导激光雷达特征预测
        
        通过计算激光雷达特征(Query)与图像特征(Key, Value)之间的注意力，
        实现图像信息对激光雷达BEV特征的引导和增强。
        
        Args:
            x: 激光雷达BEV特征 [B, C, H, W]
            img_feature: 图像特征 [B, C, H', W']
            
        Returns:
            fused_feature: 融合后的特征 [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 添加位置编码：为特征添加空间位置信息
        if self.pos_emd:
            pe_lidar = self.pe_lidar.repeat(B, 1, 1, 1)  # 激光雷达位置编码
            pe_img = self.pe_img.repeat(B, 1, 1, 1)      # 图像位置编码
            x = x + pe_lidar
            img_feature = img_feature + pe_img
        
        # 重塑特征为序列格式以适配注意力计算
        query = x.reshape(B, C, -1).permute(0, 2, 1)           # 激光雷达特征作为Query
        key = img_feature.reshape(B, C, -1).permute(0, 2, 1)   # 图像特征作为Key
        
        # 执行多头交叉注意力计算：A = Attention(Q, K, V)
        attn_output, attn_output_weights = self.multihead_attn(
            query, key, key)  # Value与Key相同
        
        # 恢复空间维度
        attn_output = attn_output.permute(0, 2, 1).reshape(B, C, H, W)
        attn_output = self.conv_cross(attn_output)
        
        # 特征融合：将注意力输出与原始激光雷达特征拼接
        fused_feature = torch.cat([x, attn_output], dim=1)
        fused_feature = self.conv(fused_feature)
        
        return fused_feature

    def forward(self, x, img_feature=None):
        """
        前向传播：实现图像引导的激光雷达BEV特征预测
        
        通过编码器-解码器架构和交叉注意力机制，
        利用图像特征引导激光雷达BEV特征的长距离预测。
        
        Args:
            x: 输入的激光雷达BEV特征 [B, 128, H, W]
            img_feature: 图像特征 [B, 2048, H', W'] (可选)
            
        Returns:
            x11d: 预测的激光雷达BEV特征 [B, 128, H, W]
        """
        # 编码阶段：逐步压缩特征并保存池化索引
        x11 = F.relu(self.bn11(self.conv11(x)))     # 第一层编码
        x1p, id1 = self.max_pool(x11)               # 第一次下采样

        x21 = F.relu(self.bn21(self.conv21(x1p)))   # 第二层编码
        x2p, id2 = self.max_pool(x21)               # 第二次下采样

        x41 = F.relu(self.bn41(self.conv41(x2p)))   # 瓶颈层特征

        # 交叉注意力阶段：利用图像特征引导激光雷达特征
        if self.use_cross:
            img_feature = self.conv_reduce(img_feature)  # 图像特征降维
            x41 = self.cross_attention(x41, img_feature) # 应用交叉注意力

        # 解码阶段：逐步恢复特征分辨率
        x41d = F.relu(self.bn41d(self.conv41d(x41))) # 瓶颈解码

        x3d = self.max_unpool(x41d, id2)            # 第一次上采样
        x21d = F.relu(self.bn21d(self.conv21d(x3d))) # 第二层解码

        x2d = self.max_unpool(x21d, id1)            # 第二次上采样
        x11d = self.conv11d(x2d)                    # 输出层

        return x11d


class AlignFAnew(nn.Module):
    """
    BEV对齐与融合模块 - 实现论文3.3节的BEV对齐与融合机制
    
    该模块负责对齐来自不同传感器的BEV特征，并将它们融合为统一的表示。
    通过可学习的空间变换和特征融合，实现激光雷达和相机BEV特征的有效结合。
    
    对应论文公式(3): F_fused = Fusion(Align(F_lidar), Align(F_camera))
    其中Align函数通过可学习的空间变换确保特征在空间和语义上的一致性
    """
    def __init__(self, features):
        super(AlignFAnew, self).__init__()

        # 空间变换参数生成器：学习特征对齐所需的空间偏移
        # 对应论文公式(3)中的Align函数的实现
        self.delta_gen1 = nn.Sequential(
                        nn.Conv2d(features, 128, kernel_size=1, bias=False),
                        InPlaceABNSync(128),
                        nn.Conv2d(128, 2, kernel_size=3, padding=1, bias=False)  # 输出2通道的空间偏移
                        )

        # 初始化空间偏移为零，确保训练初期的稳定性
        self.delta_gen1[2].weight.data.zero_()

    def bilinear_interpolate_torch_gridsample2(self, input, size, delta=0):
        """
        可学习的双线性插值 - 实现基于空间偏移的特征对齐
        
        通过学习到的空间偏移delta对输入特征进行空间变换，
        实现不同传感器BEV特征之间的精确对齐。
        
        Args:
            input: 输入特征图 [B, C, H, W]
            size: 输出尺寸 (H, W)
            delta: 学习到的空间偏移 [B, 2, H, W]
            
        Returns:
            output: 对齐后的特征图 [B, C, H, W]
        """
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 2.0
        
        # 归一化因子：将像素坐标归一化到[-1, 1]范围
        norm = torch.tensor([[[[(out_w-1)/s, (out_h-1)/s]]]]).type_as(input).to(input.device) 
        
        # 创建标准网格坐标
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        
        # 应用学习到的空间偏移进行特征对齐
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        # 执行双线性插值采样
        output = F.grid_sample(input, grid, align_corners=True)
        return output

    def forward(self, low_stage, high_stage):
        """
        前向传播：实现BEV特征的对齐与融合
        
        通过学习空间变换参数实现特征对齐，然后将对齐后的特征
        与高级特征融合，形成统一的多模态BEV表示。
        
        Args:
            low_stage: 低级特征(通常是相机BEV特征) [B, C1, H, W]
            high_stage: 高级特征(通常是激光雷达BEV特征) [B, C2, H, W]
            
        Returns:
            concat: 融合后的特征 [B, C1+C2, H, W]
        """
        h, w = low_stage.size(2), low_stage.size(3)
        
        # 第一步：基于两个特征的拼接学习空间变换参数
        concat = torch.cat((low_stage, high_stage), 1)
        delta1 = self.delta_gen1(concat)  # 生成空间偏移参数
        
        # 第二步：应用空间变换对低级特征进行对齐
        # 对应论文公式(3)中的Align操作
        low_stage = self.bilinear_interpolate_torch_gridsample2(low_stage, (h, w), delta1)

        # 第三步：特征融合 - 将对齐后的特征与高级特征拼接
        # 对应论文公式(3)中的Fusion操作
        concat = torch.cat((low_stage, high_stage), 1)
        return concat


class SuperFusion(nn.Module):
    """
    SuperFusion主网络 - 实现论文的多级LiDAR-Camera融合架构
    
    该网络通过多级融合策略，结合深度感知的相机到BEV变换、
    图像引导的激光雷达BEV预测和BEV对齐与融合，实现长距离高清地图生成。
    
    网络架构对应论文图2的整体框架：
    1. 深度感知的相机到BEV变换 (3.1节)
    2. 图像引导的激光雷达BEV预测 (3.2节) 
    3. BEV对齐与融合 (3.3节)
    4. 高清地图解码器 (3.4节)
    
    主要创新点：
    - 多级融合架构实现长距离感知
    - 交叉注意力机制增强特征表示
    - 可学习的BEV对齐提升融合效果
    """
    def __init__(self, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=36, lidar=False, camC=64, lidarC=128, downsample=16, use_depth_enc=False, pretrained=True, add_depth_channel=False, ppdim=15):
        super(SuperFusion, self).__init__()
        # 特征通道配置
        self.camC = camC      # 相机特征通道数
        self.lidarC = lidarC  # 激光雷达特征通道数
        self.downsample = downsample  # 图像下采样倍数
        self.add_depth_channel = add_depth_channel  # 是否添加深度通道

        # Lift-Splat-Shoot模块：实现深度感知的相机到BEV变换 (论文3.1节)
        self.lss = LiftSplatShoot(data_conf, camC, downsample = downsample, use_depth_enc=use_depth_enc, pretrained=pretrained, add_depth_channel=add_depth_channel)

        self.lidar = lidar  # 是否使用激光雷达

        # BEV对齐与融合模块：实现多模态特征的空间对齐 (论文3.3节)
        self.fuser_AlignFA = AlignFAnew(self.camC+self.lidarC)

        # 图像引导的激光雷达BEV预测模块：通过交叉注意力增强激光雷达特征 (论文3.2节)
        self.lidar_pred = LidarPred(use_cross=True, num_heads=4, neck_dim=256, cross_dim=256)

        if lidar:
            # PointPillar编码器：将激光雷达点云转换为BEV特征
            self.pp = PointPillarEncoder(
                self.lidarC, data_conf['xbound'], data_conf['ybound'], data_conf['zbound'], ppdim=ppdim)
            # 融合后的BEV编码器：处理相机+激光雷达融合特征
            self.bevencode = BevEncode(inC=self.camC+self.lidarC, outC=data_conf['num_channels'], instance_seg=instance_seg,
                                       embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)
        else:
            # 仅相机的BEV编码器：处理纯相机BEV特征
            self.bevencode = BevEncode(inC=self.camC, outC=data_conf['num_channels'], instance_seg=instance_seg,
                                       embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)

    def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, depth_enc, projected_depth):
        """
        SuperFusion网络前向传播 - 实现多级LiDAR-Camera融合
        
        该方法实现了论文图2所示的完整多级融合流程：
        1. 深度感知的相机到BEV变换
        2. 图像引导的激光雷达BEV预测  
        3. BEV对齐与融合
        4. 高清地图解码
        
        Args:
            img: 多视角图像 [B, N, 3, H, W]
            trans: 相机平移向量 [B, N, 3]
            rots: 相机旋转矩阵 [B, N, 3, 3]
            intrins: 相机内参矩阵 [B, N, 3, 3]
            post_trans: 图像后处理平移 [B, N, 3]
            post_rots: 图像后处理旋转 [B, N, 3, 3]
            lidar_data: 激光雷达点云数据 [B, N_points, 4]
            lidar_mask: 激光雷达掩码 [B, N_points]
            car_trans: 车辆位置 [B, 3]
            yaw_pitch_roll: 车辆姿态 [B, 3]
            depth_enc: 用于深度编码的稀疏深度图
            projected_depth: 投影深度图
            
        Returns:
            x: 语义分割预测结果
            x_embedded: 实例嵌入特征
            x_direction: 方向预测结果
            depth: 深度分布预测
        """
        # 第一步：深度感知的相机到BEV变换 (论文3.1节)
        # 对应论文公式(1): 通过深度分布和几何变换将图像特征投影到BEV空间
        # torch.Size([4, 6, 64, 8, 22])
        topdown, depth, image_features_out = self.lss(img, rots, trans, intrins, post_rots, post_trans, depth_enc, projected_depth)
        # print(topdown.shape)

        if self.lidar:
            # 第二步：激光雷达点云编码
            # 使用PointPillar将激光雷达点云转换为BEV特征表示
            lidar_feature, neck_feature = self.pp(
                lidar_data, lidar_mask)  
            
            # 第三步：图像引导的激光雷达BEV预测 (论文3.2节)
            # 对应论文公式(2): 通过交叉注意力机制利用图像特征引导激光雷达特征预测
            lidar_feature = self.lidar_pred(lidar_feature, image_features_out)

            # 第四步：BEV对齐与融合 (论文3.3节)
            # 对应论文公式(3): 通过可学习的空间变换对齐并融合多模态BEV特征
            topdown = self.fuser_AlignFA(topdown, lidar_feature)
                                
        # 第五步：高清地图解码 (论文3.4节)
        # 通过BEV编码器生成最终的语义分割、实例嵌入和方向预测结果
        x, x_embedded, x_direction = self.bevencode(topdown)
        return x, x_embedded, x_direction, depth
