import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss - 用于处理类别不平衡问题的损失函数
    
    该损失函数通过动态调整难易样本的权重，解决类别不平衡问题，
    特别适用于目标检测和语义分割任务中的前景背景不平衡。
    
    对应论文中提到的处理类别不平衡的策略，虽然论文主要使用加权BCE，
    但Focal Loss提供了更精细的难易样本平衡机制。
    
    公式: FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
    其中：
    - α_t: 类别权重，平衡正负样本
    - γ: 聚焦参数，降低易分类样本的权重
    - p_t: 预测概率
    """
    def __init__(self, alpha=1, gamma=2, reduce='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha    # 类别权重参数α，用于平衡正负样本
        self.gamma = gamma    # 聚焦参数γ，用于降低易分类样本的权重
        self.reduce = reduce  # 损失聚合方式：'mean'、'sum'或其他

    def forward(self, inputs, targets):
        """
        前向传播：计算Focal Loss
        
        通过动态调整难易样本权重，重点关注难分类样本，
        从而提升模型在类别不平衡数据上的性能。
        
        Args:
            inputs: 预测值logits [B, C, H, W]
            targets: 真值标签 [B, C, H, W]
            
        Returns:
            F_loss: Focal Loss值
        """
        # 计算基础的二元交叉熵损失
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        
        # 计算预测概率 p_t
        pt = torch.exp(-BCE_loss)
        
        # 计算Focal Loss: FL(p_t) = -α * (1-p_t)^γ * log(p_t)
        # (1-pt)^gamma项降低易分类样本的权重，alpha项平衡正负样本
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce == 'mean':
            return torch.mean(F_loss)
        elif self.reduce == 'sum':
            return torch.sum(F_loss)
        else:
            raise NotImplementedError


class SimpleLoss(torch.nn.Module):
    """
    简单二元交叉熵损失 - 实现论文4.1节的语义分割损失
    
    该损失函数用于语义分割任务，通过加权的二元交叉熵损失
    处理正负样本不平衡问题，提升对道路元素的检测精度。
    
    对应论文公式(4): L_seg = -α∑[y*log(σ(p)) + (1-y)*log(1-σ(p))]
    其中 α 是正样本权重，σ是sigmoid函数，p是预测logits
    """
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        # 使用带权重的二元交叉熵损失，pos_weight用于平衡正负样本
        # 对应论文公式(4)中的权重参数α
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        """
        前向传播：计算语义分割损失
        
        Args:
            ypred: 预测的logits [B, C, H, W]
            ytgt: 真值标签 [B, C, H, W]
            
        Returns:
            loss: 语义分割损失值
        """
        # 计算加权二元交叉熵损失
        # 对应论文公式(4)的具体实现
        loss = self.loss_fn(ypred, ytgt)
        return loss


class DiscriminativeLoss(nn.Module):
    """
    判别性损失函数 - 实现论文4.1节的实例嵌入损失
    
    该损失函数用于实例分割任务，通过三个组件确保同一实例的像素
    在嵌入空间中聚集，不同实例的像素在嵌入空间中分离。
    
    对应论文公式(5): L_embed = α*L_var + β*L_dist + γ*L_reg
    其中：
    - L_var: 类内方差损失，确保同一实例像素聚集
    - L_dist: 类间距离损失，确保不同实例分离  
    - L_reg: 正则化损失，防止嵌入向量过大
    """
    def __init__(self, embed_dim, delta_v, delta_d):
        super(DiscriminativeLoss, self).__init__()
        # 损失函数超参数，对应论文公式(5)中的各项参数
        self.embed_dim = embed_dim    # 嵌入向量维度
        self.delta_v = delta_v        # 类内方差阈值，控制同一实例的紧凑性
        self.delta_d = delta_d        # 类间距离阈值，控制不同实例的分离度

    def forward(self, embedding, seg_gt):
        """
        前向传播：计算判别性损失的三个组件
        
        实现论文公式(5)的完整计算过程，包括类内方差损失、
        类间距离损失和正则化损失。
        
        Args:
            embedding: 预测的嵌入特征 [B, embed_dim, H, W]
            seg_gt: 实例分割真值 [B, H, W]
            
        Returns:
            var_loss: 类内方差损失，确保同一实例像素聚集
            dist_loss: 类间距离损失，确保不同实例分离
            reg_loss: 正则化损失，防止嵌入向量过大
        """
        if embedding is None:
            return 0, 0, 0
        bs = embedding.shape[0]

        # 初始化三个损失组件
        var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

        # 逐批次计算损失
        for b in range(bs):
            embedding_b = embedding[b]  # (embed_dim, H, W)
            seg_gt_b = seg_gt[b]  # (H, W)

            # 获取当前批次中的所有实例标签(排除背景)
            labels = torch.unique(seg_gt_b)
            labels = labels[labels != 0]  # 移除背景标签
            num_lanes = len(labels)
            if num_lanes == 0:
                # 仅有背景，保持梯度连接但不增加损失
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                reg_loss = reg_loss + _nonsense * _zero
                continue

            centroid_mean = []  # 存储每个实例的嵌入中心
            
            # 计算每个实例的嵌入中心和类内方差损失
            for lane_idx in labels:
                seg_mask_i = (seg_gt_b == lane_idx)  # 当前实例的掩码
                if not seg_mask_i.any():
                    continue
                embedding_i = embedding_b[:, seg_mask_i]  # 当前实例的嵌入特征

                # 计算当前实例的嵌入中心(质心)
                mean_i = torch.mean(embedding_i, dim=1)
                centroid_mean.append(mean_i)

                # 计算类内方差损失 L_var
                # 对应论文公式(5)中的第一项：确保同一实例的像素在嵌入空间中聚集
                # L_var = (1/N) * Σ max(0, ||μ_c - x_i|| - δ_v)^2
                var_loss = var_loss + torch.mean(F.relu(torch.norm(embedding_i-mean_i.reshape(self.embed_dim, 1), dim=0) - self.delta_v) ** 2) / num_lanes
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

            # 计算类间距离损失(仅当有多个实例时)
            if num_lanes > 1:
                # 计算所有实例中心之间的距离矩阵
                centroid_mean1 = centroid_mean.reshape(-1, 1, self.embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, self.embed_dim)
                dist = torch.norm(centroid_mean1-centroid_mean2, dim=2)  # shape (num_lanes, num_lanes)
                
                # 将对角线元素设为delta_d以排除自身距离的影响
                dist = dist + torch.eye(num_lanes, dtype=dist.dtype, device=dist.device) * self.delta_d

                # 计算类间距离损失 L_dist
                # 对应论文公式(5)中的第二项：确保不同实例在嵌入空间中分离
                # L_dist = (1/N) * Σ max(0, δ_d - ||μ_ca - μ_cb||)^2
                # 除以2是因为距离矩阵的对称性导致重复计算
                dist_loss = dist_loss + torch.sum(F.relu(-dist + self.delta_d)**2) / (num_lanes * (num_lanes-1)) / 2

            # 正则化损失在原论文中未使用，此处注释掉
            # 计算正则化损失 L_reg
            # 对应论文公式(5)中的第三项：防止嵌入向量过大
            # reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        # 对批次进行平均
        var_loss = var_loss / bs
        dist_loss = dist_loss / bs
        reg_loss = reg_loss / bs
        return var_loss, dist_loss, reg_loss


def calc_loss():
    pass
