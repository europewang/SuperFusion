from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import kornia


class DDNTemplate(nn.Module):
    """
    深度分布网络模板 - 实现论文3.1节的深度感知相机编码
    
    该模块实现了深度感知的相机到BEV变换中的深度估计部分，
    通过预测每个像素的深度分布来增强3D几何理解能力。
    
    对应论文公式(1)中的深度分布预测部分：
    D(u,v) = softmax(f_depth(I(u,v)))
    其中 f_depth 是深度预测网络，D(u,v) 是像素(u,v)的深度分布
    
    主要功能：
    1. 基于预训练骨干网络提取图像特征
    2. 预测像素级深度分布
    3. 支持深度编码和深度通道增强
    """

    def __init__(self, constructor, feat_extract_layer, num_classes, use_depth_enc,add_depth_channel, pretrained_path=None, aux_loss=None):
        """
        初始化深度分布网络
        Args:
            constructor [function]: 模型构造函数
            feat_extract_layer [string]: 特征提取层名称
            num_classes [int]: 深度区间数量，对应论文中的D
            use_depth_enc [bool]: 是否使用深度编码
            add_depth_channel [bool]: 是否添加深度通道
            pretrained_path [string]: (可选) 预训练模型路径
            aux_loss [bool]: 是否包含辅助损失
        """
        super().__init__()
        self.num_classes = num_classes        # 深度区间数量，对应论文中的D
        self.pretrained_path = pretrained_path
        self.pretrained = pretrained_path is not None
        self.aux_loss = aux_loss
        self.use_depth_enc = use_depth_enc    # 是否使用深度编码
        self.add_depth_channel = add_depth_channel  # 是否添加深度通道

        if self.pretrained:
            # 预处理模块 - ImageNet标准化参数
            self.norm_mean = torch.Tensor([0.485, 0.456, 0.406])
            self.norm_std = torch.Tensor([0.229, 0.224, 0.225])

        # 获取骨干网络模型
        self.model = self.get_model(constructor=constructor)
        self.feat_extract_layer = feat_extract_layer
        # 配置特征提取层，用于获取中间特征
        self.model.backbone.return_layers = {
            feat_extract_layer: 'features',
            **self.model.backbone.return_layers
        }

    def get_model(self, constructor):
        """
        Get model
        Args:
            constructor [function]: Model constructor
        Returns:
            model [nn.Module]: Model
        """
        # Get model
        model = constructor(pretrained=False,
                            pretrained_backbone=False,
                            num_classes=self.num_classes,
                            aux_loss=self.aux_loss,
                            use_depth_enc = self.use_depth_enc)

        # Update weights
        if self.pretrained_path is not None:
            model_dict = model.state_dict()

            # Get pretrained state dict
            pretrained_dict = torch.load(self.pretrained_path)
            pretrained_dict = self.filter_pretrained_dict(model_dict=model_dict, pretrained_dict=pretrained_dict)

            # Update current model state dict
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            if self.add_depth_channel:
                print("initialize depth weights")
                layer = model.backbone.conv1
                new_in_channels = 4
                new_layer = nn.Conv2d(in_channels=new_in_channels, 
                                out_channels=layer.out_channels, 
                                kernel_size=layer.kernel_size, 
                                stride=layer.stride, 
                                padding=layer.padding,
                                bias=layer.bias)

                new_layer.weight.data.normal_(0, 0.001)
                new_layer.weight.data[:, :layer.in_channels, :, :] = layer.weight
                
                model.backbone.conv1 = new_layer
        else:
            if self.add_depth_channel:
                print("initialize depth weights 4")
                layer = model.backbone.conv1
                new_in_channels = 4
                new_layer = nn.Conv2d(in_channels=new_in_channels, 
                                out_channels=layer.out_channels, 
                                kernel_size=layer.kernel_size, 
                                stride=layer.stride, 
                                padding=layer.padding,
                                bias=layer.bias)

                # new_layer.weight.data.normal_(0, 0.001)
                # new_layer.weight.data[:, :layer.in_channels, :, :] = layer.weight
                
                model.backbone.conv1 = new_layer

        return model

    def filter_pretrained_dict(self, model_dict, pretrained_dict):
        """
        Removes layers from pretrained state dict that are not used or changed in model
        Args:
            model_dict [dict]: Default model state dictionary
            pretrained_dict [dict]: Pretrained model state dictionary
        Returns:
            pretrained_dict [dict]: Pretrained model state dictionary with removed weights
        """
        # Removes aux classifier weights if not used
        if "aux_classifier.0.weight" in pretrained_dict and "aux_classifier.0.weight" not in model_dict:
            pretrained_dict = {key: value for key, value in pretrained_dict.items()
                               if "aux_classifier" not in key}

        # Removes final conv layer from weights if number of classes are different
        if self.use_depth_enc:
            key_filter = []
            for key, value in pretrained_dict.items():
                if key.startswith('classifier'):
                    key_filter.append(key)
            #print(key_filter)
            for key in key_filter:
                pretrained_dict.pop(key)
        else:
            # key_filter = []
            # for key, value in pretrained_dict.items():
            #     key_filter.append(key)
            # print(key_filter)

            model_num_classes = model_dict["classifier.4.weight"].shape[0]
            pretrained_num_classes = pretrained_dict["classifier.4.weight"].shape[0]
            if model_num_classes != pretrained_num_classes:
                pretrained_dict.pop("classifier.4.weight")
                pretrained_dict.pop("classifier.4.bias")

        return pretrained_dict

    def forward(self, images, depth_enc=None, projected_depth=None):
        """
        前向传播：实现深度感知的图像特征提取和深度分布预测
        
        该方法实现了论文公式(1)中深度分布预测的完整流程：
        1. 图像预处理和特征提取
        2. 深度编码融合(可选)
        3. 深度分布预测
        
        Args:
            images [torch.Tensor(N, 3, H_in, W_in)]: 输入图像
            depth_enc [torch.Tensor]: 深度编码信息(可选)
            projected_depth [torch.Tensor]: 投影深度信息(可选)
            
        Returns:
            result [dict[torch.Tensor]]: 深度分布预测结果
                features [torch.Tensor(N, C, H_out, W_out)]: 提取的图像特征
                logits [torch.Tensor(N, num_classes, H_out, W_out)]: 深度分布logits - 对应论文公式(1)中的D(u,v)
                aux [torch.Tensor(N, num_classes, H_out, W_out)]: 辅助分类分数(可选)
        """
        # 第一步：图像预处理
        # 注释掉标准化预处理，直接使用原始图像
        # x = self.preprocess(images)
        x = images
        
        # 第二步：深度通道融合(可选)
        # 如果启用深度通道增强，将投影深度信息与RGB图像拼接
        if self.add_depth_channel:
            x = torch.cat((x, projected_depth), 1)  # 拼接深度通道，形成RGBD输入

        # 第三步：骨干网络特征提取
        # 使用预训练骨干网络提取多尺度图像特征
        result = OrderedDict()
        features = self.model.backbone(x)
        result['features'] = features['features']  # 保存中间特征用于后续处理
        feat_shape = features['features'].shape[-2:]  # 记录特征图尺寸用于上采样

        # 第四步：深度分布预测
        # 对应论文公式(1): D(u,v) = softmax(f_depth(I(u,v)))
        x = features["out"]  # 获取骨干网络输出特征
        result["image_features_out"] = x
        # print("features[out] shape: ", x.shape) # torch.Size([4, 2048, 32, 88])
        
        # 第五步：深度编码融合(可选)
        # 如果使用深度编码，将其与图像特征拼接以增强深度感知
        if self.use_depth_enc:
            #print("use_depth_enc")
            x = torch.cat((x, depth_enc), 1)  # 特征级深度编码融合
            
        # 第六步：分类器预测深度分布
        # 通过分类器网络预测每个像素的深度区间概率分布
        x = self.model.classifier(x)
        # print("self.model.classifier(x) shape: ", x.shape) # torch.Size([4, 34, 32, 88])
        
        # 第七步：特征图上采样
        # 将预测结果上采样到与特征图相同尺寸
        x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
        result["logits"] = x  # 最终深度分布logits

        # 第八步：辅助损失预测(可选)
        # 如果启用辅助损失，计算辅助分类器的预测结果
        if self.model.aux_classifier is not None:
            x = features["aux"]
            x = self.model.aux_classifier(x)
            x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
            result["aux"] = x  # 辅助预测结果，用于训练时的辅助监督

        return result

    def preprocess(self, images):
        """
        Preprocess images
        Args:
            images [torch.Tensor(N, 3, H, W)]: Input images
        Return
            x [torch.Tensor(N, 3, H, W)]: Preprocessed images
        """
        x = images
        if self.pretrained:
            # Create a mask for padded pixels
            mask = torch.isnan(x)

            # Match ResNet pretrained preprocessing
            x = kornia.normalize(x, mean=self.norm_mean, std=self.norm_std)

            # Make padded pixels = 0
            x[mask] = 0

        return x
