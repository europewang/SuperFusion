# SuperFusion：用于长距离高清地图生成的多级激光雷达-相机融合技术
郝东*、顾伟豪*、张先靖、徐金涛、艾锐、卢惠民、尤霍·坎纳拉、陈协源利  
（* 共同第一作者）  
郝东（瑞士苏黎世联邦理工学院）；顾伟豪、张先靖、徐金涛、艾锐（毫末智行）；卢惠民、陈协源利（国防科技大学）；尤霍·坎纳拉（阿尔托大学）  
通信作者：陈协源利（xieyuanli.chen@nudt.edu.cn）  
本研究由毫末智行公司资助  
arXiv：2211.15656v4 [cs.CV] 2024年11月27日  


## 摘要
环境的语义高清（HD）地图生成是自动驾驶的核心组成部分。现有方法通过融合激光雷达（LiDAR）与相机等不同传感器模态，在此任务中取得了较好性能。然而，当前研究多基于原始数据或网络特征级融合，且仅关注短距离（30米以内）高清地图生成，限制了其在实际自动驾驶场景中的部署。本文聚焦于短距离（30米以内）高清地图构建与长距离（最远90米）高清地图预测任务——后者是下游路径规划与控制任务的核心需求，可提升自动驾驶的平稳性与安全性。

为此，本文提出一种名为SuperFusion的新型网络，通过多级融合激光雷达与相机数据实现目标。该网络利用激光雷达深度信息优化图像深度估计，并借助图像特征引导长距离激光雷达特征预测。在nuScenes数据集与自主采集数据集上的实验表明，SuperFusion在所有距离区间均大幅优于当前主流基准方法。此外，将生成的高清地图应用于下游路径规划任务的结果显示，本文方法预测的长距离高清地图能为自动驾驶车辆提供更优的路径规划方案。相关代码已开源，地址为：https://github.com/haomo-ai/SuperFusion 。


## 1. 引言
检测车道并生成语义高清地图是自动驾驶车辆实现自主行驶的关键需求[6,7]。高清地图包含车道边界、道路分隔线、人行横道等语义图层，可为自动驾驶车辆提供周边基础设施、道路及环境的精确位置信息，保障安全导航[11] 。

传统高清地图构建采用离线方式：先采集点云数据，再通过同步定位与地图构建（SLAM）技术生成全局一致的地图，最后人工标注语义信息[28]。尽管部分自动驾驶企业通过该范式构建了高精度高清地图，但需投入大量人力成本，且地图更新难度大。由于自动驾驶车辆通常配备多种传感器，利用车载传感器数据构建局部高清地图以满足在线应用需求，已成为研究热点。现有方法多基于相机数据[30]或激光雷达数据[17]的鸟瞰视图（BEV）表示提取车道与人行横道；近年来，部分研究[17,20,24]通过融合多传感器模态取得了技术突破，利用不同传感器的互补信息提升高清地图生成性能 。

然而，现有方法对激光雷达与相机数据的融合方式较为简单，仅停留在原始数据级[27,26]、特征级[1,34]或最终BEV级[17,20,24]，未能充分发挥两种传感器的优势。此外，受传感器测量范围限制，现有方法仅关注短距离（30米以内）高清地图生成，难以满足实际自动驾驶场景中路径规划、运动控制等下游任务的需求。如图1所示，若生成的高清地图距离过短，规划算法可能因感知范围有限而生成非平稳路径（需频繁重规划），甚至出现与人行道交叉的路径。这种频繁变化的控制指令会降低用户乘坐舒适度，影响自动驾驶体验 。

为解决上述问题，本文提出一种多级激光雷达-相机融合方法（命名为SuperFusion），通过三个不同层级融合激光雷达与相机数据：
- **数据级融合**：将投影后的激光雷达数据与图像结合作为相机编码器输入，并利用激光雷达深度信息监督相机到BEV空间的变换；
- **特征级融合**：通过交叉注意力机制，以图像特征引导长距离激光雷达BEV特征预测；
- **最终BEV级融合**：设计BEV对齐模块对相机与激光雷达的BEV特征进行对齐与融合。

通过这种多级融合策略，SuperFusion既能生成高精度短距离高清地图，又能对原始激光雷达数据未覆盖的长距离区域进行精确语义预测 。

本文在公开的nuScenes数据集与自主采集的真实自动驾驶场景数据集上，对SuperFusion进行了全面评估，并与当前主流方法对比。实验结果表明，SuperFusion在所有距离区间均显著优于基准方法。此外，将生成的高清地图应用于路径规划任务的结果证实，本文提出的融合方法在长距离高清地图生成方面具有显著优势 。

本文的主要贡献如下：  
1. 提出新型多级激光雷达-相机融合网络，充分利用两种传感器的互补信息，生成高质量融合BEV特征，可支持多种下游任务；  
2. SuperFusion在短距离与长距离高清地图生成任务中均大幅超越现有主流融合方法；  
3. 首次实现最远90米的长距离高清地图生成，为自动驾驶下游规划任务提供关键技术支撑 。


## 2. 相关工作
### 2.1 激光雷达-相机融合
现有融合策略可分为三个层级：数据级、特征级与BEV级融合。  
- **数据级融合**：通过相机投影矩阵将激光雷达点云投影到图像上，生成稀疏深度图，并将其与图像数据[27,26]结合，或用图像语义特征对稀疏深度图进行增强[33,19]，以提升网络输入质量；  
- **特征级融合**：基于Transformer在特征空间融合不同模态数据[1,34]，先生成激光雷达特征图，再通过交叉注意力机制查询图像特征，最后拼接特征用于下游任务；  
- **BEV级融合**：分别提取激光雷达与相机的BEV特征，再通过拼接[17]或融合模块[20,24]进行融合。例如，HDMapNet[17]在相机分支中利用MLP将透视视图（PV）特征映射为BEV特征，在激光雷达分支中利用PointPillars[16]编码BEV特征；近期BEVFusion系列研究[20,24]在相机分支中采用LSS[30]进行视角变换，在激光雷达分支中采用VoxelNet[35]，最终通过BEV对齐模块融合特征。  

与现有方法不同，本文方法融合了上述三个层级的激光雷达-相机融合策略，充分发挥两种传感器的互补特性 。

### 2.2 高清地图生成
传统语义高清地图重建需通过SLAM算法[28]聚合激光雷达点云，再进行人工标注，不仅耗时耗力，且更新困难。HDMapNet[17]是首个无需人工标注的局部高清地图构建方法，通过在BEV空间融合激光雷达与六路环视相机数据，实现语义高清地图生成。此外，VectorMapNet[23]将地图元素表示为折线集合，通过集合预测框架对其建模；Image2Map[32]利用Transformer从图像中端到端生成高清地图。部分研究[10,12,3]还针对车道等特定地图元素开展检测工作。  

现有方法仅能在短距离（通常小于30米）内进行地图分割，而本文首次聚焦于最远90米的长距离高清地图生成任务 。


## 3. 方法原理
### 3.1 深度感知的相机到BEV变换
首先在原始数据级融合激光雷达与相机数据，并利用激光雷达的深度信息辅助相机特征提升至BEV空间。为此，本文提出**深度感知的相机到BEV变换模块**（如图2所示），输入为RGB图像\(I\)与对应的稀疏深度图\(D_{sparse}\)（通过相机投影矩阵将3D激光雷达点云\(P\)投影到图像平面生成） 。

相机骨干网络包含两个分支：  
1. 第一个分支提取2D图像特征\(F \in \mathbb{R}^{W_{F} \times H_{F} \times C_{F}}\)（其中\(W_{F}\)、\(H_{F}\)、\(C_{F}\)分别为特征图的宽度、高度与通道数）；  
2. 第二个分支连接深度预测网络，为2D特征\(F\)中的每个元素估计类别深度分布\(D \in \mathbb{R}^{W_{F} \times H_{F} \times D}\)（其中\(D\)为离散深度区间数量）。  

为提升深度估计精度，采用补全方法[15]对\(D_{sparse}\)进行处理，生成密集深度图\(D_{dense}\)，并将每个像素的深度值离散化为深度区间，最终转换为独热编码向量，用于监督深度预测网络。  

通过\(D\)与\(F\)的外积生成最终视锥体特征网格\(M\)，公式如下：  
\[M(u,v)=D(u,v)\otimes F(u,v) \tag{1}\]  
其中\(M \in \mathbb{R}^{W_{F} \times H_{F} \times D \times C_{F}}\)。  

最后，将视锥体中的每个体素分配给最近的柱体（pillar），并采用LSS[30]中的求和池化操作生成相机BEV特征\(C \in \mathbb{R}^{W \times H \times C_{F}}\) 。

本文提出的深度感知相机到BEV模块与现有深度预测方法[30,31]的区别在于：  
- LSS[30]的深度预测仅通过语义分割损失进行隐式监督，不足以生成精确深度估计；  
- CaDDN[31]虽利用激光雷达深度进行监督，但未将激光雷达数据作为输入，难以生成鲁棒可靠的深度估计；  
- 本文方法同时利用补全后的密集激光雷达深度图进行监督，并将稀疏深度图作为RGB图像的额外通道，结合深度先验与精确深度监督，使网络在复杂环境中具有更强的泛化能力 。


### 3.2 图像引导的激光雷达BEV预测
在激光雷达分支中，采用PointPillars[16]结合动态体素化[36]作为点云编码器，为每个点云\(P\)生成激光雷达BEV特征\(L \in \mathbb{R}^{W \times H \times C_{L}}\)。如图3（a）所示，激光雷达对地面的有效测量范围通常较短（32线旋转激光雷达的有效范围约30米），导致部分激光雷达BEV特征对应区域为空（无有效数据）。相比之下，相机对地面的可视距离更远，因此本文提出**BEV预测模块**，以图像特征为引导，对激光雷达分支中未覆盖的地面区域进行预测（如图3（b）所示） 。

BEV预测模块为编码器-解码器网络：  
- **编码器**：通过多个卷积层将原始BEV特征\(L\)压缩为瓶颈特征\(B \in \mathbb{R}^{W/8 \times H/8 \times C_{B}}\)；  
- **交叉注意力机制**：动态捕捉瓶颈特征\(B\)与透视视图（FV）图像特征\(F\)的关联。通过三个全连接层将瓶颈特征\(B\)转换为查询（Query，\(Q\)），将FV图像特征\(F\)转换为键（Key，\(K\)）与值（Value，\(V\)）；通过\(Q\)与\(K\)的内积计算注意力亲和矩阵（反映激光雷达BEV中每个体素与对应相机特征的关联度），经softmax归一化后，对\(V\)进行加权聚合，得到聚合特征\(A\)。  

交叉注意力机制的公式如下：  
\[A=Attention(Q, K, V)=softmax\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V \tag{2}\]  
其中\(d_{k}\)为用于缩放的通道维度。  

对聚合特征\(A\)应用卷积层降低通道数，与原始瓶颈特征\(B\)拼接后，再通过一个卷积层得到最终瓶颈特征\(B'\)。\(B'\)融合了图像特征的视觉引导信息，输入解码器后生成完整的预测激光雷达BEV特征\(L'\)。通过这种特征级融合，可更精准地预测长距离激光雷达BEV特征 。


### 3.3 BEV对齐与融合
目前，相机与激光雷达分支分别生成了各自的BEV特征，但受深度估计误差与外参不精确影响，两种特征通常存在错位，直接拼接会导致性能下降。为此，本文在BEV级融合两种特征，并设计**对齐与融合模块**（如图4所示）：输入为相机与激光雷达的BEV特征，输出为对齐后的融合BEV特征 。

该模块首先学习相机BEV特征的流场\(\Delta \in \mathbb{R}^{W \times H \times 2}\)，再通过流场将原始相机BEV特征\(C\)扭曲（warp）为与激光雷达特征\(L'\)对齐的BEV特征\(C'\)，公式如下：  
\[
\begin{aligned} 
C'_{w h}= & \sum_{w'=1}^{W} \sum_{h'=1}^{H} C_{w' h'} \cdot \max \left(0,1-\left|w+\Delta_{1 w h}-w'\right|\right) \\ 
& \cdot \max \left(0,1-\left|h+\Delta_{2 w h}-h'\right|\right) 
\end{aligned} \tag{3}
\]  

参考[14,18]的方法，采用双线性插值核在\(C\)的\((w+\Delta_{1 w h}, h+\Delta_{2 w h})\)位置采样特征，其中\(\Delta_{1 w h}\)、\(\Delta_{2 w h}\)为位置\((w, h)\)处学习到的二维流场 。

最后，将对齐后的相机BEV特征\(C'\)与预测激光雷达BEV特征\(L'\)拼接，生成融合BEV特征，作为高清地图解码器的输入 。


### 3.4 高清地图解码器与训练损失
参考HDMapNet[17]，将高清地图解码器设计为全卷积网络[25]：输入为融合BEV特征，输出语义分割、实例嵌入与车道方向三个预测结果，经后处理后矢量化为地图元素 。

针对三个输出头，采用不同训练损失：  
1. **语义分割损失**：采用交叉熵损失\(L_{seg}\)监督语义分割结果；  
2. **实例嵌入损失**：定义方差损失与距离损失[5]，公式如下：  
   \[L_{var}=\frac{1}{C} \sum_{c=1}^{C} \frac{1}{N_{c}} \sum_{j=1}^{N_{c}}\left[\left\| \mu_{c}-f_{j}^{instance }\right\| -\delta_{v}\right]_{+}^{2} \tag{4}\]  
   \[L_{dist }=\frac{1}{C(C-1)} \sum_{c_{A} \neq c_{B} \in C}\left[2 \delta_{d}-\left\| \mu_{c_{A}}-\mu_{c_{B}}\right\| \right]_{+}^{2} \tag{5}\]  
   \[L_{ins }=\alpha L_{var }+\beta L_{dist} \tag{6}\]  
   其中，\(C\)为聚类数量，\(N_{c}\)与\(\mu_{c}\)分别为聚类\(c\)中的元素数量与平均嵌入向量，\(f_{j}^{instance}\)为聚类\(c\)中第\(j\)个元素的嵌入向量，\(\|\cdot\|\)为\(L_2\)范数，\([x]_{+}=\max(0,x)\)，\(\delta_v\)与\(\delta_d\)分别为方差损失与距离损失的边际值；  
3. **方向预测损失**：将方向在圆周上均匀离散为36类，采用交叉熵损失\(L_{dir}\)监督方向预测结果，仅对车道上具有有效方向的像素进行反向传播。  

在推理阶段，采用DBSCAN[8]对实例嵌入进行聚类，通过非极大值抑制[17]减少冗余，再利用预测方向贪婪连接像素，得到高清地图元素的最终矢量表示 。

深度预测采用聚焦损失（focal loss）[21]（设置\(\gamma=2.0\)），记为\(L_{dep}\)。总损失为深度估计、语义分割、实例嵌入与车道方向预测损失的加权和，公式如下：  
\[L=\lambda _{dep}L_{dep}+\lambda _{seg}L_{seg}+\lambda _{ins}L_{ins}+\lambda _{dir}L_{dir} \tag{7}\]  
其中\(\lambda_{dep}\)、\(\lambda_{seg}\)、\(\lambda_{ins}\)、\(\lambda_{dir}\)为权重系数。


## 4. 实验
### 4.1 实现细节
#### 4.1.1 模型设置
- 相机分支骨干网络采用ResNet-101[13]，激光雷达分支骨干网络采用PointPillars[16]；  
- 深度估计模块基于DeepLabV3[4]修改，生成像素级深度区间概率分布；  
- 相机骨干网络初始化采用在MS-COCO数据集[22]上预训练的DeepLabV3语义分割模型，其他组件随机初始化；  
- 图像尺寸设置为256×704，激光雷达点云体素化分辨率为0.15米；  
- BEV高清地图范围设置为\([0,90]m \times [-15,15]m\)，对应BEV特征图尺寸为600×200；  
- 离散深度区间设置为2.0–90.0米，间隔1.0米。

#### 4.1.2 训练细节
- 采用随机梯度下降（SGD）优化器训练30个epoch，初始学习率为0.1，训练过程中采用多项式学习率衰减策略；  
- 实例嵌入损失中，设置\(\alpha=\beta=1\)、\(\delta_d=3.0\)、\(\delta_v=0.5\)；  
- 总损失权重系数设置为：\(\lambda_{dep}=1.0\)、\(\lambda_{seg}=1.0\)、\(\lambda_{ins}=1.0\)、\(\lambda_{dir}=0.2\)。


### 4.2 评估指标
#### 4.2.1 交并比（IoU）
用于评估语义分割精度，计算预测高清地图\(M_1\)与地面真值地图\(M_2\)的交集与并集之比，公式如下：  
\[IoU\left( M_{1}, M_{2}\right) =\frac {\left| M_{1} \cap M_{2}\right| }{\left| M_{1} \cup M_{2}\right| } \tag{8}\]  
IoU值越高，表明分割结果与真值的重合度越高。

#### 4.2.2 单向 Chamfer 距离（CD）
用于评估预测曲线与地面真值曲线的空间距离，公式如下：  
\[CD=\frac{1}{C_{1}} \sum_{x \in C_{1}} \min _{y \in C_{2}}\| x-y\| _{2} \tag{9}\]  
其中\(C_1\)、\(C_2\)分别为预测曲线与真值曲线上的点集。CD值越低，表明两条曲线的空间偏差越小。  

单独使用CD存在局限性：IoU较低时，CD也可能较小。因此，本文结合IoU与CD筛选真阳性样本，确保评估的全面性。

#### 4.2.3 平均精度（AP）
用于评估实例检测能力，定义为不同召回率（recall）下精度（precision）的平均值，公式如下：  
\[AP=\frac{1}{10} \sum_{r \in\{0.1,0.2, ..., 1.0\}} AP_{r} \tag{10}\]  
其中\(AP_r\)为召回率\(r\)对应的精度。参考[17]的设定，结合CD与IoU阈值判断真阳性：仅当预测实例的CD小于1.0米且IoU大于0.1时，才视为真阳性。

#### 4.2.4 多距离区间评估
为验证长距离预测能力，将地面真值按距离分为三个区间：0–30米（短距离）、30–60米（中距离）、60–90米（长距离），分别计算各区间的IoU与AP，全面评估不同距离下的地图生成性能。


### 4.3 评估结果
#### 4.3.1 nuScenes数据集结果
在公开的nuScenes数据集[2]上，聚焦语义高清地图分割与实例检测任务，选取车道边界、道路分隔线、人行横道三类静态地图元素进行评估。  

**语义分割IoU结果**（表1）：SuperFusion在所有类别与所有距离区间均取得最佳性能，且优势显著。例如，在60–90米长距离区间，道路分隔线、人行横道、车道边界的IoU分别达到29.2%、12.2%、28.1%，较现有最优方法BEVFusion[24]分别提升6.8%、7.2%、6.4%。这表明多级融合策略能有效利用相机的长距离视觉信息，弥补激光雷达长距离数据缺失的问题。  

**实例检测AP结果**（表2）：SuperFusion在所有类别与距离区间的AP均大幅领先基准方法。以60–90米区间为例，车道边界AP达到38.2%，较BEVFusion[24]提升11.3%；人行横道AP达到10.7%，较BEVFusion[24]提升5.9%。这验证了本文方法在长距离实例级地图元素检测上的有效性。  

此外，实验发现：纯激光雷达方法（如PointPillars[16]）在长距离（60–90米）性能急剧下降（车道边界IoU仅6.2%），而纯相机方法（如LSS[30]）受深度估计误差影响，长距离精度也有限；激光雷达-相机融合方法（如HDMapNet[17]、BEVFusion[20,24]）虽优于单模态方法，但因融合层级单一，长距离性能仍不及SuperFusion。

#### 4.3.2 自主采集数据集结果
为验证模型的泛化能力，在真实自动驾驶场景中采集了自主数据集，传感器配置与nuScenes一致，包含21000帧数据（18000帧训练、3000帧测试），手动标注车道边界、道路分隔线两类地图元素。  

实验结果（表3）显示，SuperFusion的性能依旧领先：道路分隔线平均IoU达到53.0%（较BEVFusion[24]提升4.0%），车道边界平均IoU达到24.7%（较BEVFusion[24]提升5.9%）；实例检测AP方面，道路分隔线与车道边界分别达到42.4%、35.0%，均优于所有基准方法。这表明SuperFusion在真实复杂场景中仍具有稳定的高性能，泛化能力较强。

#### 4.3.3 定性结果
图5展示了不同方法的高清地图预测定性对比：基准方法（如HDMapNet[17]、BEVFusion[24]）仅能准确预测30米以内的地图元素，长距离区域存在明显缺失或错误；而SuperFusion在0–90米范围内均能生成完整、准确的地图，且各类元素（车道边界、分隔线、人行横道）的实例区分清晰，与地面真值高度吻合。


### 4.4 消融实验与模块分析
#### 4.4.1 各模块有效性验证
为验证SuperFusion中关键模块的作用，设计了消融实验（表4），结果如下：  
- **无深度监督（w/o Depth Supervision）**：移除激光雷达深度对相机深度估计的监督，语义分割平均IoU大幅下降（道路分隔线从38.0%降至25.4%）。原因是深度估计不准确导致相机到BEV的变换偏差，后续对齐模块失效。  
- **无深度先验（w/o Depth Prior）**：不将激光雷达稀疏深度图作为相机输入的额外通道，在复杂环境（如光照变化、遮挡）下深度估计可靠性下降，平均IoU降低约8%。  
- **无激光雷达预测（w/o LiDAR Prediction）**：移除图像引导的激光雷达BEV预测模块，长距离区域仅依赖相机信息，平均IoU下降约10%，表明该模块对长距离特征补全至关重要。  
- **无交叉注意力（w/o Cross-Attention）**：保留激光雷达BEV编码器-解码器结构，但移除与相机特征的交叉注意力交互，网络无法利用图像视觉引导，平均IoU下降约12%，验证了交叉注意力在特征级融合中的核心作用。  
- **无BEV对齐（w/o BEV Alignment）**：直接拼接相机与激光雷达BEV特征，因特征错位，平均IoU下降约9%，说明对齐模块能有效解决传感器外参误差与深度偏差带来的融合问题。  

综上，SuperFusion的多级融合模块（深度监督、深度先验、激光雷达预测、交叉注意力、BEV对齐）均不可或缺，共同支撑了高性能的长距离高清地图生成。

#### 4.4.2 模块替代方案对比
为验证本文模块设计的优越性，对比了不同替代方案（表5）：  
- **对齐模块对比**：相较于现有动态对齐（DynamicAlign[20]）与卷积对齐（ConvAlign[24]），本文提出的BEVAlign模块在所有类别IoU上均领先（道路分隔线IoU达38.0%，较ConvAlign提升4.5%）。原因是DynamicAlign更适用于3D目标检测，ConvAlign对深度误差鲁棒性不足，而BEVAlign通过流场学习实现更精细的特征对齐。  
- **深度先验添加方式对比**：测试了“深度编码器（bin）”“深度编码器（原始值）”“深度通道（bin）”“深度通道（原始值）”四种方式，结果显示“将稀疏深度图作为额外输入通道（原始值）”的效果最佳（道路分隔线IoU达38.0%）。这是因为原始深度值能保留更精细的距离信息，而额外通道的方式能更直接地将深度先验融入图像特征提取过程。


### 4.5 路径规划应用验证
为验证长距离高清地图对下游任务的价值，采用动态窗口法（DWA[9]），基于HDMapNet[17]、BEVFusion[24]与SuperFusion生成的高清地图进行路径规划实验。实验设置：随机选取100个场景，每个场景在30–90米范围内指定一个可行驶目标点；若规划路径与人行道交叉或DWA无法生成有效路径，则视为规划失败。  

结果（表6）显示：SuperFusion的路径规划成功率达72%，较HDMapNet（45%）提升27%，较BEVFusion（49%）提升23%。如图6所示，SuperFusion生成的长距离地图能为规划算法提供完整的道路边界与车道信息，避免因地图缺失导致的路径偏移或交叉；而基准方法因长距离地图不完整，易出现规划路径不平稳或碰撞风险。这证实了长距离高清地图对提升自动驾驶路径规划安全性与平稳性的关键作用。


## 5. 结论
本文提出一种新型激光雷达-相机融合网络SuperFusion，用于解决长距离高清地图生成问题。该网络通过数据级、特征级、BEV级的多级融合策略，充分利用两种传感器的互补优势：借助激光雷达深度优化相机深度估计，利用图像特征引导长距离激光雷达特征预测，通过BEV对齐模块实现特征精准融合，最终生成最远90米的高精度高清地图。  

在nuScenes数据集与自主采集数据集上的实验表明，SuperFusion在所有距离区间的语义分割IoU与实例检测AP均大幅超越现有主流方法。路径规划应用实验进一步证实，本文方法生成的长距离高清地图能显著提升自动驾驶下游规划任务的成功率与安全性。  

未来研究方向可聚焦于：1）结合时序数据进一步提升长距离地图的稳定性；2）扩展地图元素类别，支持更复杂场景的语义建模；3）优化模型推理速度，满足车载实时部署需求。


## 参考文献
[1] X. Bai, Z. Hu, X. Zhu, Q. Huang, Y. Chen, H. Fu, and C.L. Tai. Transfusion: Robust lidar-camera fusion for 3d object detection with transformers. In Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2022.  
[2] H. Caesar, V. Bankiti, A.H. Lang, S. Vora, V.E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2020.  
[3] L. Chen, C. Sima, Y. Li, Z. Zheng, J. Xu, X. Geng, H. Li, C. He, J. Shi, Y. Qiao, and J. Yan. Persformer: 3d lane detection via perspective transformer and the openlane benchmark. In Proc. of the Europ. Conf. on Computer Vision (ECCV), 2022.  
[4] L. Chen, G. Papandreou, F. Schroff, and H. Adam. Rethinking atrous convolution for semantic image segmentation. In Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2017.  
[5] B. De Brabandere, D. Neven, and L. Van Gool. Semantic instance segmentation for autonomous driving. In Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition Workshops (CVPRW), 2017.  
[6] H. Dong, X. Chen, and C. Stachniss. Online Range Image-based Pole Extractor for Long-term LiDAR Localization in Urban Environments. In Proceedings of the European Conference on Mobile Robots (ECMR), 2021.  
[7] H. Dong, X. Chen, S. S¨arkk¨a, and C. Stachniss. Online pole segmentation on range images for long-term lidar localization in urban environments. Robotics and Autonomous Systems, 159:104283, 2023.  
[8] M. Ester, H.P. Kriegel, J. Sander, and X. Xu. A density-based algorithm for discovering clusters in large spatial databases with noise. In Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, page 226–231, 1996.  
[9] D. Fox, W. Burgard, and S. Thrun. The dynamic window approach to collision avoidance. IEEE Robotics and Automation Magazine, 4(1):23–33, 1997.  
[10] N. Garnett, R. Cohen, T. Pe’er, R. Lahav, and D. Levi. 3d-lanenet: end-to-end 3d multiple lane detection. In Proc. of the IEEE Intl. Conf. on Computer Vision (ICCV), 2019.  
[11] F. Ghallabi, F. Nashashibi, G. El-Haj-Shhade, and M.A. Mittet. Lidar-based lane marking detection for vehicle positioning in an hd map. In International Conference on Intelligent Transportation Systems (ITSC), 2018.  
[12] Y. Guo, G. Chen, P. Zhao, W. Zhang, J. Miao, J. Wang, and T.E. Choe. Gen-lanenet: A generalized and scalable approach for 3d lane detection. In Proc. of the Europ. Conf. on Computer Vision (ECCV), 2020.  
[13] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2016.  
[14] Z. Huang, Y. Wei, X. Wang, H. Shi, W. Liu, and T.S. Huang. Alignseg: Feature-aligned segmentation networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(1):550–557, 2021.  
[15] J. Ku, A. Harakeh, and S.L. Waslander. In defense of classical image processing: Fast depth completion on the cpu. In Conference on Computer and Robot Vision (CRV), 2018.  
[16] A.H. Lang, S. Vora, H. Caesar, L. Zhou, J. Yang, and O. Beijbom. Pointpillars: Fast encoders for object detection from point clouds. In Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2019.  
[17] Q. Li, Y. Wang, Y. Wang, and H. Zhao. Hdmapnet: An online hd map construction and evaluation framework. In Proc. of the IEEE Intl. Conf. on Robotics & Automation (ICRA), 2022.  
[18] X. Li, A. You, Z. Zhu, H. Zhao, M. Yang, K. Yang, and Y. Tong. Semantic flow for fast and accurate scene parsing. In Proc. of the Europ. Conf. on Computer Vision (ECCV), 2020.  
[19] Y. Li, A.W. Yu, T. Meng, B. Caine, J. Ngiam, D. Peng, J. Shen, Y. Lu, D. Zhou, Q.V. Le, et al. Deepfusion: Lidar-camera deep fusion for multi-modal 3d object detection. In Proc. of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2022.  
[20] T. Liang, H. Xie, K. Yu, Z. Xia, Y.W. Zhiwei Lin, T. Tang, B. Wang, and Z. Tang. BEVFusion: A Simple and Robust LiDAR-Camera Fusion Framework. In Proc. of the Advances in Neural Information Processing Systems (NeurIPS), 2022.  
