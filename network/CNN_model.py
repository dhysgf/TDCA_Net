import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm  # 假设使用带约束的权重归一化
# 定义带约束的线性层（保持原逻辑）
class LinearWithConstraint(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, max_norm=1.0):
        super(LinearWithConstraint, self).__init__(in_features, out_features, bias)
        self.max_norm = max_norm
    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        # if self.bias:
        #     self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

class TENet(nn.Module):
    def __init__(self, num_channels, num_classes,
                 dropout_level=0.25, filters=32):
        """
        移除input_length参数，自动计算全连接层输入维度

        Args:
            num_channels: 通道数 (Nc)
            num_classes: 分类数 (Nf)
            dropout_level: Dropout比例
            filters: 卷积核数量
            Nm: 滤波器组数
        """
        super(TENet, self).__init__()

        # 核心参数
        self.dropout_level = dropout_level
        self.K1 = 7  # 第一个时间卷积核大小
        self.K2 = 3  # 第二个时间卷积核大小
        self.S = 3  # 时间卷积步长
        self.Nm =num_classes
        self.filters = filters  # 卷积核数量
        self.num_channels = num_channels  # 输入通道数
        self.num_classes = num_classes  # 输出分类数

        # 通道特征提取层（压缩空间维度）
        self.chan_feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=self.Nm, out_channels=self.filters, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_features=self.filters, momentum=0.99, eps=0.001),
            nn.ELU(alpha=1),
            # 分组卷积压缩通道维度（输出空间维度变为1）
            nn.Conv2d(
                in_channels=self.filters,
                out_channels=self.filters,
                kernel_size=(self.num_channels, 1),  # 高度=输入通道数，压缩空间维度
                groups=self.filters  # 分组卷积，每个卷积核处理1个输入通道
            ),
            nn.BatchNorm2d(num_features=self.filters, momentum=0.99, eps=0.001),
            nn.ELU(alpha=1),
            nn.Dropout(self.dropout_level)
        )
        # 时间特征提取层 - 第一层
        self.time1_feature_extraction = nn.Sequential(
            nn.Conv2d(
                in_channels=self.filters,
                out_channels=self.filters,
                kernel_size=(1, self.K1),  # 时间维度卷积
                stride=(1, self.S)
            ),
            nn.BatchNorm2d(num_features=self.filters, momentum=0.99, eps=0.001),
            nn.ELU(alpha=1),
            nn.Dropout(self.dropout_level)
        )

        # 时间特征提取层 - 第二层
        self.time2_feature_extraction = nn.Sequential(
            nn.Conv2d(
                in_channels=self.filters,
                out_channels=self.filters,
                kernel_size=(1, self.K2),  # 时间维度卷积
                stride=(1, self.S)
            ),
            nn.BatchNorm2d(num_features=self.filters, momentum=0.99, eps=0.001),
            nn.ELU(alpha=1),
            nn.Dropout(self.dropout_level)
        )

        # 全连接层（延迟初始化，在首次前向时根据输入维度创建）
        self.dense_layers = None

    def forward(self, x):
        """
        输入: [batch, Nm, Nc, Nt] （Nm:滤波器组数, Nc:通道数, Nt:时间长度）
        输出: [batch, num_classes]
        """
        # 1. 通道特征提取（压缩空间维度）
        # 输入x: [batch, Nm, Nc, Nt]
        out = self.chan_feature_extraction(x)  # 输出: [batch, filters, 1, Nt]（空间维度压缩为1）

        # 2. 时间特征提取第一层
        out = self.time1_feature_extraction(out)  # 输出: [batch, filters, 1, t1]（t1为第一次时间卷积后长度）

        # 3. 时间特征提取第二层
        out = self.time2_feature_extraction(out)  # 输出: [batch, filters, 1, t2]（t2为第二次时间卷积后长度）

        # 4. 动态计算全连接层输入维度并初始化
        # 获取时间维度长度t2（out形状: [batch, filters, 1, t2]）
        t2 = out.shape[-1]
        # 计算flatten后的维度（filters * t2，因为空间维度为1）
        fc_input_dim = self.filters * t2
        # 首次前向时初始化全连接层
        if self.dense_layers is None:
            self.dense_layers = nn.Sequential(
                nn.Flatten(),  # 展平为 [batch, filters * t2]
                LinearWithConstraint(fc_input_dim, self.num_classes, max_norm=0.5)
            ).to(out.device)  # 确保与特征在同一设备

        # 5. 分类层输出
        logits = self.dense_layers(out)  # 输出: [batch, num_classes]
        return logits