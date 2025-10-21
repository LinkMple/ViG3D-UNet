# 2024.05.10 test code for building pyramid_Vig3dUNet

# 2024.03.10 test code for building pyramid_Vig3d

# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from torchsummary import summary

# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.models.helpers import load_pretrained
# #from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model

from dynamic_network_architectures.vigModel.gcn_lib import Grapher3d, act_layer
from dynamic_network_architectures.vigModel.gcn_lib import DropPath

from dynamic_network_architectures.vigModel.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.vigModel.building_blocks.unet_decoder import UNetDecoder



class ChannelAttention3d(nn.Module):
    def __init__(self, channel, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        # Avg/Max Pooling
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        # Shared MLP
        y_avg = self.fc(y_avg).view(b, c, 1, 1, 1)
        y_max = self.fc(y_max).view(b, c, 1, 1, 1)
        # Fusion
        y = self.sigmoid(y_avg + y_max)
        return x * y.expand_as(x)


class SpatialAttention3d(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Avg/Max along channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate and convolve
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CBAM3d(nn.Module):
    def __init__(self, channel, reduction_ratio=8, spatial_kernel=7):
        super().__init__()
        self.channel_att = ChannelAttention3d(channel, reduction_ratio)
        self.spatial_att = SpatialAttention3d(spatial_kernel)

    def forward(self, x):
        x = self.channel_att(x)  # Channel attention first
        x = self.spatial_att(x)  # Then spatial attention
        return x

class SEBlock(nn.Module):
    def __init__(self, channel, r=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ , _= x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y

class FFN3d(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv3d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm3d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv3d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm3d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x #.reshape(B, C, N, 1)


class Stem3d(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, in_dim=3, out_dim=64, act='relu'):
        super().__init__()        
        self.convs = nn.Sequential(
            nn.Conv3d(in_dim, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm3d(out_dim//2),
            act_layer(act),
            nn.Conv3d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm3d(out_dim),
            act_layer(act),
            nn.Conv3d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm3d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Downsample3d(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, strides: Union[int, List[int], Tuple[int, ...]], in_dim=3, out_dim=768, ):
        super().__init__()        
        self.conv = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, 3, stride=strides, padding=1),
            nn.BatchNorm3d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DeepGCN3d(torch.nn.Module):
    def __init__(self,
        img_size = [80,192,160], 
        k = 27, # neighbor num (default:27)
        conv = 'mr', # graph conv layer {edge, mr}
        act = 'gelu', # activation layer {relu, prelu, leakyrelu, gelu, hswish}
        norm = 'batch', # batch or instance normalization {batch, instance}
        bias = True, # bias of conv layer True or False
        dropout = 0.0, # dropout rate
        use_dilation = True, # use dilated knn or not
        epsilon = 0.2, # stochastic epsilon for gcn
        stochastic = False, # stochastic for gcn, True or False
        drop_path = 0.0, # drop path rate
        blocks = [2,4,6,2], # number of basic blocks in the backbone
        channels = [64, 128, 256, 320], # number of channels of deep features
        reduce_ratios = [4, 2, 1, 1], # graph convolution reduction ratio
        # n_classes = 1000, # Dimension of out_channels
        # emb_dims = 1024 # Dimension of embeddings
        ):
        super(DeepGCN3d, self).__init__()

        self.n_blocks = sum(blocks) # 20
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        max_dilation = 49 // max(num_knn)  # 49//9 = 5

        self.stem = Stem3d(in_dim=1, out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], img_size[0]//4, img_size[1]//4, img_size[2]//4))
        img_size = [x // 4 for x in img_size] 

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if 0 < i < 3:
                self.backbone.append(Downsample3d(2, channels[i-1], channels[i]))
                img_size = [x // 2 for x in img_size]
            elif i == 3:
                self.backbone.append(Downsample3d((1,2,2), channels[i-1], channels[i]))
                img_size = [img_size[0], img_size[1] // 2, img_size[2] // 2]
            for j in range(blocks[i]):
                self.backbone += [
                    Seq(Grapher3d(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                                    bias, stochastic, epsilon, reduce_ratios[i], img_size, drop_path=dpr[idx],
                                    relative_pos=True),
                          FFN3d(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                         )]
                idx += 1
        self.backbone = Seq(*self.backbone)

        # self.prediction = Seq(nn.Conv3d(channels[-1], 1024, 1, bias=True),
        #                       nn.BatchNorm3d(1024),
        #                       act_layer(act),
        #                       nn.Dropout(dropout),
        #                       nn.Conv3d(1024, n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        vig_out = []
        vig_feat =[]
        B, C, H, W, D = x.shape
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            vig_out.append(x)
            if i in [1,6,13,16]:
                vig_feat.append(x)

        return vig_feat       # [1 6 13 16] is vig block feature out layer

class Pyramid_vig3dUnet(nn.Module):
    def __init__(self, n_channels=1, num_classes=2, img_size=[80,192,160],features_per_stage = (32, 64, 128, 256, 320, 320), patch_size=2):
        super(Pyramid_vig3dUnet, self).__init__()

        self.num_classes = num_classes
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.img_size = img_size
        self.features_per_stage = features_per_stage

        self.pvig = DeepGCN3d(
            img_size = self.img_size,
            k = 27,                         # neighbor num (default:27)
            conv = 'mr',                    # graph conv layer {edge, mr}
            act = 'gelu',                   # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            norm = 'batch',                 # batch or instance normalization {batch, instance}
            bias = True,                    # bias of conv layer True or False
            dropout = 0.0,                  # dropout rate
            use_dilation = True,            # use dilated knn or not
            epsilon = 0.2,                  # stochastic epsilon for gcn
            stochastic = False,             # stochastic for gcn, True or False
            drop_path = 0.0,                # drop path rate
            blocks = [2,4,6,2],             # number of basic blocks in the backbone
            channels = [64, 128, 256, 320], # number of channels of deep features
            reduce_ratios = [4, 2, 1, 1],   # graph convolution reduction ratio
        )

        self.cnn_encoder = PlainConvEncoder(
            input_channels = 1,
            n_stages = 6,
            features_per_stage = self.features_per_stage,  # (32, 64, 128, 256, 320, 320) 
            conv_op = nn.Conv3d, 
            kernel_sizes = 3, 
            strides =  [1, 2, 2, 2, 2, (1,2,2)],
            n_conv_per_stage = (2, 2, 2, 2, 2, 2),
            conv_bias = True,
            norm_op = nn.InstanceNorm3d,
            norm_op_kwargs = {"affine": True},
            dropout_op = None,
            dropout_op_kwargs = None, 
            nonlin = nn.LeakyReLU,
            nonlin_kwargs =  {"negative_slope":0.01, "inplace":True},
            return_skips=True,
            nonlin_first=False
        )

        self.chanels_skips = [self.features_per_stage[4] + self.features_per_stage[5],  # 320 + 320 = 640
                              self.features_per_stage[3] + self.features_per_stage[4],  # 256 + 320 = 576
                              self.features_per_stage[2] + self.features_per_stage[3],  # 128 + 256 = 384
                              self.features_per_stage[1] + self.features_per_stage[2],  # 64  + 128 = 192
                              self.features_per_stage[1],                               #             64 
                              self.features_per_stage[0]]                               #             32 

        # Convolutional Block Attention Module
        self.CBAM_0 = CBAM3d(self.chanels_skips[3])  # 原 self.SE_0
        self.CBAM_1 = CBAM3d(self.chanels_skips[2])  # 原 self.SE_1
        self.CBAM_2 = CBAM3d(self.chanels_skips[1])  # 原 self.SE_2
        self.CBAM_3 = CBAM3d(self.chanels_skips[0])  # 原 self.SE_3
        # self.SE_0 = SEBlock(self.chanels_skips[3])
        # self.SE_1 = SEBlock(self.chanels_skips[2])
        # self.SE_2 = SEBlock(self.chanels_skips[1])
        # self.SE_3 = SEBlock(self.chanels_skips[0])

        self.decoder = UNetDecoder(
            self.cnn_encoder, 
            num_classes = self.num_classes, 
            chanels_skips = self.chanels_skips,
            chanels_decoder = [640, 320, 256, 128, 64, 32],
            n_conv_per_stage = (2, 2, 2, 2, 2), # n_conv_per_stage_decoder
            deep_supervision = True,
            nonlin_first=False
        )

    def forward(self, inputs):
        # b, c, h, w, d = inputs.shape
        # patch_size = self.patch_size

        vig_feat = self.pvig(inputs)
        enc_feat = self.cnn_encoder(inputs)

        skips =[]
        skips.append(enc_feat[0])
        skips.append(enc_feat[1])
        
        feat_0 = torch.cat([vig_feat[0], enc_feat[2]], dim=1)   
        skip_2 = self.CBAM_0(feat_0)
        skips.append(skip_2)

        feat_1 = torch.cat([vig_feat[1], enc_feat[3]], dim=1)
        skip_3 = self.CBAM_1(feat_1)
        skips.append(self.CBAM_1(skip_3))

        feat_2 = torch.cat([vig_feat[2], enc_feat[4]], dim=1)
        skip_4 = self.CBAM_2(feat_2)
        skips.append(self.CBAM_2(skip_4)) 

        feat_3 = torch.cat([vig_feat[3], enc_feat[5]], dim=1)
        skip_5 = self.CBAM_3(feat_3)
        skips.append(self.CBAM_3(skip_5)) 
        
        # print(self.decoder)
        seg_out = self.decoder(skips)

        return seg_out


def input_test():
    #model = pvig_ti_224_gelu()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Pyramid_vig3dUnet().to(device)
    print(model)
    
    # 测试输入
    x = torch.randn(2, 1, 80, 192, 160).to(device)
    
    # 前向传播（返回多个分割输出）
    y = model(x)
    
    # 正确打印输出形状（处理多输出情况）
    if isinstance(y, (list, tuple)):
        print(f"模型输出包含 {len(y)} 个结果:")
        for i, out in enumerate(y):
            print(f"输出 {i+1} 形状: {out.shape}")  # 每个输出的形状
    else:
        print(f"输出形状: {y.shape}")

    #summary(model, input_size=[1, 80, 192, 160], batch_size=2)
    # summary(model, input_size=(1,80,192,160))
    pass


if __name__ == '__main__':
    input_test()
