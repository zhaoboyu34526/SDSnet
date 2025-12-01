"""
# Code Adapted from:
# https://github.com/sthalles/deeplab_v3
#
# MIT License
#
# Copyright (c) 2018 Thalles Santos Silva
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
"""
import logging
from numpy import var
import torch
from torch import nn
import random
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.distributions as tdist

__all__ = ['ResNet', 'resnet50', 'resnet101']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    """
    Bottleneck Layer for Resnet
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = Norm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = Norm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = Norm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """
    Resnet Global Module for Initialization
    """

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        # self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        
    
        self.bn1 = Norm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                #  or isinstance(m, nn.InstanceNorm2d)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                Norm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for index in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_tuple = self.layer1([x, w_arr])  # 400
        low_level = x_tuple[0]

        x_tuple = self.layer2(x_tuple)  # 100
        x_tuple = self.layer3(x_tuple)  # 100
        aux_out = x_tuple[0]
        x_tuple = self.layer4(x_tuple)  # 100

        x = x_tuple[0]
        w_arr = x_tuple[1]

        return x


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        print("########### pretrained ##############")
        forgiving_state_restore(model, model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        print("########### pretrained ##############")
        forgiving_state_restore(model, model_zoo.load_url(model_urls['resnet101']))
    return model

def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """
    layer = torch.nn.SyncBatchNorm
    normalization_layer = layer(in_channels)
    return normalization_layer


def freeze_weights(*models):
    for model in models:
        for k in model.parameters():
            k.requires_grad = False

def unfreeze_weights(*models):
    for model in models:
        for k in model.parameters():
            k.requires_grad = True

def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or \
                isinstance(module, nn.GroupNorm) or isinstance(module, nn.SyncBatchNorm):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

def initialize_embedding(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                module.weight.data.zero_() #original



def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
        else:
            pass
            # print("Skipped loading parameter", k)
            # logging.info("Skipped loading parameter %s", k)
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net

def forgiving_state_restore_ibn(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    net_state_dict = net.state_dict() ## model
    new_loaded_dict = {}
    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
        else:
            pass
            # print("Skipped loading parameter", k)
            # logging.info("Skipped loading parameter %s", k)
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net

class StyleHallucination(nn.Module):
    '''
    Style Hallucination Module.
    Reference:
      Zhao et al. Style-Hallucinated Dual Consistency Learning for Domain Generalized Semantic Segmentation. ECCV 2022.
      https://arxiv.org/pdf/2204.02548.pdf
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.concentration_mean = torch.tensor([args.concentration_coeff]*args.batch_size)
        self.concentration_std = torch.tensor([args.concentration_coeff]*args.batch_size)
        self.concentration_coeff = args.concentration_coeff
        self.batch_size = args.batch_size
        self._dirichlet_mean = torch.distributions.dirichlet.Dirichlet(concentration=self.concentration_mean)
        self._dirichlet_std = torch.distributions.dirichlet.Dirichlet(concentration=self.concentration_std)

        # self.register_buffer("proto_mean", torch.zeros((args.base_style_num,args.style_dim), requires_grad=False))
        # self.register_buffer("proto_std", torch.zeros((args.base_style_num,args.style_dim), requires_grad=False))


    def forward(self, x):
        B,C,H,W = x.size()
        x_mean = x.mean(dim=[2,3], keepdim=True) # B,C,1,1
        x_std = x.std(dim=[2,3], keepdim=True) + 1e-7 # B,C,1,1
        # x_mean, x_std = x_mean.detach(), x_std.detach()

        if B == self.batch_size:
            combine_weights_mean = self._dirichlet_mean.sample((B,)).to(x.device)
            combine_weights_mean = combine_weights_mean.detach()
            combine_weights_std = self._dirichlet_std.sample((B,)).to(x.device)
            combine_weights_std = combine_weights_std.detach()
        else:
            dirichlet_mean = torch.tensor([self.concentration_coeff]*B)
            dirichlet_std = torch.tensor([self.concentration_coeff]*B)
            combine_weights_mean = torch.distributions.dirichlet.Dirichlet(concentration=dirichlet_mean).sample((B,)).to(x.device)
            combine_weights_mean = combine_weights_mean.detach()
            combine_weights_std = torch.distributions.dirichlet.Dirichlet(concentration=dirichlet_std).sample((B,)).to(x.device)
            combine_weights_std = combine_weights_std.detach()

        x_norm = (x - x_mean) / x_std
        
        new_mean = combine_weights_mean @ x_mean.squeeze(-1).squeeze(-1) # B,C
        new_std = combine_weights_std @ x_std.squeeze(-1).squeeze(-1)

        x_new = x_norm * new_std.unsqueeze(-1).unsqueeze(-1) + new_mean.unsqueeze(-1).unsqueeze(-1)

        return x, x_new



class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn
        print("output_stride = ", output_stride)
        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 4:
            rates = [4 * r for r in rates]
        elif output_stride == 16:
            pass
        elif output_stride == 32:
            rates = [r // 2 for r in rates]
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=1, bias=False),
            Norm2d(256), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class DeepV3Plus(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-101', args=None):
        super(DeepV3Plus, self).__init__()
        self.args = args
        self.trunk = trunk

        if 'concentration_coeff' in vars(args): ## SHM not use in validation
            self.SHM = StyleHallucination(args)

        channel_1st = 4
        self.channel_conv = nn.Sequential(
            nn.Conv2d(channel_1st, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3)
        )
        channel_2nd = 64
        channel_3rd = 256
        channel_4th = 512
        prev_final_channel = 1024
        final_channel = 2048

        if trunk == 'resnet-50':
            resnet = resnet50()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-101':
            resnet = resnet101()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4


        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        os = 16


        self.output_stride = os
        self.aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256,
                                                    output_stride=os)

        self.bot_fine = nn.Sequential(
            nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True))

        self.bot_aspp = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final2 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))

        self.dsn = nn.Sequential(
            nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
            Norm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        
        initialize_weights(self.dsn)

        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        # Setting the flags
        self.eps = 1e-5


    def forward(self, x, style_hallucination=False, out_prob=False, return_style_features=False):

        if isinstance(return_style_features, list) and len(return_style_features):
            multi_feature = True
        else:
            multi_feature = False

        x_size = x.size()
        x = self.channel_conv(x)   
        x = self.layer0(x)

        # ## return style_features for style compose is return the feature before layer1
        if not multi_feature and return_style_features:
            return x
        f_style = {}

        if style_hallucination:
            x, x_style = self.SHM(x)
            x = torch.cat((x, x_style),dim=0)
            
        f_style['layer0'] = x

        x = self.layer1(x)  # 400
        low_level = x
        f_style['layer1'] = low_level

        x = self.layer2(x)  # 100
        f_style['layer2'] = x

        x = self.layer3(x)  # 100
        aux_out = x
        f_style['layer3'] = aux_out

        x = self.layer4(x)  # 100
        f_style['layer4'] = x

        # f_style_return = {}
        all_levels = list(f_style.keys())
        if multi_feature:
            for k in all_levels:
                if k not in return_style_features:
                    del f_style[k]

        x = self.aspp(x)
        dec0_up = self.bot_aspp(x)

        dec0_fine = self.bot_fine(low_level)
        dec0_up = Upsample(dec0_up, low_level.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = self.final1(dec0)
        dec2 = self.final2(dec1)
        main_out = Upsample(dec2, x_size[2:])
        
        if self.training or out_prob:
            aux_out = self.dsn(aux_out)
            output = {
                'main_out': main_out,
                'aux_out': aux_out,
                'features': f_style
            }
                
            return output
        else:
            return main_out




def DeepR50V3PlusD(args, num_classes):
    """
    Resnet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-50")
    return DeepV3Plus(num_classes, trunk='resnet-50', args=args)


def DeepR101V3PlusD(args, num_classes):
    """
    Resnet 101 Based Network, the origin 7x7
    """
    print("Model : DeepLabv3+, Backbone : ResNet-101")
    return DeepV3Plus(num_classes, trunk='resnet-101', args=args)

if __name__ == "__main__":
    # Test the network
    import torch
    from torch.autograd import Variable
    from torchviz import make_dot
    import numpy as np
    import os
    import argparse   
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument('--concentration_coeff', type=float, default=0.0156,
                    help='coefficient for concentration')
    parser.add_argument('--base_style_num', type=int, default=64,
                    help='num of base style for style space, it should be same with the style dim, and it can also be larger for over modeling')
    parser.add_argument('--style_dim', type=int, default=64,
                    help='style compose dimension')
    parser.add_argument('--rc_layers', nargs='*', type=str, default=['layer4'],
                    help='a list of layers for retrospection loss : layer 0,1,2,3,4')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--rc_weights', nargs='*', type=float, default=[1.0],
                    help='weight for each layer feature of retrospection layer')
    
    args = parser.parse_args()
    # Network
    net = DeepR50V3PlusD(num_classes=8, args=args)
    net = net.cuda(0)

    # Test the network
    x = Variable(torch.randn(4, 8, 512, 512))
    x = x.cuda(0)
    y = net(x, style_hallucination=True, out_prob=True, return_style_features=args.rc_layers)
    print(y.size())

    # Visualize the network
    g = make_dot(y)
    g.render('deeplabv3plus_resnet')
    print('Done')