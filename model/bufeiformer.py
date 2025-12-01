import math
import torch
from torch import nn
import torch.nn.functional as F

# U 比 former 类 效果好，遥感还是适合 U 型 简单 结构吧， 多通道信息
# 基础的降采样模块(考虑空洞卷积) 参考MobileNext（第二个卷积 使用 分离卷积（3*3，ratio先尝试5，对应分类patch大小为11的感受野）） + inception Transformer ； 可以参考AFFormer和PID 讲频率的故事

cudan = 1

class MobileNext(nn.Module):
    def __init__(self, inp, mid):
        super(MobileNext,self).__init__()
        self.inp = inp
        self.mid = mid
        self.relu = nn.ReLU6(inplace=True)
        self.conv_3x3_bn = nn.Sequential(
            nn.Conv2d(self.inp, self.inp, 3, stride=1, padding=1, dilation=1,  bias=False),
            nn.BatchNorm2d(self.inp),
            nn.ReLU6(inplace=True)
        )
        self.conv_3x3_bn1 = nn.Sequential(
            nn.Conv2d(self.inp, self.inp, 3, stride=1, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(self.inp),
            nn.ReLU6(inplace=True)
        )

        self.redu = nn.Conv2d(self.inp, self.mid, 1, 1)
        self.up = nn.Conv2d(self.mid, self.inp, 1, 1)

    def forward(self, x):
        x1 = self.conv_3x3_bn(x)
        x1 = self.relu(x1+3)/6
        x1 = self.redu(x1)
        x1 = self.up(x1)
        x1 = self.conv_3x3_bn1(x1)
        x1 = self.relu(x1 + 3) / 6
        return x + x1



class inception(nn.Module):
    def __init__(self, inp, mid, num_heads,):
        super(inception, self).__init__()
        self.inp = inp
        self.mid = mid
        self.num_heads = num_heads
        self.key_dim = int(self.inp / self.num_heads)
        self.scale = self.key_dim ** -0.5

        self.to_q = nn.Conv2d(self.inp, self.inp, 1)
        self.to_k = nn.Conv2d(self.inp, self.inp, 1)
        self.to_v = nn.Conv2d(self.inp, self.inp*16, 1)

        self.con11 = nn.Conv2d(self.inp, self.mid, kernel_size=1, stride=1)
        self.con12 = nn.Conv2d(self.mid, self.inp, kernel_size=1, stride=1)
        self.dw = nn.Conv2d(self.inp, self.inp, kernel_size=3, stride=1, padding=1)


    def lowpath(self, x):
        n, c, h, w = x.shape
        x = nn.Conv2d(c, c, 3, stride=2, padding=1).cuda(cudan)(x)
        x = nn.Conv2d(c, c, 3, stride=2, padding=1).cuda(cudan)(x)

        B, C, H, W = x.shape

        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, 16*self.key_dim, H * W).permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk) * self.scale
        attn = attn.softmax(dim=-1)  # dim = k

        xx = torch.matmul(attn, vv)
        nn.LayerNorm(128, eps=1e-6).cuda(cudan)(xx)
        xx = xx.permute(0, 1, 3, 2).reshape(B, C, H * 4, W * 4)
        # xx = xx.reshape(B, C/16, 4*H, 4*W)
        return xx

    def high(self, x):
        x1 = self.con11(x)
        x1 = self.con12(x1)
        x1 = self.dw(x1)
        return x + x1

    def forward(self, x):
        x1 = self.lowpath(x)
        x2 = self.high(x)
        return x1 + x2


class Block(nn.Module):
    def __init__(self, inp=64, mid=16, num_heads=8):
        super(Block, self).__init__()
        self.mob = MobileNext(inp, mid)
        self.ince = inception(inp, mid, num_heads)

    def forward(self,x):
        x = self.mob(x)
        x = self.ince(x)
        return x


# skip选层连接 使用list


# 不同层之间的特征的语义交换(考虑注意力)


# 不同层之间插值统一和的语义融合


# stem 为一个3*3 的空洞卷积 + 1*1  降维
class stem(nn.Module):
    def __init__(self, inp, oup):
        super(stem, self).__init__()
        self.conv33 = nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(inplace=True)
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(oup, oup, 1, stride=1),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv33(x)
        x = self.conv11(x)
        return x

# 主网络
class BufeiFormer(nn.Module):
    def __init__(self, n_class , oriinp = 32, mid=32, oup=64):
        super(BufeiFormer, self).__init__()
        self.oriinp = oriinp
        self.mid = mid
        self.oup = oup
        self.stem = stem(inp=self.oriinp, oup=self.oup)
        self.block = Block(inp=64, mid=16, num_heads=8)  # 特征提取
        self.block1 = Block(inp=64, mid=16, num_heads=8)  # 特征还原

        self.res = nn.Conv2d(self.oup, n_class, 1)


    def get_weight(self,rem):
        B, C, H, W = rem[0].shape
        x = torch.cat([nn.functional.adaptive_avg_pool2d(inp, (1,1)) for inp in rem], dim=1)


        # 计算注意力
        toq = nn.Sequential(nn.Linear(in_features=3*C, out_features=3*C, bias=True), nn.Sigmoid()).cuda(cudan)

        tok = nn.Sequential(nn.Linear(in_features=3*C, out_features=3*C, bias=True), nn.Sigmoid()).cuda(cudan)
        tov = nn.Sequential(nn.Linear(in_features=3*C, out_features=3*C, bias=True), nn.Sigmoid()).cuda(cudan)

        num_heads = 8
        key_dim = 8
        scale = key_dim ** -0.5

        x = x.squeeze()

        qq = toq(x).reshape(B, num_heads, key_dim, 3).permute(0, 1, 3, 2)
        kk = tok(x).reshape(B, num_heads, key_dim, 3)
        vv = tov(x).reshape(B, num_heads, key_dim, 3).permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk) * scale
        attn = attn.softmax(dim=-1)  # dim = k

        xx = torch.matmul(attn, vv)
        nn.LayerNorm(8, eps=1e-6).cuda(cudan)(xx)
        xx = xx.permute(0, 1, 3, 2).reshape(B, 3*C)

        return xx


    def se_fuse(self, rem, weight):
        B, C, H, W = rem[0].shape
        rem[0] = rem[0]*weight[:, 0:C, None,None]
        rem[0] = rem[1]*weight[:, C:2*C, None,None]
        rem[0] = rem[2]*weight[:, 2*C:, None,None]
        return rem


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n //= m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self,x):
        rem = list()
        x = self.stem(x)
        rem.append(x)
        x = self.block(x)
        rem.append(x)
        x = self.block(x)
        rem.append(x)
        weight = self.get_weight(rem)
        skip = self.se_fuse(rem, weight)
        # x = self.block1(x) + skip[2]
        x = skip[2]
        x = self.block1(x) + skip[1]
        x = self.block1(x) + skip[0]
        x = self.res(x)
        return x


### 不同的block

class BufeiFormer1(nn.Module):
    def __init__(self, n_class , oriinp = 32, mid=32, oup=64):
        super(BufeiFormer1, self).__init__()
        self.oriinp = oriinp
        self.mid = mid
        self.oup = oup
        self.stem = stem(inp=self.oriinp, oup=self.oup)
        self.block = Block(inp=64, mid=16, num_heads=8)  # 特征提取
        self.block1 = Block(inp=64, mid=16, num_heads=8)  # 特征提取
        self.block2 = Block(inp=64, mid=16, num_heads=8)  # 特征还原
        self.block3 = Block(inp=64, mid=16, num_heads=8)  # 特征还原

        self.res = nn.Conv2d(self.oup, n_class, 1)


    def get_weight(self,rem):
        B, C, H, W = rem[0].shape
        x = torch.cat([nn.functional.adaptive_avg_pool2d(inp, (1,1)) for inp in rem], dim=1)


        # 计算注意力
        toq = nn.Sequential(nn.Linear(in_features=3*C, out_features=3*C, bias=True), nn.Sigmoid()).cuda(cudan)

        tok = nn.Sequential(nn.Linear(in_features=3*C, out_features=3*C, bias=True), nn.Sigmoid()).cuda(cudan)
        tov = nn.Sequential(nn.Linear(in_features=3*C, out_features=3*C, bias=True), nn.Sigmoid()).cuda(cudan)

        num_heads = 8
        key_dim = 8
        scale = key_dim ** -0.5

        x = x.squeeze()

        qq = toq(x).reshape(B, num_heads, key_dim, 3).permute(0, 1, 3, 2)
        kk = tok(x).reshape(B, num_heads, key_dim, 3)
        vv = tov(x).reshape(B, num_heads, key_dim, 3).permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk) * scale
        attn = attn.softmax(dim=-1)  # dim = k

        xx = torch.matmul(attn, vv)
        nn.LayerNorm(8, eps=1e-6).cuda(cudan)(xx)
        xx = xx.permute(0, 1, 3, 2).reshape(B, 3*C)

        return xx


    def se_fuse(self, rem, weight):
        B, C, H, W = rem[0].shape
        rem[0] = rem[0]*weight[:, 0:C, None,None]
        rem[0] = rem[1]*weight[:, C:2*C, None,None]
        rem[0] = rem[2]*weight[:, 2*C:, None,None]
        return rem


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n //= m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self,x):
        rem = list()
        x = self.stem(x)
        rem.append(x)
        x = self.block(x)
        rem.append(x)
        x = self.block1(x)
        rem.append(x)
        weight = self.get_weight(rem)
        skip = self.se_fuse(rem, weight)
        # x = self.block1(x) + skip[2]
        x = skip[2]
        x = self.block2(x) + skip[1]
        x = self.block3(x) + skip[0]
        x = self.res(x)
        return x




# 另外一个更轻量的方法，直接concat到一起

class BufeiFormer2(nn.Module):
    def __init__(self, n_class , oriinp = 32, mid=32, oup=64):
        super(BufeiFormer2, self).__init__()
        self.oriinp = oriinp
        self.mid = mid
        self.oup = oup
        self.stem = stem(inp=self.oriinp, oup=self.oup)
        self.block = Block(inp=64, mid=16, num_heads=8)  # 特征提取
        self.block1 = Block(inp=64, mid=16, num_heads=8)  # 特征提取
        self.block2 = Block(inp=64, mid=16, num_heads=8)  # 特征还原
        self.block3 = Block(inp=64, mid=16, num_heads=8)  # 特征还原

        self.res = nn.Conv2d(self.oup*3, n_class, 1)


    def get_weight(self,rem):
        B, C, H, W = rem[0].shape
        x = torch.cat([nn.functional.adaptive_avg_pool2d(inp, (1,1)) for inp in rem], dim=1)


        # 计算注意力
        toq = nn.Sequential(nn.Linear(in_features=3*C, out_features=3*C, bias=True), nn.Sigmoid()).cuda(cudan)

        tok = nn.Sequential(nn.Linear(in_features=3*C, out_features=3*C, bias=True), nn.Sigmoid()).cuda(cudan)
        tov = nn.Sequential(nn.Linear(in_features=3*C, out_features=3*C, bias=True), nn.Sigmoid()).cuda(cudan)

        num_heads = 8
        key_dim = 8
        scale = key_dim ** -0.5

        x = x.squeeze()

        qq = toq(x).reshape(B, num_heads, key_dim, 3).permute(0, 1, 3, 2)
        kk = tok(x).reshape(B, num_heads, key_dim, 3)
        vv = tov(x).reshape(B, num_heads, key_dim, 3).permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk) * scale
        attn = attn.softmax(dim=-1)  # dim = k

        xx = torch.matmul(attn, vv)
        nn.LayerNorm(8, eps=1e-6).cuda(cudan)(xx)
        xx = xx.permute(0, 1, 3, 2).reshape(B, 3*C)

        return xx


    def se_fuse(self, rem, weight):
        B, C, H, W = rem[0].shape
        rem[0] = rem[0]*weight[:, 0:C, None,None]
        rem[0] = rem[1]*weight[:, C:2*C, None,None]
        rem[0] = rem[2]*weight[:, 2*C:, None,None]
        return rem


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n //= m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self,x):
        rem = list()
        x = self.stem(x)
        rem.append(x)
        x = self.block(x)
        rem.append(x)
        x = self.block1(x)
        rem.append(x)
        weight = self.get_weight(rem)
        skip = self.se_fuse(rem, weight)
        # x = self.block1(x) + skip[2]
        # x = skip[2]
        # x = self.block2(x) + skip[1]
        # x = self.block3(x) + skip[0]
        x = torch.cat((skip[0], skip[1], skip[2]), dim=1)
        x = self.res(x)
        return x