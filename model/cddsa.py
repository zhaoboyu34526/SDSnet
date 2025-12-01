import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(".")
import time

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = F.relu(out, inplace=True)
        out = self.fc2(out)
        out = F.relu(out, inplace=True)
        out = self.fc3(out)
        out = F.relu(out, inplace=True)

        return out


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class Decoder(nn.Module):
    def __init__(self, dim, out_channel):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.adain1 = AdaptiveInstanceNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.adain2 = AdaptiveInstanceNorm2d(dim)
        self.conv3 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.adain3 = AdaptiveInstanceNorm2d(dim)
        self.conv4 = nn.Conv2d(dim, out_channel, 3, 1, 1, bias=True)

    def forward(self,x):
        out = self.conv1(x)
        out = self.adain1(out)
        out = self.conv2(out)
        out = self.adain2(out)
        out = self.conv3(out)
        out = self.adain3(out)
        out = self.conv4(out)
        out = torch.tanh(out)
        return out

class Ada_Decoder(nn.Module):
    def __init__(self, anatomy_out_channel, z_length, out_channel):
        super(Ada_Decoder, self).__init__()
        self.dec = Decoder(anatomy_out_channel, out_channel)
        self.mlp = MLP(z_length, self.get_num_adain_params(self.dec), 256)

    def forward(self, anatomy, style):
        adain_params = self.mlp(style)  # [bs, z_length] --> [4, 48]
        self.assgin_adain_params(adain_params, self.dec)
        images = self.dec(anatomy)
        return images

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

    def assgin_adain_params(self, adain_params, model):
        """
        Assign the adain_params to the AdaIN layers in model
        """
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]


class _DomainSpecificBatchNorm(nn.Module):
    _version = 2

    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_DomainSpecificBatchNorm, self).__init__()
        #         self.bns = nn.ModuleList([nn.modules.batchnorm._BatchNorm(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        bn = self.bns[domain_label]
        return bn(x)


class DomainSpecificBatchNorm2d(_DomainSpecificBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

if __name__ == '__main__':
    bn = DomainSpecificBatchNorm2d(16, 5)
    for name, paramter in bn.named_parameters():
        print(f'{name}:{paramter.shape}')

def normalize(x, norm_type, num_domain=1):
    if norm_type == 'batchnorm':
        return nn.BatchNorm2d(x)
    elif norm_type == 'instancenorm':
        return nn.InstanceNorm2d(x)
    elif norm_type == 'dsbn':
        return DomainSpecificBatchNorm2d(x, num_domain)
    else:
        return nn.BatchNorm2d(x) #temp

def deconv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def deconv(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
    )

def conv_lrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.LeakyReLU(0.2, inplace=True)
    )

def conv_bn_lrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )

def conv_in_lrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )

def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def conv_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True)
    )

def conv_no_activ(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

def conv_id_unet(in_channels, out_channels, norm='batchnorm'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0),
        normalize(out_channels, norm),
        nn.ReLU(inplace=True)
    )

def upconv(in_channels, out_channels, norm='batchnorm'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        normalize(out_channels, norm)
    )

def conv_block_unet(in_channels, out_channels, kernel_size, stride=1, padding=0, norm='batchnorm', num_dm=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        normalize(out_channels, norm, num_dm),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
        normalize(out_channels, norm, num_dm),
        nn.LeakyReLU(inplace=True),
    )

def conv_block_unet_last(in_channels, out_channels, kernel_size, stride=1, padding=0, norm='batchnorm'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        normalize(out_channels, norm),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
        normalize(out_channels, norm),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
    )

def conv_preactivation_relu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, norm='batchnorm'):
    return nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        normalize(out_channels, norm)
    )


class ResConv(nn.Module):
    def __init__(self, ndf, norm):
        super(ResConv, self).__init__()
        """
        Args:
            ndf: constant number from channels
        """
        self.ndf = ndf
        self.norm = norm
        self.conv1 = conv_preactivation_relu(self.ndf, self.ndf * 2, 3, 1, 1, self.norm)
        self.conv2 = conv_preactivation_relu(self.ndf * 2 , self.ndf * 2, 3, 1, 1, self.norm)
        self.resconv = conv_preactivation_relu(self.ndf , self.ndf * 2, 1, 1, 0, self.norm)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        residual = self.resconv(residual)

        return out + residual


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        """
        Args:
            size: expected size after interpolation
            mode: interpolation type (e.g. bilinear, nearest)
        """
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        out = self.interp(x, size=self.size, mode=self.mode) #, align_corners=False
        
        return out

def calc_vector_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    feat_var = feat.var(dim=1) + eps
    feat_std = feat_var.sqrt()
    feat_mean = feat.mean(dim=1)
    return feat_mean, feat_std

def calc_tensor_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_vector_mean_std(style_feat)
    content_mean, content_std = calc_tensor_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.view(style_std.shape[0],1,1,1).expand(size) + style_mean.view(style_mean.shape[0],1,1,1).expand(size)

def adaptive_instance_normalization2(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_vector_mean_std(style_feat)
    content_mean, content_std = calc_vector_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.unsqueeze(-1).expand(
        size)) / content_std.unsqueeze(-1).expand(size)
    return normalized_feat * style_std.unsqueeze(-1).expand(size) + style_mean.unsqueeze(-1).expand(size)

class Segmentor(nn.Module):
    def __init__(self, num_output_channels, num_class):
        super(Segmentor, self).__init__()
        self.num_output_channels = num_output_channels
        # self.num_classes = num_classes + 1 #background as extra class
        
        self.conv1 = conv_bn_lrelu(self.num_output_channels, 16, 3, 1, 1)
        self.conv2 = conv_bn_lrelu(16, 16, 1, 1, 0)
        self.pred = nn.Conv2d(16, num_class, 1, 1, 0)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pred(out)

        return out


class AEncoder(nn.Module):
    def __init__(self, in_channel, width, height, ndf, num_output_channels, norm, upsample):
        super(AEncoder, self).__init__()
        """
        UNet encoder for the anatomy factors of the image
        num_output_channels: number of spatial (anatomy) factors to encode
        """
        self.in_channel = in_channel
        self.width = width 
        self.height = height
        self.ndf = ndf
        self.num_output_channels = num_output_channels
        self.norm = norm
        self.upsample = upsample

        self.unet = UNet(self.in_channel, self.width, self.height, self.ndf, self.num_output_channels, self.norm, self.upsample)

    def forward(self, x):
        out = self.unet(x)
        out = torch.tanh(out)

        return out 


class MEncoder(nn.Module):
    def __init__(self, z_length, in_channel, img_size):
        super(MEncoder, self).__init__()
        """
        VAE encoder to extract intensity (modality) information from the image
        z_length: length of the output vector
        """
        self.z_length = z_length
        self.in_channel = in_channel
        self.img_size = img_size

        self.block1 = conv_bn_lrelu(self.in_channel, 16, 3, 2, 1)  # input channel = 1, size = 384
        self.block2 = conv_bn_lrelu(16, 32, 3, 2, 1)
        self.block3 = conv_bn_lrelu(32, 64, 3, 2, 1)
        self.block4 = conv_bn_lrelu(64, 128, 3, 2, 1)
        self.fc = nn.Linear(128*pow(self.img_size//16, 2), 32)  # 16*16*128
        self.norm = nn.BatchNorm1d(32)
        self.activ = nn.LeakyReLU(0.03, inplace=True)
        self.mu = nn.Linear(32, self.z_length)
        self.logvar = nn.Linear(32, self.z_length)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return mu + eps*std

    def encode(self, x):
        return self.mu(x), self.logvar(x)

    def forward(self, img):
        """
        input is only the image [bs,3,256,256] without concated anatomy factor
        """
        out = self.block1(img)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.fc(out.view(-1, out.shape[1] * out.shape[2] * out.shape[3]))
        out = self.norm(out)
        out = self.activ(out)

        mu, logvar = self.encode(out)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar


class UNet(nn.Module):
    def __init__(self, in_channel, width, height, ndf, num_output_channels, normalization, upsample):
        super(UNet, self).__init__()
        """
        UNet autoencoder
        """
        self.in_channel = in_channel
        self.h = height
        self.w = width
        self.norm = normalization
        self.ndf = ndf
        self.num_output_channels = num_output_channels
        self.upsample = upsample

        self.encoder_block1 = conv_block_unet(self.in_channel, self.ndf, 3, 1, 1, self.norm)
        self.encoder_block2 = conv_block_unet(self.ndf, self.ndf * 2, 3, 1, 1, self.norm)
        self.encoder_block3 = conv_block_unet(self.ndf * 2, self.ndf * 4, 3, 1, 1, self.norm)
        self.encoder_block4 = conv_block_unet(self.ndf * 4, self.ndf * 8, 3, 1, 1, self.norm)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.bottleneck = ResConv(self.ndf * 8, self.norm)

        self.decoder_upsample1 = Interpolate((self.h // 8, self.w // 8), mode=self.upsample)
        self.decoder_upconv1 = upconv(self.ndf * 16, self.ndf * 8, self.norm)
        self.decoder_block1 = conv_block_unet(self.ndf * 16, self.ndf * 8, 3, 1, 1, self.norm)
        self.decoder_upsample2 = Interpolate((self.h // 4, self.w // 4), mode=self.upsample)
        self.decoder_upconv2 = upconv(self.ndf * 8, self.ndf * 4, self.norm)
        self.decoder_block2 = conv_block_unet(self.ndf * 8, self.ndf * 4, 3, 1, 1, self.norm)
        self.decoder_upsample3 = Interpolate((self.h // 2, self.w // 2), mode=self.upsample)
        self.decoder_upconv3 = upconv(self.ndf * 4, self.ndf * 2, self.norm)
        self.decoder_block3 = conv_block_unet(self.ndf * 4, self.ndf * 2, 3, 1, 1, self.norm)
        self.decoder_upsample4 = Interpolate((self.h, self.w), mode=self.upsample)
        self.decoder_upconv4 = upconv(self.ndf * 2, self.ndf, self.norm)
        self.decoder_block4 = conv_block_unet(self.ndf * 2, self.ndf, 3, 1, 1, self.norm)
        self.classifier_conv = nn.Conv2d(self.ndf, self.num_output_channels, 3, 1, 1, 1)

    def forward(self, x):
        #encoder
        s1 = self.encoder_block1(x)
        out = self.maxpool(s1)
        s2 = self.encoder_block2(out)
        out = self.maxpool(s2)
        s3 = self.encoder_block3(out)
        out = self.maxpool(s3)
        s4 = self.encoder_block4(out)
        out = self.maxpool(s4)

        #bottleneck
        out = self.bottleneck(out)

        #decoder
        out = self.decoder_upsample1(out)
        out = self.decoder_upconv1(out)
        out = torch.cat((out, s4), 1)
        out = self.decoder_block1(out)
        out = self.decoder_upsample2(out)
        out = self.decoder_upconv2(out)
        out = torch.cat((out, s3), 1)
        out = self.decoder_block2(out)
        out = self.decoder_upsample3(out)
        out = self.decoder_upconv3(out)
        out = torch.cat((out, s2), 1)
        out = self.decoder_block3(out)
        out = self.decoder_upsample4(out)
        out = self.decoder_upconv4(out)
        out = torch.cat((out, s1), 1)
        out = self.decoder_block4(out)
        out = self.classifier_conv(out)

        return out


if __name__ == '__main__':
    images = torch.FloatTensor(4, 4, 512, 512).uniform_(-1, 1)
    codes = torch.FloatTensor(4, 8).uniform_(-1,1)
    model = MEncoder(8, 4, 512)
    model(images)
    model = AEncoder(4, 512, 512, 32, 8, 'batchnorm', 'nearest')
    model(images)
    model = Ada_Decoder(4, 8, 4)
    model(images, codes)