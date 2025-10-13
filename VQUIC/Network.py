"""
VQUIC Network Architecture
A learned image compression network with vector quantization and adaptive texture warping.
"""
import math
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from einops import rearrange
from distutils.version import LooseVersion
from timm.models.layers import trunc_normal_
from thop import profile
from ptflops import get_model_complexity_info

from compressai.models.priors import CompressionModel, GaussianConditional
from compressai.ops import ste_round
from compressai.models.utils import conv, deconv, update_registered_buffers
from ELICUtilis.layers import AttentionBlock, conv3x3, CheckboardMaskedConv2d
from ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from utils import get_root_logger
from quantize import VectorQuantizer2 as VectorQuantizer
from models.suim_net import SUIMNet
from discriminator import Discriminator
from torchvision import models

# Compression scale parameters from Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Generate exponentially spaced scale table for quantization."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution layer."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

class ResidualBottleneckBlock(nn.Module):
    """Residual bottleneck block with channel reduction.

    Args:
        in_ch (int): Number of input channels
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv1 = conv1x1(in_ch, in_ch // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_ch // 2, in_ch // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(in_ch // 2, in_ch)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return out + identity


class LayerNormFunction(torch.autograd.Function):
    """Custom LayerNorm implementation for 2D feature maps."""
    
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        weight, bias, y = weight.contiguous(), bias.contiguous(), y.contiguous()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None

class LayerNorm2d(nn.Module):
    """2D Layer Normalization for spatial features.
    
    Args:
        channels (int): Number of channels
        eps (float): Epsilon for numerical stability
        requires_grad (bool): Whether parameters are trainable
    """

    def __init__(self, channels, eps=1e-6, requires_grad=True):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels), requires_grad=requires_grad))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels), requires_grad=requires_grad))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class TWAM(nn.Module):
    """Texture Warping Attention Module.
    
    Args:
        c (int): Number of channels
        num_heads (int): Number of attention heads
    """
    
    def __init__(self, c, num_heads):
        super(TWAM, self).__init__()
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.num_heads = num_heads
        self.norm_d = LayerNorm2d(c)
        self.norm_g = LayerNorm2d(c)
        self.d_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.g_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.d_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.g_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_d, x_g):
        b, c, h, w = x_d.shape

        Q_d = self.d_proj1(self.norm_d(x_d))
        Q_g_T = self.g_proj1(self.norm_g(x_g))
        V_d = self.d_proj2(x_d)
        V_g = self.g_proj2(x_g)

        Q_d = rearrange(Q_d, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        Q_g_T = rearrange(Q_g_T, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        V_d = rearrange(V_d, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        V_g = rearrange(V_g, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        Q_d = F.normalize(Q_d, dim=-1)
        Q_g_T = F.normalize(Q_g_T, dim=-1)

        attention = (Q_d @ Q_g_T.transpose(-2, -1)) * self.scale
        F_g2d = torch.matmul(F.softmax(attention, dim=-1), V_g)
        F_d2g = torch.matmul(F.softmax(attention, dim=-1), V_d)

        F_g2d = rearrange(F_g2d, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        F_d2g = rearrange(F_d2g, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return x_d + F_g2d * self.beta, x_g + F_d2g * self.gamma

class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated Deformable Convolution for texture alignment.
    
    Generates offsets and masks from separate feature inputs for better alignment.
    Reference: Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, 
                                                 self.padding, self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, 
                                         self.padding, self.dilation, self.groups, self.deformable_groups)

class TextureWarpingModule(nn.Module):
    """Texture Warping Module for feature alignment using deformable convolution.
    
    Args:
        channel (int): Number of channels
        cond_channels (int): Number of conditional channels
        deformable_groups (int): Number of deformable groups
        previous_offset_channel (int): Number of previous offset channels
    """

    def __init__(self, channel, cond_channels, deformable_groups, previous_offset_channel=0):
        super(TextureWarpingModule, self).__init__()
        self.offset_conv1 = nn.Sequential(
            nn.Conv2d(channel + cond_channels, channel, kernel_size=1),
            nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, groups=channel, kernel_size=7, padding=3),
            nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=1)
        )

        self.offset_conv2 = nn.Sequential(
            nn.Conv2d(channel + previous_offset_channel, channel, 3, 1, 1),
            nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True),
            nn.SiLU(inplace=True)
        )
        self.dcn = DCNv2Pack(channel, channel, 3, padding=1, deformable_groups=deformable_groups)

    def forward(self, x_main, inpfeat, previous_offset=None):
        offset = self.offset_conv1(torch.cat([inpfeat, x_main], dim=1))
        if previous_offset is None:
            offset = self.offset_conv2(offset)
        else:
            offset = self.offset_conv2(torch.cat([offset, previous_offset], dim=1))
        warp_feat = self.dcn(x_main, offset)
        return warp_feat, offset

class NormLayer(nn.Module):
    """Flexible normalization layer supporting multiple norm types.
    
    Args:
        channels (int): Number of input channels
        norm_type (str): Type of normalization ('bn', 'in', 'gn', 'none')
    """
    
    def __init__(self, channels, norm_type='bn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        self.norm_type = norm_type
        self.channels = channels
        
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels, affine=True)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=False)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        elif norm_type == 'none':
            self.norm = lambda x: x * 1.0
        else:
            raise ValueError(f'Norm type {norm_type} not supported.')

    def forward(self, x):
        return self.norm(x)

class ActLayer(nn.Module):
    """Flexible activation layer supporting multiple activation types.
    
    Args:
        channels (int): Number of channels (used for PReLU)
        relu_type (str): Type of activation ('relu', 'leakyrelu', 'prelu', 'silu', 'gelu', 'none')
    """
    
    def __init__(self, channels, relu_type='leakyrelu'):
        super(ActLayer, self).__init__()
        relu_type = relu_type.lower()
        
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'none':
            self.func = lambda x: x * 1.0
        elif relu_type == 'silu':
            self.func = nn.SiLU(True)
        elif relu_type == 'gelu':
            self.func = nn.GELU()
        else:
            raise ValueError(f'Activation type {relu_type} not supported.')

    def forward(self, x):
        return self.func(x)

class ResBlock(nn.Module):
    """Preactivation residual block.
    
    Args:
        in_channel (int): Number of input channels
        out_channel (int): Number of output channels
        norm_type (str): Type of normalization
        act_type (str): Type of activation
    """
    
    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            NormLayer(in_channel, norm_type),
            ActLayer(in_channel, act_type),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            NormLayer(out_channel, norm_type),
            ActLayer(out_channel, act_type),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
        )

    def forward(self, input):
        return self.conv(input) + input

class ResnetBlock(nn.Module):
    """ResNet-style block with GroupNorm and SiLU activation.
    
    Args:
        channels_in (int): Number of input channels
        channels_out (int): Number of output channels
    """

    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=channels_in, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=channels_out, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU(inplace=True)
        
        if channels_in != channels_out:
            self.residual_func = nn.Conv2d(channels_in, channels_out, kernel_size=1)
        else:
            self.residual_func = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        return x + self.residual_func(residual)

class UpDownSample(nn.Module):
    """Flexible upsampling or downsampling module.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        scale_factor (float): Scaling factor for spatial dimensions
        direction (str): 'up' for upsampling or 'down' for downsampling
    """

    def __init__(self, in_channels, out_channels, scale_factor, direction):
        super().__init__()
        self.scale_factor = scale_factor
        self.direction = direction
        assert direction in ['up', 'down'], "Direction must be 'up' or 'down'"
        
        if self.scale_factor != 1:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        if self.scale_factor != 1:
            _, _, h, w = x.shape
            if self.direction == 'up':
                new_h, new_w = int(self.scale_factor * h), int(self.scale_factor * w)
                x = self.conv(x)
                x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            else:
                new_h, new_w = int(h / self.scale_factor), int(w / self.scale_factor)
                x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
                x = self.conv(x)
        return x

class VQGANDecoder(nn.Module):
    """VQGAN-style hierarchical decoder.
    
    Args:
        base_channels (int): Base number of channels
        proj_patch_size (int): Projection patch size (must be power of 2)
        resolution_scale_rates (list): Scale rates for each resolution level
        channel_multipliers (list): Channel multipliers for each level
        decoder_num_blocks (int): Number of ResNet blocks per level
    """

    def __init__(self, base_channels, proj_patch_size, resolution_scale_rates, channel_multipliers, decoder_num_blocks):
        super(VQGANDecoder, self).__init__()
        self.log_size = int(math.log(proj_patch_size, 2))
        self.channel_dict = {}
        self.resolution_scalerate_dict = {}

        resolution_scale_rates = resolution_scale_rates[::-1]

        for idx, scale in enumerate(range(self.log_size + 1)):
            self.channel_dict[f'Level_{2**scale}'] = channel_multipliers[idx] * base_channels
            self.resolution_scalerate_dict[f'Level_{2**scale}'] = resolution_scale_rates[idx]

        self.decoder_dict = nn.ModuleDict()
        self.pre_upsample_dict = nn.ModuleDict()

        for scale in range(self.log_size, -1, -1):
            if scale == self.log_size:
                in_channel = self.channel_dict[f'Level_{2**scale}']
            else:
                in_channel = self.channel_dict[f'Level_{2**(scale + 1)}']
            stage_channel = self.channel_dict[f'Level_{2**scale}']
            upsample_rate = self.resolution_scalerate_dict[f'Level_{2**scale}']

            self.decoder_dict[f'Level_{2**scale}'] = nn.Sequential(
                *[ResnetBlock(stage_channel, stage_channel) for _ in range(decoder_num_blocks)]
            )
            self.pre_upsample_dict[f'Level_{2**scale}'] = UpDownSample(
                in_channel, stage_channel, upsample_rate, direction='up'
            )

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=self.channel_dict['Level_1'], eps=1e-6, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.channel_dict['Level_1'], 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, quant_res_dict, return_feat=True):
        dec_res = {}
        x = quant_res_dict

        for scale in range(self.log_size, -1, -1):
            x = self.pre_upsample_dict[f'Level_{2**scale}'](x)
            x = self.decoder_dict[f'Level_{2**scale}'](x)
            dec_res[f'Level_{2**scale}'] = x
        x = self.conv_out(x)
        
        if return_feat:
            return x, dec_res
        else:
            return x

class MainDecoder(nn.Module):
    """Main decoder with texture warping and attention modules.
    
    Args:
        base_channels (int): Base number of channels
        channel_multipliers (list): Channel multipliers for each level
    """

    def __init__(self, base_channels, channel_multipliers):
        super(MainDecoder, self).__init__()
        self.num_levels = len(channel_multipliers)

        self.decoder_dict = nn.ModuleDict()
        self.pre_upsample_dict = nn.ModuleDict()
        self.align_func_dict = nn.ModuleDict()
        self.attn = nn.ModuleDict()

        for i in reversed(range(self.num_levels)):
            if i == self.num_levels - 1:
                channels_prev = base_channels * channel_multipliers[i]
            else:
                channels_prev = base_channels * channel_multipliers[i + 1]
            channels = base_channels * channel_multipliers[i]

            if i != self.num_levels - 1:
                self.pre_upsample_dict[f'Level_{2**i}'] = nn.Sequential(
                        nn.UpsamplingNearest2d(scale_factor=2),
                    nn.Conv2d(channels_prev, channels, kernel_size=3, padding=1)
                )

            previous_offset_channel = 0 if i == self.num_levels - 1 else channels_prev
            self.attn[f'Level_{2**i}'] = TWAM(channels, num_heads=channel_multipliers[i])
            self.align_func_dict[f'Level_{2**i}'] = TextureWarpingModule(
                    channel=channels,
                    cond_channels=channels,
                    deformable_groups=4,
                previous_offset_channel=previous_offset_channel
            )

            if i != self.num_levels - 1:
                self.decoder_dict[f'Level_{2**i}'] = ResnetBlock(2 * channels, channels)

    def forward(self, dec_res_dict, x_d, fidelity_ratio=1.0):
        x_d, x_g = self.attn[f'Level_{2**(self.num_levels - 1)}'](x_d, dec_res_dict['Level_8'])
        x_d, offset = self.align_func_dict[f'Level_{2**(self.num_levels - 1)}'](x_d, x_g)

        for scale in reversed(range(self.num_levels - 1)):
            x_d = self.pre_upsample_dict[f'Level_{2**scale}'](x_d)
            upsample_offset = F.interpolate(offset, scale_factor=2, align_corners=False, mode='bilinear') * 2
            x_d, x_g = self.attn[f'Level_{2**scale}'](x_d, dec_res_dict[f'Level_{2**scale}'])
            warp_feat, offset = self.align_func_dict[f'Level_{2**scale}'](x_d, x_g, previous_offset=upsample_offset)
            x_d = self.decoder_dict[f'Level_{2**scale}'](torch.cat([x_d, warp_feat], dim=1))
        return dec_res_dict['Level_1'] + fidelity_ratio * x_d

class Quantizer:
    """Flexible quantizer supporting multiple quantization strategies."""
    
    def quantize(self, inputs, quantize_type="noise"):
        """Apply quantization to inputs.
        
        Args:
            inputs: Input tensor
            quantize_type: Type of quantization ('noise', 'ste', or 'round')
        
        Returns:
            Quantized tensor
        """
        if quantize_type == "noise":
            half = 0.5
            noise = torch.empty_like(inputs).uniform_(-half, half)
            return inputs + noise
        elif quantize_type == "ste":
            return torch.round(inputs) - inputs.detach() + inputs
        else:
            return torch.round(inputs)


class Vgg19(nn.Module):
    """VGG19 feature extractor for perceptual loss.
    
    Args:
        requires_grad (bool): Whether to compute gradients for VGG parameters
    """
    
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class VGGLoss(nn.Module):
    """VGG perceptual loss for image quality assessment.
    
    Computes weighted L1 loss across multiple VGG19 feature layers.
    """
    
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        loss = sum(self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach()) 
                   for i in range(len(x_vgg)))
        return loss

class TestModel(CompressionModel):
    """VQUIC Compression Model with Vector Quantization and Adaptive Decoding.
    
    Args:
        N (int): Channel number of main encoder network
        M (int): Channel number of latent space
        num_slices (int): Number of slices for channel-wise entropy coding
    """

    def __init__(self, N=192, M=320, num_slices=5, **kwargs):
        super().__init__(entropy_bottleneck_channels=192)
        self.N = int(N)
        self.M = int(M)
        self.num_slices = num_slices
        self.groups = [0, 16, 16, 32, 64, 192]
        # Analysis transform (encoder)
        self.g_a = nn.Sequential(
            conv(3, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            conv(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            AttentionBlock(N),
            conv(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            conv(N, M),
            AttentionBlock(M),
        )

        self.quant_conv = nn.Conv2d(M, 256, 1)

        # Hyperprior analysis
        self.h_a = nn.Sequential(
            conv3x3(M, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        # Hyperprior synthesis
        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N * 3 // 2),
            nn.ReLU(inplace=True),
            conv3x3(N * 3 // 2, 2 * M),
        )

        self.quantizer = Quantizer()
        # Decoder modules
        self.decoder = VQGANDecoder(
            base_channels=64,
            proj_patch_size=8,
            resolution_scale_rates=[1, 2, 2, 2],
            channel_multipliers=[1, 2, 4, 4],
            decoder_num_blocks=3
        )
        self.main_decoder = MainDecoder(base_channels=64, channel_multipliers=[1, 2, 4, 4])
        self.decoder.load_state_dict(torch.load("VQCNIR_LOLBlur_G.pth")['params'], strict=False)
        self.deconv_end = nn.Sequential(deconv(64, 3))

        # Segmentation network
        self.suimnet = SUIMNet(base='VGG', n_classes=7)
        self.suimnet.load_state_dict(torch.load("ckpt/SUIM_epoch_37_loss0.1613_acc_93.2023.pth"))

        # Vector quantizers
        self.quantize = VectorQuantizer(16384, 256, beta=0.25, remap=None, sane_index_shape=False)
        self.good_quantize = VectorQuantizer(16384, 32, beta=0.25, remap=None, sane_index_shape=False)
        self.bad_quantize = VectorQuantizer(16384, 32, beta=0.25, remap=None, sane_index_shape=False)

        # Loss modules
        self.criterionVGG = VGGLoss()
        self.discriminator = Discriminator()


    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


    def forward(self, x, noisequant=False):
        """Forward pass with vector quantization and adaptive decoding.
        
        Args:
            x: Input image tensor
            noisequant: Whether to use noise quantization during training
        
        Returns:
            dict: Dictionary containing reconstructed image, losses, and auxiliary outputs
        """
        y = self.g_a(x)
        B, C, H, W = y.size()

        yy = self.quant_conv(y)

        # Compute VQ losses
        _, good_qloss, _ = self.good_quantize(yy)
        _, bad_qloss, _ = self.bad_quantize(yy)
        y_quant, qloss, _ = self.quantize(yy)
        g_quant, good_loss_2, _ = self.good_quantize(y_quant)

        # Decode through VQGAN decoder
        dec, dec_feat_dict = self.decoder(g_quant)

        # Main decoder path with texture warping
        y_hat = self.main_decoder(dec_feat_dict, y_quant)
        x_hat = self.deconv_end(y_hat).clamp_(0, 1)

        # Compute losses
        loss_VGG = self.criterionVGG(x_hat, x)
        mask = self.suimnet(x_hat)

        disc_real = self.discriminator(x)
        disc_fake = self.discriminator(x_hat)

        gen_loss = -torch.mean(disc_fake)
        rec_loss = F.mse_loss(x_hat, x)

        if noisequant:
            λ = self.calculate_lambda(rec_loss, gen_loss)
        else:
            λ = 0.

        d_loss_real = torch.mean(F.relu(1. - disc_real))
        d_loss_fake = torch.mean(F.relu(1. + disc_fake))
        gan_loss = 0.5 * (d_loss_real + d_loss_fake)

        return {
            "x_hat": x_hat,
            "vq_loss": qloss,
            "mask": mask,
            "f_loss": loss_VGG,
            "good_vq_loss": good_qloss,
            "bad_vq_loss": bad_qloss,
            "gloss": good_loss_2,
            "gen_loss": λ * gen_loss,
            "gan_loss": gan_loss
        }


    def calculate_lambda(self, perceptual_loss, gan_loss):
        """Calculate adaptive weight for GAN loss using gradient balancing.
        
        Args:
            perceptual_loss: Perceptual loss value
            gan_loss: GAN loss value
        
        Returns:
            Balanced weight for GAN loss
        """
        last_layer = self.deconv_end[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ

    def load_state_dict(self, state_dict):
        """Load model state dict with proper buffer updates."""
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)
    
    @classmethod
    def from_state_dict(cls, state_dict):
        """Create a new model instance from state dict."""
        net = cls()
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        """Update entropy coding tables."""
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress_tensor(self, tensor):
        """Compress tensor using custom binary packing.
        
        Args:
            tensor: Input tensor with values in range [0, 16383]
        
        Returns:
            Compressed bytes
        """
        import numpy as np
        data = tensor.cpu().numpy()
        integer_list = []
        binary_str = ''

        for num in data:
            if not (0 <= num <= 16383):
                raise ValueError("All numbers must be between 0 and 16383.")
            binary_str += format(num, '014b')

        # Convert every 16 bits to an integer
        for i in range(0, len(binary_str), 16):
            binary_chunk = binary_str[i:i + 16]
            integer = int(binary_chunk, 2)
            integer_list.append(integer)

        compressed_data = np.array(integer_list, dtype=np.uint16).tobytes()
        return compressed_data

    def compress_codebook_decompress(self, x):
        """Compress using codebook and decompress.
        
        Args:
            x: Input image
        
        Returns:
            dict: Contains reconstructed image
        """
        y = self.g_a(x)
        yy = self.quant_conv(y)
        y_quant, qloss, (_, _, min_encoding_indices) = self.quantize(yy)
        g_quant, _, _ = self.good_quantize(y_quant)

        tensor_bytes = min_encoding_indices.cpu().numpy().tobytes()
        compressed_bytes = zlib.compress(tensor_bytes)
        bpp_1 = len(compressed_bytes) * 8 / (x.shape[0] * x.shape[2] * x.shape[3])
        print(f"BPP: {bpp_1}")

        dec, dec_feat_dict = self.decoder(g_quant)
        y_hat = self.main_decoder(dec_feat_dict, y_quant)
        x_hat = self.deconv_end(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def inference(self, x):
        import time
        y_enc_start = time.time()
        y = self.g_a(x)
        y_enc = time.time() - y_enc_start
        B, C, H, W = y.size()  ## The shape of y to generate the mask

        z_enc_start = time.time()
        z = self.h_a(y)
        z_enc = time.time() - z_enc_start
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        z_dec_start = time.time()
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        z_dec = time.time() - z_dec_start

        anchor = torch.zeros_like(y).to(x.device)
        non_anchor = torch.zeros_like(y).to(x.device)

        anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
        non_anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        non_anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]

        y_slices = torch.split(y, self.groups[1:], 1)

        anchor_split = torch.split(anchor, self.groups[1:], 1)
        non_anchor_split = torch.split(non_anchor, self.groups[1:], 1)
        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W).to(x.device),
                                              [2 * i for i in self.groups[1:]], 1)
        y_hat_slices = []
        y_likelihood = []
        params_start = time.time()
        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

            else:
                support_slices = torch.concat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            ##support mean and scale
            support = torch.concat([latent_means, latent_scales], dim=1) if slice_index == 0 else torch.concat(
                [support_slices_ch_mean, support_slices_ch_scale, latent_means, latent_scales], dim=1)
            ### checkboard process 1
            y_anchor = anchor_split[slice_index]
            means_anchor, scales_anchor, = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)

            scales_hat_split = torch.zeros_like(y_anchor).to(x.device)
            means_hat_split = torch.zeros_like(y_anchor).to(x.device)

            scales_hat_split[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
            scales_hat_split[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
            means_hat_split[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
            means_hat_split[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]

            y_anchor_quantilized_for_gs = self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor

            y_anchor_quantilized_for_gs[:, :, 0::2, 1::2] = 0
            y_anchor_quantilized_for_gs[:, :, 1::2, 0::2] = 0

            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_quantilized_for_gs)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)).chunk(2, 1)

            scales_hat_split[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_hat_split[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            means_hat_split[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            means_hat_split[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]
            # entropy estimation
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scales_hat_split, means=means_hat_split)

            y_non_anchor = non_anchor_split[slice_index]

            y_non_anchor_quantilized_for_gs = self.quantizer.quantize(y_non_anchor - means_non_anchor,
                                                                      "ste") + means_non_anchor
            y_non_anchor_quantilized_for_gs[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantilized_for_gs[:, :, 1::2, 1::2] = 0

            y_hat_slice = y_anchor_quantilized_for_gs + y_non_anchor_quantilized_for_gs
            y_hat_slices.append(y_hat_slice)
            ### ste for synthesis model
            y_likelihood.append(y_slice_likelihood)

        params_time = time.time() - params_start
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        y_hat = torch.cat(y_hat_slices, dim=1)
        
        y_dec_start = time.time()
        x_hat = self.g_s(y_hat)
        y_dec = time.time() - y_dec_start
        
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "time": {'y_enc': y_enc, "y_dec": y_dec, "z_enc": z_enc, "z_dec": z_dec, "params": params_time}
        }


if __name__ == "__main__":
    # Test model complexity and performance
    model = TestModel(N=192, M=320, num_slices=5)
    test_input = torch.Tensor(1, 3, 256, 256)
    
    out = model(test_input)
    print(f"Output shape: {out['x_hat'].shape}")
    
    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    print(f'FLOPs: {flops}, Params: {params}')
    
    flops, params = profile(model, (test_input,))
    print(f'FLOPs: {flops}, Params: {params}')