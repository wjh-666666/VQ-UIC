import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models

class MyUpSample2X(nn.Module):
    def __init__(self, in_channels, out_channels, f_size=3):
        super(MyUpSample2X, self).__init__()
        self.up_sample=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=f_size, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels,momentum=0.8)

    def forward(self, layer_input, skip_input):
        u = self.up_sample(layer_input)
        u = self.conv(u)
        u = F.relu(self.bn(u))
        # u = torch.cat([u, skip_input], dim=1)
        u = u+skip_input
        return u

# Residual Skip Block (RSB)
class RSB(nn.Module):
    def __init__(self, filters, kernel_size, strides=1, skip=True):
        super(RSB, self).__init__()
        f1, f2, f3, f4 = filters
        self.skip = skip
        
        # sub-block1
        self.conv1 = nn.Conv2d(f1, f1, kernel_size=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(f1, momentum=0.8)
        
        # sub-block2
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(f2, momentum=0.8)
        
        # sub-block3
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(f3, momentum=0.8)

        # optional skip connection
        if not skip:
            self.conv4 = nn.Conv2d(f1, f4, kernel_size=1, stride=strides)
            self.bn4 = nn.BatchNorm2d(f4, momentum=0.8)

    def forward(self, x):
        shortcut = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if not self.skip:
            shortcut = F.relu(self.bn4(self.conv4(shortcut)))

        x = F.relu(x + shortcut)
        return x

# SUIM Encoder with RSB blocks
class SuimEncoderRSB(nn.Module):
    def __init__(self, channels=1):
        super(SuimEncoderRSB, self).__init__()
        
        # Encoder block 1
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=5, stride=1)
        
        # Encoder block 2
        self.bn1 = nn.BatchNorm2d(64, momentum=0.8)
        self.rsb2a = RSB([64, 64, 128, 128], kernel_size=3, strides=2, skip=False)
        self.rsb2b = RSB([64, 64, 128, 128], kernel_size=3, skip=True)
        self.rsb2c = RSB([64, 64, 128, 128], kernel_size=3, skip=True)
        
        # Encoder block 3
        self.rsb3a = RSB([128, 128, 256, 256], kernel_size=3, strides=2, skip=False)
        self.rsb3b = RSB([128, 128, 256, 256], kernel_size=3, skip=True)
        self.rsb3c = RSB([128, 128, 256, 256], kernel_size=3, skip=True)
        self.rsb3d = RSB([128, 128, 256, 256], kernel_size=3, skip=True)

    def forward(self, x):
        # Encoder block 1
        enc_1 = self.conv1(x)

        # Encoder block 2
        x = F.relu(self.bn1(enc_1))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.rsb2a(x)
        x = self.rsb2b(x)
        x = self.rsb2c(x)
        enc_2 = x

        # Encoder block 3
        x = self.rsb3a(x)
        x = self.rsb3b(x)
        x = self.rsb3c(x)
        x = self.rsb3d(x)
        enc_3 = x

        return [enc_1, enc_2, enc_3]


class SuimDecoderRSB(nn.Module):
    def __init__(self, n_classes):
        super(SuimDecoderRSB, self).__init__()

        # Decoder block 1
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder block 2
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder block 3
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Final output layer
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)

    def concat_skip(self, layer_input, skip_input, filters):
        """ For concatenation of skip connections from the encoders. """
        u = F.relu(self.bn1(self.conv1(layer_input)))
        u = torch.cat((u, skip_input), dim=1)  # Concatenation on channel axis
        return u

    def forward(self, enc_inputs):
        enc_1, enc_2, enc_3 = enc_inputs
        
        # Decoder block 1
        dec_1 = F.relu(self.bn1(self.conv1(enc_3)))
        dec_1 = self.upsample1(dec_1)
        
        # Adjust dimensions using slicing for padding (similar to the Keras Lambda)
        dec_1 = dec_1[:, :, :-2, :-2]  # Slice to match dimensions (cropping)
        enc_2 = enc_2[:, :, :-1, :-1]  # Adjusting the dimensions of encoder 2
        dec_1 = F.pad(dec_1, (1, 1, 1, 1))  # Zero padding to match dimensions
        enc_2 = F.pad(enc_2, (1, 1, 1, 1))  # Zero padding for skip connection
        
        dec_1s = self.concat_skip(enc_2, dec_1, 256)

        # Decoder block 2
        dec_2 = F.relu(self.bn2(self.conv2(dec_1s)))
        dec_2 = self.upsample2(dec_2)

        dec_2s = F.relu(self.bn2(self.conv2(dec_2)))
        dec_2s = self.upsample2(dec_2s)
        
        # Adjusting the dimensions of encoder 1
        enc_1 = F.pad(enc_1, (2, 2, 2, 2))  # Zero padding to match dimensions

        dec_2s = self.concat_skip(enc_1, dec_2s, 128)

        # Decoder block 3
        dec_3 = F.relu(self.bn3(self.conv3(dec_2s)))
        dec_3s = F.relu(self.bn3(self.conv3(dec_3)))

        # Final output layer with sigmoid activation
        out = torch.sigmoid(self.out_conv(dec_3s))


# Model class to combine encoder and decoder
class SUIMNet(nn.Module):
    def __init__(self, base='RSB', n_classes=5):
        super(SUIMNet, self).__init__()
        self.base = base
        self.n_classes = n_classes

        if self.base == 'RSB':
            self.encoder = SuimEncoderRSB(channels=3)
            self.decoder = SuimDecoderRSB(n_classes)

        elif self.base == 'VGG':
            # 使用预训练的 VGG16 模型
            vgg = models.vgg16(pretrained=True)
            self.encoder = vgg.features  # VGG16的卷积层部分

            # 提取不同阶段的池化层
            self.pool1 = nn.Identity()  # 第1池化层将由forward方法获取
            self.pool2 = nn.Identity()  # 第2池化层将由forward方法获取
            self.pool3 = nn.Identity()  # 第3池化层将由forward方法获取
            self.pool4 = nn.Identity()  # 第4池化层将由forward方法获取

            # 定义解码器部分
            self.up1 = MyUpSample2X(512, 256)  # pool4 和 pool3
            self.up2 = MyUpSample2X(256, 128)  # dec1 和 pool2
            self.up3 = MyUpSample2X(128, 64)  # dec2 和 pool1
            self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            
            self.output_conv = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)

    def forward(self, x):
        if self.base == 'RSB':
            enc_outputs = self.encoder(x)
            out = self.decoder(enc_outputs)
        elif self.base == 'VGG':
            pool1 = self.encoder[:5](x)  # 经过block1的卷积和池化
            pool2 = self.encoder[5:10](pool1)  # 经过block2的卷积和池化
            pool3 = self.encoder[10:17](pool2)  # 经过block3的卷积和池化
            pool4 = self.encoder[17:24](pool3)  # 经过block4的卷积和池化
            
            dec1 = self.up1(pool4, pool3)
            dec2 = self.up2(dec1, pool2)
            dec3 = self.up3(dec2, pool1)
            dec4 = self.up4(dec3)  # 上采样到原始分辨率
            
            out = self.output_conv(dec4)
            
            return torch.sigmoid(out)  # 使用 sigmoid 作为激活函数（适用于二元分割）

if __name__ == "__main__":
    model = SUIMNet(base='VGG', n_classes=5)
    print(model)