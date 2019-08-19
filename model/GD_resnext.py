import torch
import torch.nn as nn
import torch.nn.functional as F

from .se_resnet import BottleneckX, SEResNeXt

class MultiPrmSequential(nn.Sequential):
    def __init__(self, *args):
        super(MultiPrmSequential, self).__init__(*args)

    def forward(self, input, cat_feature):
        for module in self._modules.values():
            input = module(input, cat_feature)
        return input

def make_secat_layer(block, inplanes, planes, cat_planes, block_count, stride=1):
    outplanes = planes * block.expansion
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, cat_planes, 16, stride, downsample))
    for i in range(1, block_count):
        layers.append(block(outplanes, planes, cat_planes, 16))

    return MultiPrmSequential(*layers)

class SECatBottleneckX(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cat_channel, cardinality=16, stride=1, downsample=None):
        super(SECatBottleneckX, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        # self.selayer = SeLayer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, cat_feature):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out = self.selayer(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FeatureConv(nn.Module):
    def __init__(self, input_dim=512, output_dim=256, input_size=32, output_size=16):
        super(FeatureConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.output_size = output_size
        
        if input_size == output_size * 4:
            stride1, stride2 = 2, 2
        elif input_size == output_size * 2:
            stride1, stride2 = 2, 1
        else:
            stride1, stride2 = 1, 1
        
        self.network = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride1, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=stride2, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.network(x)

class DecoderBlock(nn.Module):
    def __init__(self, inplanes, planes, color_fc_out, block_num):
        super(DecoderBlock, self).__init__()
        self.secat_layer = make_secat_layer(SECatBottleneckX, inplanes, planes//4, color_fc_out, block_num)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x, cat_feature):
        out = self.secat_layer(x, cat_feature)
        return self.ps(out)
        

class Generator(nn.Module):
    def __init__(self, input_size, cv_class_num, iv_class_num, input_dim=1, output_dim=3, 
                 layers=[12, 8, 5, 5], net_opt=None):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.cv_class_num = cv_class_num
        self.iv_class_num = iv_class_num
        self.cardinality = 16
        self.Linear = nn.Linear(cv_class_num, 16*16*32)

        self.color_fc_out = 64
        self.colorFC = nn.Sequential(
            nn.Linear(cv_class_num, self.color_fc_out),
            nn.Linear(self.color_fc_out, self.color_fc_out),
            nn.Linear(self.color_fc_out, self.color_fc_out),
            nn.Linear(self.color_fc_out, self.color_fc_out)
        )

        self.conv1 = self._make_encoder_block_first(self.input_dim, 16)
        self.conv2 = self._make_encoder_block(16, 32)
        self.conv3 = self._make_encoder_block(32, 64)
        self.conv4 = self._make_encoder_block(64, 128)
        self.conv5 = self._make_encoder_block(128, 256)
        
        self.deconv1 = DecoderBlock(256 + 256 + 64, 4*256, self.color_fc_out, layers[0])
        self.deconv2 = DecoderBlock(256 + 128, 4*128, self.color_fc_out, layers[1])
        self.deconv3 = DecoderBlock(128 + 64, 4*64, self.color_fc_out, layers[2])
        self.deconv4 = DecoderBlock(64 + 32, 4*32, self.color_fc_out, layers[3])
        self.deconv5 = nn.Sequential(
            nn.Conv2d(32 + 16, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )

        self.featureConv = FeatureConv()

        self.colorConv = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Tanh(),
        )

        self.deconv_for_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # output is 64 * 64
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # output is 128 * 128
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # output is 256 * 256
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1, output_padding=0), # output is 256 * 256
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _make_encoder_block(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def _make_encoder_block_first(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, input, feature_tensor, c_tag_class):
        out1 = self.conv1(input)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)

        # ==============================
        # it's about color variant tag set
        # temporally, don't think about noise z
        c_tag_tensor = self.Linear(c_tag_class)
        c_tag_tensor = c_tag_tensor.view(-1, 32, 16, 16)
        c_tag_tensor = self.colorConv(c_tag_tensor)

        c_se_tensor = self.colorFC(c_tag_class)

        # ==============================
        # Convolution Layer for Feature Tensor

        feature_tensor = self.featureConv(feature_tensor)
        concat_tensor = torch.cat([out5, feature_tensor, c_tag_tensor], 1)
        # concat_tensor = torch.cat([concat_tensor, c_tag_tensor], 1)
        out4_prime = self.deconv1(concat_tensor, c_se_tensor)

        # ==============================
        # out4_prime should be input of Guide Decoder
        
        decoder_output = self.deconv_for_decoder(out4_prime)

        # ==============================

        concat_tensor = torch.cat([out4_prime, out4], 1)
        out3_prime = self.deconv2(concat_tensor, c_se_tensor)

        concat_tensor = torch.cat([out3_prime, out3], 1)
        out2_prime = self.deconv3(concat_tensor, c_se_tensor)

        concat_tensor = torch.cat([out2_prime, out2], 1)
        out1_prime = self.deconv4(concat_tensor, c_se_tensor)

        concat_tensor = torch.cat([out1_prime, out1], 1)
        full_output = self.deconv5(concat_tensor)

        return full_output, decoder_output
