import torch
import torch.nn as nn
import torch.nn.functional as F

from params import *


class WaveGANGenerator(nn.Module):
    def __init__(
        self,
        noise_dim=NOISE_DIM,
        upsample=True,
        dim_mul=32,
        model_size=64,
        slice_len=SLICE_LEN,
        use_batch_norm=False,
        use_style_gan=False
    ):
        super(WaveGANGenerator, self).__init__()
        assert slice_len in [65536] #leave as array for possibility of different lengths

        self.use_batch_norm = use_batch_norm
        self.dim_mul = dim_mul
        self.slice_len = slice_len
        self.model_size = model_size,
        print("Model size: ", self.model_size)

        #[100] -> [16, 1024]
        # self.model_size = 64
        self.fc1 = nn.Linear(noise_dim, 4 * 4 * self.model_size * self.dim_mul)

        self.bn1 = nn.BatchNorm1d(num_features=self.model_size * self.dim_mul)

        stride = 4
        if upsample:
            stride = 1
            upsample = 4

        tconv_layers = [
            Transpose1dLayer(
                self.dim_mul * self.model_size,
                (self.dim_mul * self.model_size) // 2,
                25,
                stride,
                upsample=upsample,
                use_batch_norm=use_batch_norm,
                use_style_gan=use_style_gan
            ),
            Transpose1dLayer(
                (self.dim_mul * self.model_size) // 2,
                (self.dim_mul * self.model_size) // 4,
                25,
                stride,
                upsample=upsample,
                use_batch_norm=use_batch_norm,
                use_style_gan=use_style_gan
            ),
            Transpose1dLayer(
                (self.dim_mul * self.model_size) // 4,
                (self.dim_mul * self.model_size) // 8,
                25,
                stride,
                upsample=upsample,
                use_batch_norm=use_batch_norm,
                use_style_gan=use_style_gan
            ),
            Transpose1dLayer(
                (self.dim_mul * self.model_size) // 8,
                (self.dim_mul * self.model_size) // 16,
                25,
                stride,
                upsample=upsample,
                use_batch_norm=use_batch_norm,
                use_style_gan=use_style_gan
            ),
            Transpose1dLayer(
                (self.dim_mul * self.model_size) // 16,
                self.model_size,
                25,
                stride,
                upsample=upsample,
                use_style_gan=use_style_gan
            ),
            Transpose1dLayer(
                self.model_size, 1, 25, stride, upsample=upsample,
                use_style_gan=use_style_gan
            )
        ]

        self.tconv_list = nn.ModuleList(tconv_layers)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data) #TODO: play around with different initializations
        
    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, self.dim_mul * self.model_size, 16)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        for tconv in self.tconv_list[:-1]:
            x = F.relu(tconv(x))
        x = torch.tanh(self.tconv_list[-1](x))
        return x


class Transpose1dLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=11,
        upsample=None,
        output_padding=1,
        use_batch_norm=False,
        use_style_gan=False
    ):
        super(Transpose1dLayer, self).__init__()
        self.upsample = upsample

        sequence = []
        if upsample:
            reflection_pad = nn.ConstantPad1d(kernel_size//2, value=0)
            conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
            conv1d.weight.data.normal_(0.0, 0.02)
            sequence = [reflection_pad, conv1d]
        else:
            Conv1dTrans = nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride, padding, output_padding
            )
            sequence = [Conv1dTrans]
        
        if use_style_gan:
            sequence.append(AdaIN(out_channels, NOISE_DIM))
        if use_batch_norm:
            batch_norm = nn.BatchNorm1d(out_channels)
            sequence.append(batch_norm)
        
        self.layer = nn.Sequential(*sequence)

    def forward(self, x):
        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=self.upsample, mode="nearest")
        return self.layer(x)

class MappingLayers(nn.Module):
    def __init__(self, z_dim, hidden_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear((z_dim), (hidden_dim)),
            nn.ReLU(),
            nn.Linear((hidden_dim), (hidden_dim)),
            nn.ReLU(),
            nn.Linear((hidden_dim), (w_dim))
        )

    def forward(self, noise):
        return self.mapping(noise)

class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm1d(channels)

        self.style_scale_transform = nn.Linear(w_dim, channels)
        self.style_shift_transform = nn.Linear(w_dim, channels)

    def forward(self, image, w):
        normalized_image = self.instance_norm(image)
        style_scale = self.style_scale_transform(w)[:, None, None]
        style_shift = self.style_shift_transform(w)[:, None, None]
        
        transformed_image = style_scale * normalized_image + style_shift
        return transformed_image