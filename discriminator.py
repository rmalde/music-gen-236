import torch
import torch.nn as nn
import torch.nn.functional as F

from params import *

class WaveGANDiscriminator(nn.Module):
    '''Discriminator with structure similar to WaveGAN paper'''
    def __init__(
        self,
        model_size=64,
        shift_factor=2,
        alpha=0.2,
        dim_mul=32,
        slice_len=SLICE_LEN,
        use_batch_norm=False,
        spectral_norm=False,
    ):
        super(WaveGANDiscriminator, self).__init__()
        assert slice_len in [65536] #leave as array for possibility of different lengths


        self.model_size = model_size
        self.use_batch_norm = use_batch_norm
        self.dim_mul = dim_mul
        self.slice_len = slice_len

        conv_layers = [
             Conv1d(
                1,
                model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor,
            ),
            Conv1d(
                model_size,
                2 * model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor,
            ),
            Conv1d(
                2 * model_size,
                4 * model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor,
            ),
            Conv1d(
                4 * model_size,
                8 * model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor,
            ),
            Conv1d(
                8 * model_size,
                16 * model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=shift_factor,
            ),
            Conv1d(
                16 * model_size,
                32 * model_size,
                25,
                stride=4,
                padding=11,
                use_batch_norm=use_batch_norm,
                alpha=alpha,
                shift_factor=0,
            )
        ]
        self.fc_input_size = self.slice_len

        self.conv_layers = nn.ModuleList(conv_layers)

        self.fc1 = nn.Linear(self.fc_input_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
        x =  x.view(-1, self.fc_input_size)
        if self.spectral_norm:
            x = nn.utils.spectral_norm(x)

        return self.fc1(x)


class Conv1d(nn.Module):
    '''1d Convolutions for WaveGAN Discriminator'''
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        alpha=0.2,
        shift_factor=2,
        stride=4,
        padding=11,
        use_batch_norm=False,
        drop_prob=0,
    ):
        super(Conv1d, self).__init__()

        self.phase_shuffle = PhaseShuffle(shift_factor)

        sequence = []

        conv1d = nn.Conv1d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        sequence.append(conv1d)

        if use_batch_norm:
            sequence.append(nn.BatchNorm1d(output_channels))

        sequence.append(nn.LeakyReLU(negative_slope=alpha))
        if shift_factor > 0:
            sequence.append(self.phase_shuffle)
        
        if drop_prob > 0:
            sequence.append(nn.Dropout2d(drop_prob))

        self.layer = nn.Sequential(*sequence)

    def forward(self, x):
        return self.layer(x)


class PhaseShuffle(nn.Module):
    '''
    Phase Shuffle module, similar to implementation in WaveGAN paper.
    Shifts the inputs at each layer slightly along the time axis so that
    the discriminator does not learn to distinguish the exact phase of
    real vs generated. 
    '''
    def __init__(self, shift_factor):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor
    
    def forward(self, x):
        if self.shift_factor==0:
            return x

        k_list = (
            torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1)
            - self.shift_factor
        )
        k_list = k_list.numpy().astype(int)

        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)

        x_shuffle = x.clone()

        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode="reflect")
            else:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode="reflect")

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape, x.shape)
        return x_shuffle

class TransGANDiscriminator(nn.Module):
    '''Transformer GAN Discriminator, similar architecture to TransGAN paper, adapted to audio'''
    def __init__(
        self,
        noise_dim=NOISE_DIM,
        upsample=True,
        dim_mul=32,
        model_size=64,
        slice_len=SLICE_LEN,
        use_batch_norm=True, 
    ):
        super().__init__()
        self.model_size = model_size
        self.dim_mul = dim_mul
        self.use_batch_norm = use_batch_norm
        self.slice_len = slice_len

        transformer_layers = [
            Transformer1dLayer(
                1,
                self.model_size
            ),
            Transformer1dLayer(
                self.model_size,
                self.model_size * 2
            ),
            Transformer1dLayer(
                self.model_size * 2,
                self.model_size * 4
            ),
            Transformer1dLayer(
                self.model_size * 4,
                self.model_size * 8
            ),
            Transformer1dLayer(
                self.model_size * 8,
                self.model_size * 16
            ),
            Transformer1dLayer(
                self.model_size * 16,
                self.model_size * 32
            )
        ]
        self.transformer_list = nn.ModuleList(transformer_layers)

        self.fc1 = nn.Linear(self.slice_len, 1)

        self.bn1 = nn.BatchNorm1d(num_features=self.model_size * self.dim_mul)

    def forward(self, x):

        for conv in self.conv_layers:
            x = conv(x)
        x =  x.view(-1, self.fc_input_size)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.fc1(x)
        return x

class Transformer1dLayer(nn.Module):
    '''Single layer for TransGAN'''
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=25,
        use_batch_norm=True,
        n_head=8,
        alpha=0.2,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.use_batch_norm = use_batch_norm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_head = n_head
        self.alpha = alpha

        encoder_layer = nn.TransformerEncoderLayer(self.in_channels, self.n_head)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, x):
        x = self.transformer(x)
        
        if self.use_batch_norm:
            batch_norm = nn.BatchNorm1d(self.out_channels)
            x = batch_norm(x)

        x = nn.LeakyReLU(negative_slope=self.alpha)(x)
        B, H, W = x.shape
        x = x.view(B, -1, self.out_channels)
        return x


        

        