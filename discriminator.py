import torch
import torch.nn as nn
import torch.nn.functional as F

from params import *

class WaveGANDiscriminator(nn.Module):
    def __init__(
        self,
        model_size=64,
        shift_factor=2,
        alpha=0.2,
        dim_mul=32,
        slice_len=SLICE_LEN,
        use_batch_norm=False,
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
        # x = nn.utils.spectral_norm(x)

        return self.fc1(x)





class Conv1d(nn.Module):
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
        sequence = []

        conv1d = nn.Conv1d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        sequence.append(conv1d)

        if use_batch_norm:
            sequence.append(nn.BatchNorm1d(output_channels))

        sequence.append(nn.LeakyReLU(negative_slope=alpha))
        if shift_factor == 0:
            # PhaseShuffle(shift_factor)
            print("Phase shuffle not implemented yet")
        
        if drop_prob > 0:
            sequence.append(nn.Dropout2d(drop_prob))

        self.layer = nn.Sequential(*sequence)

    def forward(self, x):
        return self.layer(x)