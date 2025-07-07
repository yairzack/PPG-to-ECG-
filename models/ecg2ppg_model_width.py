import torch
import torch.nn as nn
import numpy as np
from models.customLosses import RelMSE, RelMAL, CrossCorrelationShiftLoss, SpectralMSELoss

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super(DownSampleBlock, self).__init__()
        layers = []
        for i in range(num_convs):
            conv_in_channels = in_channels if i == 0 else out_channels
            layers.append(nn.Conv1d(conv_in_channels, out_channels, kernel_size=15, padding='same'))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.LeakyReLU())
        self.convs = nn.Sequential(*layers)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        prev_x = x
        x = self.convs(x)
        x = self.pool(x)
        return x, prev_x

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, num_convs):
        super(UpSampleBlock, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=2, stride=2)
        layers = []
        layers.append(nn.Conv1d(in_channels + skip_channels, out_channels, kernel_size=15, padding='same'))
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.LeakyReLU())
        for _ in range(num_convs - 1):
            layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=15, padding='same'))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.LeakyReLU())
        self.convs = nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.convs(x)
        return x

class ECG2PPGps(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=24, depth=6, num_convs_per_level=1):
        super(ECG2PPGps, self).__init__()
        self.depth = depth

        # Left encoder
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15, padding='same'),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU()
        )
        self.down_blocks_left = nn.ModuleList()
        self.left_encoder_channels = [base_channels]

        in_ch = base_channels
        for i in range(depth):
            out_ch = base_channels + 24 * (i + 1)
            self.down_blocks_left.append(DownSampleBlock(in_ch, out_ch, num_convs=num_convs_per_level))
            self.left_encoder_channels.append(out_ch)
            in_ch = out_ch

        # Left decoder
        self.up_blocks_left = nn.ModuleList()
        for i in reversed(range(depth)):
            skip_ch = self.left_encoder_channels[i]
            in_ch = self.left_encoder_channels[i+1]
            out_ch = self.left_encoder_channels[i]
            self.up_blocks_left.append(UpSampleBlock(in_ch, skip_ch, out_ch, num_convs=num_convs_per_level))

        # Transition conv
        self.transition_conv = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=15, padding='same'),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU()
        )

        # Right encoder
        self.down_blocks_right = nn.ModuleList()
        self.right_encoder_channels = [base_channels]
        in_ch = base_channels
        for i in range(depth):
            out_ch = base_channels + 24 * (i + 1)
            self.down_blocks_right.append(DownSampleBlock(in_ch, out_ch, num_convs=num_convs_per_level))
            self.right_encoder_channels.append(out_ch)
            in_ch = out_ch

        # Right decoder
        self.up_blocks_right = nn.ModuleList()
        for i in reversed(range(depth)):
            skip_ch = self.right_encoder_channels[i]
            in_ch = self.right_encoder_channels[i+1]
            out_ch = self.right_encoder_channels[i]
            self.up_blocks_right.append(UpSampleBlock(in_ch, skip_ch, out_ch, num_convs=num_convs_per_level))

        # Final conv
        self.final_conv = nn.Sequential(
            nn.Conv1d(base_channels, out_channels, kernel_size=15, padding='same'),
            nn.LeakyReLU()
        )

    def forward(self, x):
        # Left tower
        x = self.initial_conv(x)
        skips_left = []
        for down in self.down_blocks_left:
            x, skip = down(x)
            skips_left.append(skip)
        for up, skip in zip(self.up_blocks_left, reversed(skips_left)):
            x = up(x, skip)
        x = self.transition_conv(x)

        # Right tower
        skips_right = []
        for down in self.down_blocks_right:
            x, skip = down(x)
            skips_right.append(skip)
        for up, skip in zip(self.up_blocks_right, reversed(skips_right)):
            x = up(x, skip)

        x = self.final_conv(x)
        return x

def create_windows(data, window_length, stride):
    windows = []
    for i in range(window_length, len(data), stride):
        windows.append(data[i - window_length:i])
    return np.array(windows)

def ECG2PPGps_loss(y_pred, y_true):
    mse_loss = nn.MSELoss()(y_pred, y_true)
    mal_loss = torch.max(torch.abs(y_pred - y_true))
    crossCorrCriterion = CrossCorrelationShiftLoss()
    crossCorr_loss = crossCorrCriterion(y_pred, y_true)
    spectralCriterion = SpectralMSELoss()
    spectral_loss = spectralCriterion(y_pred, y_true)

    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    r_matrix = torch.corrcoef(torch.stack([y_true_flat, y_pred_flat]))
    r = r_matrix[0, 1]
    r_loss = 1 - torch.abs(r)

    total_loss = mse_loss + mal_loss + r_loss
    return total_loss
