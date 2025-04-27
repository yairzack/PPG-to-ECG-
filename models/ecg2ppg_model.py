# models/ecg2ppg_model.py

import torch
import torch.nn as nn
import numpy as np

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleBlock, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=15, padding='same')
        self.bn = nn.BatchNorm1d(out_channels)
        self.activ = nn.LeakyReLU()

    def forward(self, x):
        prev_x = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        x = self.pool(x)
        return x, prev_x

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpSampleBlock, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv1d(in_channels + skip_channels, out_channels, kernel_size=15, padding='same')
        self.bn = nn.BatchNorm1d(out_channels)
        self.activ = nn.LeakyReLU()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x

class ECG2PPGps(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Left encoder
        self.conv1 = nn.Conv1d(in_channels, 24, kernel_size=15, padding='same')
        self.bn1 = nn.BatchNorm1d(24)
        self.activ1 = nn.LeakyReLU()

        self.down1 = DownSampleBlock(24, 48)
        self.down2 = DownSampleBlock(48, 72)
        self.down3 = DownSampleBlock(72, 96)
        self.down4 = DownSampleBlock(96, 120)
        self.down5 = DownSampleBlock(120, 144)
        self.down6 = DownSampleBlock(144, 168)
        self.down7 = DownSampleBlock(168, 192)
        self.down8 = DownSampleBlock(192, 192)

        # Left decoder
        self.up1 = UpSampleBlock(192, 192, 168)
        self.up2 = UpSampleBlock(168, 168, 192)
        self.up3 = UpSampleBlock(192, 144, 144)
        self.up4 = UpSampleBlock(144, 120, 120)
        self.up5 = UpSampleBlock(120, 96, 96)
        self.up6 = UpSampleBlock(96, 72, 72)
        self.up7 = UpSampleBlock(72, 48, 48)
        self.up8 = UpSampleBlock(48, 24, 24)

        # Transition
        self.conv2 = nn.Conv1d(24, 24, kernel_size=15, padding='same')
        self.bn2 = nn.BatchNorm1d(24)
        self.activ2 = nn.LeakyReLU()

        # Right encoder
        self.down9 = DownSampleBlock(24, 48)
        self.down10 = DownSampleBlock(48, 72)
        self.down11 = DownSampleBlock(72, 96)
        self.down12 = DownSampleBlock(96, 120)
        self.down13 = DownSampleBlock(120, 144)
        self.down14 = DownSampleBlock(144, 168)
        self.down15 = DownSampleBlock(168, 192)
        self.down16 = DownSampleBlock(192, 192)

        # Right decoder
        self.up9 = UpSampleBlock(192, 192, 192)
        self.up10 = UpSampleBlock(192, 168, 168)
        self.up11 = UpSampleBlock(168, 144, 144)
        self.up12 = UpSampleBlock(144, 120, 120)
        self.up13 = UpSampleBlock(120, 96, 96)
        self.up14 = UpSampleBlock(96, 72, 72)
        self.up15 = UpSampleBlock(72, 48, 48)
        self.up16 = UpSampleBlock(48, 24, 24)

        # Final conv
        self.conv3 = nn.Conv1d(24, out_channels, kernel_size=15, padding='same')
        self.activ3 = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activ1(x)
        x, prev_x1L = self.down1(x)
        x, prev_x2L = self.down2(x)
        x, prev_x3L = self.down3(x)
        x, prev_x4L = self.down4(x)
        x, prev_x5L = self.down5(x)
        x, prev_x6L = self.down6(x)
        x, prev_x7L = self.down7(x)
        x, prev_x8L = self.down8(x)

        x = self.up1(x, prev_x8L)
        x = self.up2(x, prev_x7L)
        x = self.up3(x, prev_x6L)
        x = self.up4(x, prev_x5L)
        x = self.up5(x, prev_x4L)
        x = self.up6(x, prev_x3L)
        x = self.up7(x, prev_x2L)
        x = self.up8(x, prev_x1L)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activ2(x)

        x, prev_x1R = self.down9(x)
        x, prev_x2R = self.down10(x)
        x, prev_x3R = self.down11(x)
        x, prev_x4R = self.down12(x)
        x, prev_x5R = self.down13(x)
        x, prev_x6R = self.down14(x)
        x, prev_x7R = self.down15(x)
        x, prev_x8R = self.down16(x)

        x = self.up9(x, prev_x8R)
        x = self.up10(x, prev_x7R)
        x = self.up11(x, prev_x6R)
        x = self.up12(x, prev_x5R)
        x = self.up13(x, prev_x4R)
        x = self.up14(x, prev_x3R)
        x = self.up15(x, prev_x2R)
        x = self.up16(x, prev_x1R)

        x = self.conv3(x)
        x = self.activ3(x)
        return x

def create_windows(data, window_length, stride):
    windows = []
    for i in range(window_length, len(data), stride):
        windows.append(data[i - window_length:i])
    return np.array(windows)

def ECG2PPGps_loss(y_pred, y_true):
    mse_loss = nn.MSELoss()(y_pred, y_true)
    mal_loss = torch.max(torch.abs(y_pred - y_true))
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    r_matrix = torch.corrcoef(torch.stack([y_true_flat, y_pred_flat]))
    r = r_matrix[0, 1]
    r_loss = 1 - torch.abs(r)
    total_loss = mse_loss + mal_loss + r_loss
    return total_loss
