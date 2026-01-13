import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

from torchvision.utils import make_grid


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Stride 2 convolution reduces resolution by half
        self.sp_conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.t_conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0),
            padding_mode='replicate'
        )
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.sp_conv(x)
        x = self.act(x)
        x = self.t_conv(x)

        return x
    
class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Stride 2 convolution reduces resolution by half
        self.sp_conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
        )
        self.t_conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1),
            padding=(1, 0, 0),
            padding_mode='replicate'
        )
        self.act = nn.SiLU()

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=(1, 2, 2), mode="nearest", align_corners=False
        )
        x = self.sp_conv(x)
        x = self.act(x)
        x = F.interpolate(
            x, scale_factor=(2, 1, 1), mode="nearest", align_corners=False            
        )
        x = self.t_conv(x)

        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        stride=1,
        padding=1,
        dropout=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(
            num_groups=16, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.sp_conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size, kernel_size),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
        )
        self.t_conv1 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1, 1),
            stride=(stride, 1, 1),
            padding=(padding, 0, 0),
        )

        self.norm2 = nn.GroupNorm(
            num_groups=16, num_channels=out_channels, eps=1e-6, affine=True
        )
        self.sp_conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(1, kernel_size, kernel_size),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
        )
        self.t_conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1, 1),
            stride=(stride, 1, 1),
            padding=(padding, 0, 0),
        )

        # If input and output channels differ, we need a 1x1 convolution
        # to match dimensions for the addition.
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = x

        # First path
        h = self.norm1(h)
        h = self.act(h)
        h = self.sp_conv1(h)
        h = self.act(h)
        h = self.t_conv1(h)

        # Second path
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.sp_conv2(h)
        h = self.act(h)
        h = self.t_conv2(h)

        # Add residual connection
        return h + self.shortcut(x)


class TAEEncoder(nn.Module):
    def __init__(self, cfg):
        super(TAEEncoder, self).__init__()
        self.in_conv = nn.Conv3d(
            in_channels=1,
            out_channels=16,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=0,
        )

        self.block1 = ResnetBlock(in_channels=16, out_channels=16)
        self.down1 = Downsample(in_channels=16)

        self.block2 = ResnetBlock(in_channels=16, out_channels=32)
        self.down2 = Downsample(in_channels=32)

        self.block3 = ResnetBlock(in_channels=32, out_channels=64)
        self.down3 = Downsample(in_channels=64)

        self.block4 = ResnetBlock(in_channels=64, out_channels=64)
        self.up1 = Upsample(in_channels=64)

        self.block5 = ResnetBlock(in_channels=64, out_channels=32)
        self.up2 = Upsample(in_channels=32)

        self.block4 = ResnetBlock(in_channels=32, out_channels=16)
        self.up3 = Upsample(in_channels=16)

        self.out_conv = nn.Conv3d(
            in_channels=16,
            out_channels=1,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=0,
        )

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1) # Add channel dimension

        N, C, T, H, W = x.size()

        x = self.in_conv(x)

        x = self.block1(x)
        x = self.down1(x)

        x = self.block2(x)
        x = self.down2(x)

        x = self.block3(x)
        x = self.down3(x)

        x = self.block4(x)
        x = self.up1(x)

        x = self.block5(x)
        x = self.up2(x)

        x = self.block4(x)
        x = self.up3(x)

        return x


class TAE(nn.Module):
    def __init__(self, cfg):
        super(TAE, self).__init__()
        self.encoder = TAEEncoder(cfg)

    def forward(self, x):
        return self.encoder(x)


class TAEModule(pl.LightningModule):
    def __init__(self, cfg):
        super(TAEModule, self).__init__()

        self.model = TAE(cfg)

        self.cfg = cfg

    def forward(self, x):
        B, T, H, W = x.size()

        self.model(x)

        return x

    def training_step(self, batch, batch_idx):
        x = batch

        y = self(x)

        return y

    def validation_step(self, batch, batch_idx):
        x = batch

    def on_validation_epoch_end(self):
        if self.cfg.logging:
            sample = torch.from_numpy(
                self.trainer.datamodule.val_dataloader().dataset[0]
            ).to(self.device)
            x = sample.unsqueeze(0)
            x = x.repeat(100, 1, 1, 1)  # TODO: hack, remove later

            x_preds_past, x_preds_future, _, _ = self(x)

            self.logger.log_image(
                "val/sample_predictions",
                [make_grid(x_preds_future[0][:, None], nrow=10)],
                self.current_epoch,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.param.lr)

        return optimizer
