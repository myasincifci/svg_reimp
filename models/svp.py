import torch
from torch import nn

from .modules.autoencoder import ConvEncoder, ConvDecoder
from .modules.mylstm import LSTM
import pytorch_lightning as pl

from torchvision.utils import make_grid

class SVP(pl.LightningModule):
    def __init__(self, cfg):
        super(SVP, self).__init__()
        self.encoder = ConvEncoder(tuple(cfg.data.shape), 32, 128, act_fn=nn.GELU, variational=False)
        self.lstm = LSTM(128, cfg.lstm.hidden_dim, cfg.lstm.layers, skip=cfg.vf_skip)
        self.decoder = ConvDecoder(tuple(cfg.data.shape), 32, 128, act_fn=nn.GELU)

        self.cfg = cfg

    def forward(self, x):
        B, T, H, W = x.size()

        z = self.encoder(x.view(B*T, 1, H, W)).view(B, T, -1)
        z_ = self.lstm(z[:, :self.cfg.n_past], future=self.cfg.n_future)
        x_ = self.decoder(z_.reshape(B*T, -1)).reshape(B, T, H, W)

        preds_past = x_[:, :self.cfg.n_past-1]
        preds_future = x_[:, self.cfg.n_past-1:-1]

        return preds_past, preds_future
    
    def training_step(self, batch, batch_idx):
        x = batch
        preds_past, preds_future = self(x)
        
        loss_pst = nn.functional.mse_loss(preds_past, x[:,1:self.cfg.n_past], reduction='none').mean(dim=(0,2,3)).sum()
        loss_fut = nn.functional.mse_loss(preds_future, x[:,self.cfg.n_past:], reduction='none').mean(dim=(0,2,3)).sum()

        self.log('train/loss', loss_pst, prog_bar=True)
        self.log('train/loss_past', loss_fut)

        return loss_pst + loss_fut
    
    def validation_step(self, batch, batch_idx):
        x = batch
        preds_past, preds_future = self(x)
        
        loss_pst = nn.functional.mse_loss(preds_past, x[:,1:self.cfg.n_past], reduction='none').mean(dim=(0,2,3)).sum()
        loss_fut = nn.functional.mse_loss(preds_future, x[:,self.cfg.n_past:], reduction='none').mean(dim=(0,2,3)).sum()

        self.log('val/loss', loss_pst, prog_bar=True)
        self.log('val/loss_past', loss_fut)

    def on_validation_epoch_end(self):
        sample = torch.from_numpy(self.trainer.datamodule.val_dataloader().dataset[0]).to(self.device)
        x = sample.unsqueeze(0)
        x = x.repeat(100, 1, 1, 1) # TODO: hack, remove later
        
        x_preds_past, x_preds_future = self(x)    

        self.logger.log_image('val/sample_predictions', [make_grid(x_preds_future[0][:,None], nrow=10)], self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.param.lr)

        return optimizer