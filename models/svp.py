import torch
from torch import nn

from .modules.autoencoder import ConvEncoder, ConvDecoder
from .modules.mylstm import LSTM
import pytorch_lightning as pl

from torchvision.utils import make_grid

# TODO: work in progress
class VSVP(pl.LightningModule):
    def __init__(self, cfg):
        super(SVP, self).__init__()
        self.encoder = ConvEncoder(tuple(cfg.data.shape), cfg.autoencoder.c_hid, cfg.autoencoder.latent_dim, act_fn=nn.GELU, variational=True)
        self.lstm = LSTM(cfg.autoencoder.latent_dim, cfg.lstm.hidden_dim, cfg.lstm.layers, skip=cfg.vf_skip)
        self.decoder = ConvDecoder(tuple(cfg.data.shape), cfg.autoencoder.c_hid, cfg.autoencoder.latent_dim, act_fn=nn.GELU)

        self.cfg = cfg

class SVP(pl.LightningModule):
    def __init__(self, cfg):
        super(SVP, self).__init__()
        self.encoder = ConvEncoder(tuple(cfg.data.shape), cfg.autoencoder.c_hid, cfg.autoencoder.latent_dim, act_fn=nn.GELU, variational=False)
        self.lstm = LSTM(cfg.autoencoder.latent_dim, cfg.lstm.hidden_dim, cfg.lstm.layers, skip=cfg.vf_skip)
        self.decoder = ConvDecoder(tuple(cfg.data.shape), cfg.autoencoder.c_hid, cfg.autoencoder.latent_dim, act_fn=nn.GELU)

        self.cfg = cfg

    def forward(self, x):
        B, T, H, W = x.size()

        z = self.encoder(x.view(B*T, 1, H, W)).view(B, T, -1)
        z_ = self.lstm(z[:, :self.cfg.n_past], future=self.cfg.n_future)
        x_ = self.decoder(z_.reshape(B*T, -1)).reshape(B, T, H, W)
        x_rec = self.decoder(z.reshape(B*T, -1)).reshape(B, T, H, W)

        preds_past = x_[:, :self.cfg.n_past-1]
        preds_future = x_[:, self.cfg.n_past-1:-1] 

        return preds_past, preds_future, x_rec
    
    def training_step(self, batch, batch_idx):
        x = batch
        preds_past, preds_future, x_rec = self(x)
        
        loss_pst = nn.functional.mse_loss(preds_past, x[:,1:self.cfg.n_past], reduction='none').mean(dim=(0,2,3)).sum()
        loss_fut = nn.functional.mse_loss(preds_future, x[:,self.cfg.n_past:], reduction='none').mean(dim=(0,2,3)).sum()
        loss_rec = nn.functional.mse_loss(x_rec, x, reduction='none').mean(dim=(0,2,3)).sum()

        self.log('train/loss', loss_fut, prog_bar=True)
        self.log('train/loss_past', loss_pst)
        self.log('train/loss_rec', loss_rec)

        return loss_pst + loss_fut  + loss_rec
    
    def validation_step(self, batch, batch_idx):
        x = batch
        preds_past, preds_future, x_rec = self(x)
        
        loss_pst = nn.functional.mse_loss(preds_past, x[:,1:self.cfg.n_past], reduction='none').mean(dim=(0,2,3)).sum()
        loss_fut = nn.functional.mse_loss(preds_future, x[:,self.cfg.n_past:], reduction='none').mean(dim=(0,2,3)).sum()
        loss_rec = nn.functional.mse_loss(x_rec, x, reduction='none').mean(dim=(0,2,3)).sum()


        self.log('val/loss', loss_fut, prog_bar=True)
        self.log('val/loss_past', loss_pst)
        self.log('val/loss_rec', loss_rec)

    def on_validation_epoch_end(self):
        if self.cfg.logging:
            sample = torch.from_numpy(self.trainer.datamodule.val_dataloader().dataset[0]).to(self.device)
            x = sample.unsqueeze(0)
            x = x.repeat(100, 1, 1, 1) # TODO: hack, remove later
            
            x_preds_past, x_preds_future, _ = self(x)    

            self.logger.log_image('val/sample_predictions', [make_grid(x_preds_future[0][:,None], nrow=10)], self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.param.lr)

        return optimizer