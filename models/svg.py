import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .modules.dcgan_64 import encoder, decoder
from .modules.lstm import lstm

class SVG_Deterministic(pl.LightningModule):
    def __init__(self, cfg):
        super(SVG_Deterministic, self).__init__()
        self.encoder = encoder(128)
        self.lstm = lstm(128, 128, 256, 2, cfg.param.batch_size)
        self.decoder = decoder(128, skip=cfg.skip)
        self.cfg = cfg

    # def forward(self, x):
        
        
    #     return mse

    def training_step(self, batch, batch_idx):
        x = batch
        
        # initialize the hidden state.
        self.lstm.hidden = self.lstm.init_hidden()

        x = x.permute((1,0,2,3))
        T, B, H, W = x.size()

        h_seq = [self.encoder(x[t][:,None]) for t in range(T)]

        mse = 0
        for i in range(1, self.cfg.n_past+self.cfg.n_future):
            if i <= self.cfg.n_past:	
                h, skip = h_seq[i-1]
            else:
                h = h_pred # h_seq[i-1][0]
            s = h if i == 1 else h_pred

            if self.cfg.vf_skip:
                h_pred = self.lstm(h) + s
            else:
                h_pred = self.lstm(h)

            if i >= self.cfg.n_past:
                x_pred = self.decoder([h_pred, skip])
                mse += F.mse_loss(x_pred.squeeze(), x[i])
        
        self.log('train/loss', mse, prog_bar=True)
        
        return mse

    def validation_step(self, batch, batch_idx):
        x = batch
        
        # initialize the hidden state.
        self.lstm.hidden = self.lstm.init_hidden()

        x = x.permute((1,0,2,3))
        T, B, H, W = x.size()

        h_seq = [self.encoder(x[t][:,None]) for t in range(T)]

        mse = 0
        for i in range(1, self.cfg.n_past+self.cfg.n_future):
            if i <= self.cfg.n_past:	
                h, skip = h_seq[i-1]
            else:
                h = h_pred # h_seq[i-1][0]
            s = h if i == 1 else h_pred

            if self.cfg.vf_skip:
                h_pred = self.lstm(h) + s
            else:
                h_pred = self.lstm(h)

            if i >= self.cfg.n_past:
                x_pred = self.decoder([h_pred, skip])
                mse += F.mse_loss(x_pred.squeeze(), x[i])

        self.log('val/loss', mse, prog_bar=True)

    # def on_validation_epoch_end(self):
    #     sample = self.trainer.datamodule.val_dataloader().dataset[0]
    #     x = sample.unsqueeze(0)
        
    #     # initialize the hidden state.
    #     self.lstm.hidden = self.lstm.init_hidden()

    #     x = x.permute((1,0,2,3))
    #     T, B, H, W = x.size()

    #     h_seq = [self.encoder(x[t][:,None]) for t in range(T)]

    #     predictions = []
    #     for i in range(1, self.cfg.n_past+self.cfg.n_future):
    #         if i <= self.cfg.n_past:	
    #             h, skip = h_seq[i-1]
    #             s = h
    #         else:
    #             h = h_pred # h_seq[i-1][0]

    #         if self.cfg.vf_skip:
    #             h_pred = self.lstm(h) + s
    #         else:
    #             h_pred = self.lstm(h)

    #         if i >= self.cfg.n_past:
    #             x_pred = self.decoder([h_pred, skip])
    #             predictions.append(x_pred.squeeze().cpu())

    #     predictions = torch.stack(predictions, dim=0)
    #     self.logger.experiment.add_images('val/sample_predictions', predictions, self.current_epoch)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.param.lr)

        return optimizer