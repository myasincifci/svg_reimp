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

    def forward(self, x):
        # initialize the hidden state.
        self.lstm.hidden = self.lstm.init_hidden()

        x = x.permute((1,0,2,3))
        T, B, H, W = x.size()

        h_seq = [self.encoder(x[t][:,None]) for t in range(T)]

        x_preds_past = []
        x_preds_future = []
        h_pred = None
        for i in range(1, self.cfg.n_past+self.cfg.n_future):
            if i <= self.cfg.n_past:	
                h, skip = h_seq[i-1]
            else:
                h = h_pred # h_seq[i-1][0]

            if self.cfg.vf_skip:
                s = h if i == 1 else h_pred
                h_pred = self.lstm(h) + s
            else:
                h_pred = self.lstm(h)

            # predictions [x_1_hat, ..., x_T_hat], T = n_past + n_future - 1
            if i < self.cfg.n_past:
                x_preds_past.append(self.decoder([h_pred, skip]))
            else:
                x_preds_future.append(self.decoder([h_pred, skip]))

        return torch.stack(x_preds_past, dim=0).permute((1,0,2,3,4)), torch.stack(x_preds_future, dim=0).permute((1,0,2,3,4))

    def training_step(self, batch, batch_idx):
        x_preds_past, x_preds_future = self(batch) 
        loss = F.mse_loss(x_preds_future.squeeze(), batch[:,self.cfg.n_past:], reduction='none').mean(dim=(0,2,3)).sum()
        self.log('train/loss', loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x_preds_past, x_preds_future = self(batch) 
        loss = F.mse_loss(x_preds_future.squeeze(), batch[:,self.cfg.n_past:], reduction='none').mean(dim=(0,2,3)).sum()

        self.log('val/loss', loss, prog_bar=True)

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