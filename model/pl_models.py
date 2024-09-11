
import torch
import pytorch_lightning as pl
from .loss.con_loss import SimCLRLoss
from .metric.metrics import accuracy


# lightening model
class ExtractorModel(pl.LightningModule):
    def __init__(self, model, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = cfg.lr
        self.wd = cfg.wd
        self.max_epochs = cfg.max_epochs
        self.restart_times = cfg.restart_times
        self.criterion = SimCLRLoss(cfg.loss_temp)
        self.metric = accuracy
    
    def forward(self, x):
        self.model.set_saveFea(True)
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        # scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.max_epochs, gamma=0.8, last_epoch=-1, verbose=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.max_epochs // self.restart_times, eta_min=0,last_epoch=-1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    # remain to be implemented
    def training_step(self, batch, batch_idx):
        data, labels = batch
        self.model.set_saveFea(False)
        proj = self.model(data)
        # self.criterion.to(data.device)   # put it in the loss function
        loss, logits, logits_labels = self.criterion(proj)
        top1, top5 = self.metric(logits, logits_labels, topk=(1,5))
        self.log_dict({'ext/train/loss': loss, 'ext/train/acc': top1[0], 'ext/train/acc5': top5[0], 'ext/train/lr': self.optimizers().param_groups[-1]['lr']}, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, labels = batch
        self.model.set_saveFea(False)
        proj = self.model(data)
        # self.criterion.to(data.device)    # put it in the loss function
        loss, logits, logits_labels = self.criterion(proj)
        top1, top5 = self.metric(logits, logits_labels, topk=(1,5))
        self.log_dict({'ext/val/loss': loss, 'ext/val/acc': top1[0], 'ext/val/acc5': top5[0]}, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        data, labels = batch
        fea = self(data)
        return fea

class MLPModel(pl.LightningModule):
    def __init__(self, model, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = cfg.lr
        self.wd = cfg.wd
        self.criterion = torch.nn.CrossEntropyLoss()
        self.metric = accuracy
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
    
    def training_step(self, batch, batch_idx):
        data, labels = batch
        logits = self.model(data)
        loss = self.criterion(logits, labels)
        top1= self.metric(logits, labels, topk=(1,))
        self.log_dict({'mlp/train/loss': loss, 'mlp/train/acc': top1[0]}, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        logits = self.model(data)
        loss = self.criterion(logits, labels)
        top1 = self.metric(logits, labels, topk=(1,))
        self.log_dict({'mlp/val/loss': loss, 'mlp/val/acc': top1[0]}, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        data, labels = batch
        logits = self(data)
        return logits.argmax(dim=1)
        
    
if __name__ == '__main__':
    pass