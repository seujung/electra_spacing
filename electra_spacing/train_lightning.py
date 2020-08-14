import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from transformers import AdamW, get_linear_schedule_with_warmup
from electra_spacing.dataset import SpacingDataset
from electra_spacing.model import KoELECTRASpacingModel
from torchnlp.metrics import get_token_accuracy

class KoELECTRASpacing(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.model = KoELECTRASpacingModel()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        


    def forward(self, input_ids, token_type_ids):
        return self.model(input_ids, token_type_ids)

    def prepare_data(self):
        self.dataset = SpacingDataset(file_path=self.hparams.file_path)
        train_length = int(len(self.dataset) * self.hparams.train_ratio)

        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_length, len(self.dataset) - train_length],
        )
        
        self.hparams.total_steps = int(self.hparams.max_epochs * len(self.train_dataset) / self.hparams.batch_size)
        self.hparams.warmup_steps = int(self.hparams.total_steps * self.hparams.warmup_ratio)
    
    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size
        )
        return val_loader
    
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.1,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.total_steps
        )
        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
        )
    
    def training_step(self, batch, batch_idx):
        self.model.train()

        input_ids, token_type_ids, labels, labels_weight = batch
  
        pred_tokens = self.forward(input_ids, token_type_ids)
        loss_tokens = self.loss_fn(pred_tokens.transpose(1, 2), labels)

        loss_space_tokens = self.loss_fn(pred_tokens.transpose(1, 2), labels * labels_weight)
        loss_space_tokens = loss_space_tokens * self.hparams.weight_value
    
        total_loss = loss_tokens + loss_space_tokens

        token_acc = get_token_accuracy(
            labels.cpu(),
            pred_tokens.max(2)[1].cpu(),
            ignore_index=0,
        )[0]

        tensorboard_logs = {
            "train/token_acc": token_acc,
            "train/token_loss" : loss_tokens,
            "train/space_loss" : loss_space_tokens
        }

        return {
                "loss": total_loss,
                "log": tensorboard_logs,
        }

        

    def validation_step(self, batch, batch_idx):
        self.model.eval()

        input_ids, token_type_ids, labels, labels_weight = batch

        pred_tokens = self.forward(input_ids, token_type_ids)
        loss_tokens = self.loss_fn(pred_tokens.transpose(1, 2), labels)

        loss_space_tokens = self.loss_fn(pred_tokens.transpose(1, 2), labels * labels_weight)
        loss_space_tokens = loss_space_tokens * self.hparams.weight_value
    
        total_loss = loss_tokens + loss_space_tokens

        token_acc = get_token_accuracy(
            labels.cpu(),
            pred_tokens.max(2)[1].cpu(),
            ignore_index=0,
        )[0]

        tensorboard_logs = {
            "val/token_acc": token_acc,
            "val/token_loss" : loss_tokens,
            "val/space_loss" : loss_space_tokens
        }

        return{
            "val_loss": total_loss,
            "val_token_acc": torch.Tensor([token_acc]),
        }
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_token_acc = torch.stack([x["val_token_acc"] for x in outputs]).mean()

        tensorboard_logs = {
            "val/loss": avg_loss,
            "val/token_acc": avg_token_acc
        }

        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }
    