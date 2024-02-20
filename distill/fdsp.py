import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

import numpy as np

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.demos import Transformer, WikiText2

import bitsandbytes as bnb

# from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback
from wandb_osh.hooks import TriggerWandbSyncHook
import wandb

import logging

logging.getLogger("lightning.pytorch").setLevel(logging.INFO)
# console_handler = logging.StreamHandler(sys.stdout)
# logging.getLogger("lightning").addHandler(console_handler)


# source: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        # linear until self.warmup, then cosine decay
        if epoch <= self.warmup:
            lr_factor = (epoch + 1) / (self.warmup + 1)
        else:
            lr_factor = 0.5 * (
                1
                + np.cos(
                    np.pi * (epoch - self.warmup) / (self.max_num_iters - self.warmup)
                )
            )

        return lr_factor


class LanguageModel(L.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.hook = TriggerWandbSyncHook(
            communication_dir="/home/gridsan/mattfeng/.wandb_osh_command_dir"
        )
        self.model = None
        self.vocab_size = vocab_size

    def configure_model(self):
        if self.model is not None:
            return

        self.model = Transformer(
            vocab_size=self.vocab_size,
            nlayers=int(32 * 1),
            nhid=4096,
            ninp=1024,
            nhead=64,
        )

    def training_step(self, batch):
        # logging.getLogger("lightning.pytorch").info("step")
        inp, tgt = batch
        output = self.model(inp, tgt)
        loss = F.nll_loss(output, tgt.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.03)
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=3, max_iters=100)
        return [optimizer], [scheduler]
        # return torch.optim.Adam(self.parameters(), lr=0.01, eps=1e-4)
        # return bnb.optim.Adam8bit(self.parameters(), lr=0.001, betas=(0.9, 0.995))

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # logging.getLogger("lightning.pytorch").info("step")
        if wandb.run is not None and batch_idx % 50 == 1:
            # logging.getLogger("lightning.pytorch").info("hook")
            self.hook()


L.seed_everything(42)

# data
dataset = WikiText2()
train_dataloader = DataLoader(dataset, batch_size=16, num_workers=39)

# model
model = LanguageModel(vocab_size=dataset.vocab_size)

# trainer
policy = {nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}
# strategy = FSDPStrategy(auto_wrap_policy=policy)
# trainer = L.Trainer(accelerator="cuda", devices=2, strategy=strategy)
logger = WandbLogger(project="large-models", offline=True)
trainer = L.Trainer(
    logger=logger,
    accelerator="cuda",
    devices=2,
    strategy="ddp",
    precision="32-true",
)
# trainer = L.Trainer(accelerator="cuda", devices=2, strategy="ddp")
trainer.fit(model, train_dataloader)
trainer.print(torch.cuda.memory_summary())
