import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

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


class LanguageModel(L.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.hook = TriggerWandbSyncHook(
            communication_dir="/home/gridsan/mattfeng/.wandb_osh_command_dir"
        )
        self.model = Transformer(
            vocab_size=vocab_size,
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
        return torch.optim.Adam(self.parameters(), lr=0.01, eps=1e-4)
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
train_dataloader = DataLoader(dataset, batch_size=32, num_workers=39)

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
    precision="16-mixed",
)
# trainer = L.Trainer(accelerator="cuda", devices=2, strategy="ddp")
trainer.fit(model, train_dataloader)
trainer.print(torch.cuda.memory_summary())
