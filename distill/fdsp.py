import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.demos import Transformer, WikiText2


class LanguageModel(L.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.model = Transformer(
            vocab_size=vocab_size,
            nlayers=32,
            nhid=4096,
            ninp=1024,
            nhead=64,
        )
    
    def training_step(self, batch):
        inp, tgt = batch
        output = self.model(inp, tgt)
        loss = F.nll_loss(output, tgt.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss


L.seed_everything(42)

# data
dataset = WikiText2()
train_dataloader = DataLoader(dataset)

# model
model = LanguageModel(vocab_size=dataset.vocab_size)

# trainer
trainer = L.Trainer(accelerator="cuda", devices=2, strategy=FSDPStrategy())
trainer.fit(model, train_dataloader)
trainer.print(torch.cuda.memory_summary())