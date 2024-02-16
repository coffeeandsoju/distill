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
            nlayers=64,
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
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)


L.seed_everything(42)

# data
dataset = WikiText2()
train_dataloader = DataLoader(dataset, num_workers=39)

# model
model = LanguageModel(vocab_size=dataset.vocab_size)

# trainer
policy = {nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}
# strategy = FSDPStrategy(auto_wrap_policy=policy)
# trainer = L.Trainer(accelerator="cuda", devices=2, strategy=strategy)
trainer = L.Trainer(accelerator="cuda", devices=2, strategy="ddp")
trainer.fit(model, train_dataloader)
trainer.print(torch.cuda.memory_summary())