# Asteroid is based on PyTorch and PyTorch-Lightning.
from torch import optim
from pytorch_lightning import Trainer
import torch
# We train the same model architecture that we used for inference above.
from asteroid.models import DPRNNTasNet
# from asteroid.models import DPTNet

# In this example we use Permutation Invariant Training (PIT) and the SI-SDR loss.
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper

# MiniLibriMix is a tiny version of LibriMix (https://github.com/JorisCos/LibriMix),
# which is a free speech separation dataset.
from asteroid.data import LibriMix

# Asteroid's System is a convenience wrapper for PyTorch-Lightning.
from asteroid.engine import System

# This will automatically download MiniLibriMix from Zenodo on the first run.
# To train 2+1 ch model 
# train_loader, val_loader = LibriMix.loaders_from_mini(task="sep_noisy", batch_size=16, n_src=3)

# To train 2 ch model
train_loader, val_loader = LibriMix.loaders_from_mini(task="sep_noisy", batch_size=16, n_src=2)

# Tell DPRNN that we want to separate to 3 sources.
# model = DPRNNTasNet(n_src=3)

# Tell DPRNN that we want to separate to 2 sources.
model = DPRNNTasNet(n_src=2)

# PITLossWrapper works with any loss function.
loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

optimizer = optim.Adam(model.parameters(), lr=1e-3)

system = System(model, optimizer, loss, train_loader, val_loader)

# Train for 1 epoch using a single GPU. If you're running this on Google Colab,
# be sure to select a GPU runtime (Runtime → Change runtime type → Hardware accelarator).
trainer = Trainer(max_epochs=100, gpus=1)
trainer.fit(system)
#trainer.fit(system, ckpt_path="./asteroid/lightning_logs/version_9/checkpoints/epoch=99-step=5000.ckpt")

