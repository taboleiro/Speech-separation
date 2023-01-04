# Asteroid is based on PyTorch and PyTorch-Lightning.
import torch
from torch import optim
from pytorch_lightning import Trainer

# We train the same model architecture that we used for inference above.
from asteroid.models import DPRNNTasNet
#from asteroid.models import DPTNet

# In this example we use Permutation Invariant Training (PIT) and the SI-SDR loss.
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper

# MiniLibriMix is a tiny version of LibriMix (https://github.com/JorisCos/LibriMix),
# which is a free speech separation dataset.
from asteroid.data import LibriMix

# Asteroid's System is a convenience wrapper for PyTorch-Lightning.
from asteroid.engine import System

# This will automatically download MiniLibriMix from Zenodo on the first run.
train_loader, val_loader = LibriMix.loaders_from_mini(task="sep_noisy", batch_size=16, n_src=2)

# Tell DPRNN that we want to separate to 2 sources.
model = DPRNNTasNet(n_src=2)
#newmodel = model.load_from_checkpoint("./lightning_logs/version_0/checkpoints/epoch=99-step=5000.ckpt")
#torch.save(newmodel.state_dict(), 'dprnn.pth')

# PITLossWrapper works with any loss function.
loss = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

optimizer = optim.Adam(model.parameters(), lr=1e-3)

system = System(model, optimizer, loss, train_loader, val_loader)

# upload the lightning_logs folder about the trained model to colab
# use the best checkpoint to reload our model
# state_dict = torch.load('./lightning_logs/version0/checkpoints/epoch=99-step=5000.ckpt')
state_dict = torch.load('./lightning_logs/version_1/checkpoints/epoch=99-step=5000.ckpt')
system.load_state_dict(state_dict=state_dict["state_dict"])
system.cpu()

to_save = system.model.serialize()
train_set = LibriMix(
    csv_dir="./MiniLibriMix/metadata/train",
    task= "sep_noisy",
    sample_rate=8000,
    n_src=2,
    segment=4,
)
to_save.update(train_set.get_infos())
# .pth file is used to test the model 
torch.save(to_save, "dprnn_model_22.pth")

#download dprnn_model_22.pth then use the same method as Svoice