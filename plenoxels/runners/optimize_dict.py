# Given a trained plenoxel grid, optimize a dictionary of patches of desired size
from plenoxels.configs import optimize_dict_config, parse_config
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

np.random.seed(0)

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


class PatchDict(torch.nn.Module):
    def __init__(self, num_atoms, patch_reso, data_dim):
        super().__init__()
        self.num_atoms = num_atoms
        self.patch_reso = patch_reso
        self.data_dim = data_dim

        self.dictionary = nn.Parameter(torch.empty(
            self.data_dim, self.patch_reso, self.patch_reso, self.patch_reso, self.num_atoms
        ))
        nn.init.uniform_(self.dictionary, 0.01, 0.1)

    def encode_patches(self, patches):
        # Compute the inverse dictionary
        atoms = self.dictionary.reshape(-1, self.num_atoms)  # [patch_size, num_atoms]
        # print(atoms.device)
        # pinv = torch.linalg.pinv(atoms) # [num_atoms, patch_size]
        # print(pinv.device)
        # Apply to the patches
        vectorized_patches = patches.view(patches.size(0), -1) # [batch_size, patch_size]
        # weights = vectorized_patches @ pinv.T  # [batch_size, num_atoms]
        weights = torch.linalg.lstsq(atoms, vectorized_patches.T, driver="gels").solution.T
        return weights @ atoms.T  # [batch_size, patch_size]
    
    def forward(self, patches):
        return self.encode_patches(patches)


def get_all_patches(model, patch_reso):
    return model.unfold(1, patch_reso, 1) \
         .unfold(2, patch_reso, 1) \
         .unfold(3, patch_reso, 1) \
         .reshape(model.shape[0], -1, patch_reso, patch_reso, patch_reso)  # [data_dim, num_patches, patch_reso, patch_reso, patch_reso]


gpu = get_freer_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
print(f'gpu is {gpu}')

train_cfg, _ = parse_config(optimize_dict_config.get_cfg_defaults())
train_log_dir = os.path.join(train_cfg.logdir, train_cfg.expname)
os.makedirs(train_log_dir, exist_ok=True)

# Load the pretrained grid
checkpoint_data = torch.load(os.path.join(train_cfg.logdir, train_cfg.reload, "model.pt"), map_location='cpu')
model = checkpoint_data["model"]["data"].squeeze()  # [data_dim, reso, reso, reso]

# Get all the patches
all_patches = get_all_patches(model, train_cfg.model.patch_reso)  # [data_dim, num_patches, patch_reso, patch_reso, patch_reso]
all_patches = all_patches.transpose(0,1).contiguous().pin_memory()  # [num_patches, data_dim, patch_reso, patch_reso, patch_reso]

# Training loop
patch_dict = PatchDict(num_atoms=train_cfg.model.num_atoms, patch_reso=train_cfg.model.patch_reso, data_dim=model.shape[0])
patch_dict = patch_dict.cuda().train()
optim = torch.optim.Adam(patch_dict.parameters(), lr=train_cfg.optim.lr)
for step in range(train_cfg.optim.num_samples):
    # Get a batch
    idx = np.random.choice(all_patches.shape[0], size=train_cfg.optim.batch_size)  # this is slow
    batch = all_patches[idx, ...]  # [batch_size, data_dim, patch_reso, patch_reso, patch_reso]
    batch = batch.reshape(train_cfg.optim.batch_size, -1)  # [batch_size, patch_size]
    batch = batch.cuda()
    # Encode the batch of patches
    encoded_batch = patch_dict(batch)
    # Compute loss
    loss = F.mse_loss(encoded_batch, batch)
    # Take step
    optim.zero_grad()
    loss.backward()
    optim.step()

    if step % 400 == 0:
        print(f'loss at step {step} is {loss}')

print(f"Saving learnt dictionary to {train_log_dir}")
torch.save({
    'step': step,
    'optimizer': optim.state_dict(),
    'model': patch_dict.state_dict(),
}, os.path.join(train_log_dir, "model.pt"))



