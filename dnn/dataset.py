import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class PLDataset(Dataset):
    def __init__(self, data_path, shape=(256, 256)):

        # load clean frames and striped frames
        self.original_frames = torch.from_numpy(np.load(os.path.join(data_path, 'origin_resized_{}_{}.npy'.format(shape[0], shape[1]))))
        self.striped_frames = torch.from_numpy(np.load(os.path.join(data_path, 'striped_resized_{}_{}.npy'.format(shape[0], shape[1]))))
        # load stripes
        freqs = torch.from_numpy(np.load(os.path.join(data_path, 'freqs.npy'))).float().unsqueeze(1)
        t = transforms.Resize([1, shape[1]])
        self.freqs = t(freqs)
        self.freqs = (self.freqs - torch.mean(self.freqs)) / (torch.max(self.freqs) - torch.min(self.freqs))

    def __len__(self):
        return self.original_frames.shape[0]
    
    def __getitem__(self, idx):
        orig, striped, freq = self.original_frames[idx,:,:,:].float(), self.striped_frames[idx,:,:,:].float(), self.freqs[idx]
        orig, striped = orig/255-0.5, striped/255-0.5

        return striped, orig, freq

class PLDataModule(LightningDataModule):

    def __init__(self, data_path, shape, bs):
        super().__init__()
        self.train_bs, self.val_bs, self.test_bs = bs, bs, bs
        self.shape = shape
        self.data_path = data_path
        
    def prepare_data(self):
        pass

    def setup(self, stage):
        self.dataset = PLDataset(self.data_path, self.shape)
        self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(self.dataset, [len(self.dataset)-100, 64, 36])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_bs, num_workers=0, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.val_bs, num_workers=0, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.test_bs, num_workers=0, shuffle=False)
