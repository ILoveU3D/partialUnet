import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class SingleSinoSet(Dataset):
    def __init__(self, sino, sino2, label, scale=(1,1,1080,144,82*21)):
        sino = self._padd(sino)
        self.sino = torch.reshape(sino, scale)
        sino2 = self._padd(sino2)
        self.sino2 = torch.reshape(sino2, scale)
        self.label = torch.reshape(label, scale)

    def __getitem__(self, item):
        s = self.sino[...,item,:,:]
        l = self.sino2[...,item,:,:]
        lb = self.label[...,item,:,:]
        assert s.shape==lb.shape and l.shape==lb.shape
        s = s.squeeze(0)
        l = l.squeeze(0)
        lb = lb.squeeze(0)
        return s, l, lb

    def __len__(self):
        return self.sino.shape[2]

    def _padd(self, sino):
        return F.pad(sino, (2,2), value=0)

    def reset(self, sino):
        batchsize = sino.shape[0]
        w,h = sino.shape[3]//21, sino.shape[2]
        return torch.reshape(sino, (1,1,batchsize*21,h,w))