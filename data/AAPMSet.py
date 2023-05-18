import numpy as np
import geom
import torch
import torch.nn.functional as F
import os
from torch.utils.data import Dataset


class AAPMSet256(Dataset):
    def __init__(self, path, device, scale=(1,1,64,256,256)):
        self.root = path
        self.files = os.listdir(path)
        self.scale = scale
        self.device = device
        self.fdk = geom.IncompleteBeijingGeometryWithFBP().to(device)

    def __getitem__(self, item):
        filename = os.path.join(self.root, self.files[item])
        img = np.fromfile(filename, dtype="float32")
        img = torch.from_numpy(img.reshape(self.scale)).to(self.device)
        sinoc01 = geom.CompleteForwardProjection.apply(img)
        sinoc01 = sinoc01.squeeze(0)
        sinoic02 = geom.IncompleteForwardProjection.apply(img)
        # sinoic01 = geom.CompleteBackwardProjection.apply(sinoic02)
        sinoic01 = self.fdk.filterBackprojection(sinoic02)
        sinoic01 = geom.CompleteForwardProjection.apply(sinoic01)
        sinoic02 = F.pad(sinoic02, (2, 2), value=0)
        sinoic02 = sinoic02.squeeze(0)
        sinoic01 = sinoic01.squeeze(0)
        return sinoc01, sinoic02, sinoic01

    def __len__(self):
        return len(self.files)