import numpy as np
import torch
import scipy.io as sco
from .options import *

parameters = sco.loadmat(beijingParameterRoot)
parameters = np.array(parameters["projection_matrix"]).astype(np.float32)
parameters = torch.from_numpy(parameters).contiguous()
volumeSize = torch.IntTensor(beijingVolumeSize)
detectorSize = torch.IntTensor(beijingSubDetectorSize)
