import torch
from models.PartialConvUnet import PartialConvUnet

input_img = torch.ones([9,1,144,1680]).cuda()
model = PartialConvUnet().cuda()
output = model(input_img)
pass