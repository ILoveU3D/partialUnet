import torch
import geom
from data import TrainController
from models import PartialConvUnet

if __name__ == '__main__':
    root = r"/media/wyk/wyk/Data/raws/trainData"
    model = PartialConvUnet()
    strategy = TrainController(model, 1, root, 3, 0)
    strategy.train()