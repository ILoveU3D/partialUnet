import setproctitle
from data import TrainController
from models import PartialConvUnet
setproctitle.setproctitle("(wyk) Partial Unet")

if __name__ == '__main__':
    root = r"/home/nanovision/wyk/data/trainData"
    model = PartialConvUnet()
    strategy = TrainController(model, cascades=2, dataRoot=root, minisize=180, device=4)
    strategy.train()