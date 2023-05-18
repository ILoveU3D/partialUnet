import torch.nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import geom
from data.AAPMSet import AAPMSet256
from data.SingleSet import SingleSinoSet

class TrainController():
    def __init__(self, model, cascades, dataRoot, minisize, device):
        self.minisize = minisize
        self.cascade = cascades
        dataset = AAPMSet256(dataRoot, device)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        self.fdk = geom.CompleteBeijingGeometryWithFBP().to(device)
        self.imgScale = dataset.scale
        self.model = model.to(device)
        self.model.init()
        self.lossFunction = torch.nn.L1Loss(reduction="sum")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=10e-4)
        self.epoch = 20
        self.checkpoint = "/home/nanovision/wyk/data/checkpoint_unet"
        self.template = "/home/nanovision/wyk/data/debug_unet"
        self.checkstep = 100

    def _train_epoch(self, e):
        self.model.train()
        lossSum = 0
        with tqdm(self.dataloader) as loader:
            for id,data in enumerate(loader):
                loader.set_description("Epoch:{}".format(e))
                label, sino, latent = data
                for cascade in range(self.cascade):
                    miniSet = SingleSinoSet(sino, latent, label)
                    miniLoader = DataLoader(miniSet, batch_size=self.minisize, shuffle=False)
                    lossSingle = 0
                    for idx,subsino in enumerate(miniLoader):
                        sinoCurr, latentCurr, labelCurr = subsino
                        sinoOut = self.model(sinoCurr, latentCurr)
                        self.optimizer.zero_grad()
                        loss = self.lossFunction(sinoOut, labelCurr)
                        loss.backward()
                        self.optimizer.step()
                        sino[...,idx*21:(idx+self.minisize)*21,:,:] = miniSet.reset(sinoOut.detach())
                        lossSingle += loss.item() / len(miniLoader)
                        if idx==0 and (id)%self.checkstep == 0:
                            sinoOut.reshape([self.minisize,144,1722])[int(self.minisize/2),...].detach().cpu().numpy().tofile("{}/out.raw".format(self.template))
                            labelCurr.reshape([self.minisize,144,1722])[int(self.minisize/2),...].detach().cpu().numpy().tofile("{}/label.raw".format(self.template))
                    latent = self.fdk(torch.zeros(self.imgScale).to(sino.device), sino)
                    latent = geom.CompleteForwardProjection.apply(latent)
                    loader.set_postfix(Cascade=cascade, CurrLoss=lossSingle, MeanLoss=lossSum / len(self.dataloader)*self.cascade)
                    lossSum += lossSingle
                del sino, latent, label
        return lossSum / len(self.dataloader)*self.cascade

    def train(self):
        for e in range(self.epoch):
            loss = self._train_epoch(e)
            if e%2 == 0:
                torch.save({"epoch": e, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), }, 
                            "{}/PartialUnet_{:.10f}.dict".format(self.checkpoint, loss))


    def test(self, sino):
        latent = self.fdk(torch.zeros(self.imgScale).to(sino.device), sino)
        label = latent
        self.model.eval()
        for cascade in range(self.cascade):
            miniSet = SingleSinoSet(sino, latent, label)
            miniLoader = DataLoader(miniSet, batch_size=self.minisize, shuffle=False)
            nextSino = torch.zeros_like(sino)
            for idx, subsino in enumerate(miniLoader):
                sinoCurr, latentCurr, labelCurr = subsino
                sinoOut = self.model(sinoCurr, latentCurr)
                nextSino[0, 0, idx:idx + self.minisize, :, :] = sinoOut.detach()
            sino = nextSino
            latent = self.fdk(torch.zeros(self.imgScale).to(latent.device), sino)
        return sino