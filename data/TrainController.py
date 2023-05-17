import torch.nn
from torch.utils.data import DataLoader
from data.AAPMSet import AAPMSet256
from data.SingleSet import SingleSinoSet

class TrainController():
    def __init__(self, model, cascades, dataRoot, minisize, device):
        self.minisize = minisize
        self.cascade = cascades
        dataset = AAPMSet256(dataRoot, device)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        self.fdk = dataset.fbp
        self.imgScale = dataset.scale
        self.model = model.to(device)
        self.lossFunction = torch.nn.MSELoss(reduction="sum")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=10e-4)

    def train(self):
        self.model.train()
        for data in self.dataloader:
            label, sino, latent = data
            print("-----New-----")
            for cascade in range(self.cascade):
                print("\t-----cascade {}-----".format(cascade))
                miniSet = SingleSinoSet(sino, latent, label)
                miniLoader = DataLoader(miniSet, batch_size=self.minisize, shuffle=False)
                for idx,subsino in enumerate(miniLoader):
                    print("\t\t-----Unet Batch {}-----".format(idx))
                    sinoCurr, latentCurr, labelCurr = subsino
                    sinoOut = self.model(sinoCurr, latentCurr)
                    self.optimizer.zero_grad()
                    loss = self.lossFunction(sinoOut, sinoCurr)
                    print("\t\t-----loss {}-----".format(loss.item()))
                    loss.backward()
                    self.optimizer.step()
                    sino[...,idx*21:(idx+self.minisize)*21,:,:] = miniSet.reset(sinoOut.detach())
                latent = self.fdk(torch.zeros(self.imgScale).to(sino.device), sino)
            del sino, latent, label

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