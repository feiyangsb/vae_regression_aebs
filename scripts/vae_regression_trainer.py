"""
@author Feiyang Cai
@email feiyang.cai@vanderbilt.edu
@create date 2020-03-15 20:46:37
@modify date 2020-03-15 20:46:37
@desc [description]
"""

from scripts.data_loader import CarlaAEBSDataset
from torch.utils.data import DataLoader
import torch
from scripts.network import VAEPerceptionNet
import torch.optim as optim
import numpy as np
import time
import os
import logging
from datetime import date

alpha = 224*224*224
beta = 1.0
theta = 0.0
rho = 0.0

today = date.today()
file_path = "./log"
filename = today.strftime("%B_%d")+"_"+str(alpha)+"_"+str(beta)+"_"+str(theta)+"_"+str(rho)+".log"
if not os.path.exists(file_path):
    os.makedirs(file_path)
logging.basicConfig(level=logging.INFO, filename=os.path.join(file_path,filename))

class VAERegression():
    def __init__(self, data_path, epoch):
        self.dataset = CarlaAEBSDataset(data_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = epoch
    
    def fit(self):
        self.model = VAEPerceptionNet()
        self.model = self.model.to(self.device)
        dataloader = DataLoader(self.dataset, batch_size=64, shuffle=True, num_workers=8)
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.5e-6, amsgrad=False) 
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(self.epoch*0.7)], gamma=0.1)
        self.model.train()
        loss_func = torch.nn.MSELoss()
        for epoch in range(self.epoch):
            loss_epoch = 0.0
            reconstruction_loss_epoch = 0.0
            kl_loss_epoch = 0.0
            label_loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()

                outputs, z_mu, z_logvar, pz_mu, r_mu, r_logvar = self.model(inputs)
                reconstruction_loss = loss_func(outputs, inputs)
                #reconstruction_loss = torch.sum((outputs - inputs)**2, dim=tuple(range(1, outputs.dim())))
                #kl_loss = 1 + z_logvar - (z_mu ).pow(2) - z_logvar.exp()
                kl_loss = 1 + z_logvar - (z_mu - rho*pz_mu).pow(2) - z_logvar.exp()
                kl_loss = torch.sum(kl_loss, axis=-1) * -0.5
                label_loss = torch.div(0.5*(r_mu-targets).pow(2), r_logvar.exp()) + 0.5 * r_logvar
                loss = alpha * reconstruction_loss + beta * kl_loss + theta * label_loss
                reconstruction_loss_mean = torch.mean(reconstruction_loss)
                kl_loss_mean = torch.mean(kl_loss)
                label_loss_mean = torch.mean(label_loss)
                loss = torch.mean(loss)
                #print(torch.mean(reconstruction_loss).item(), torch.mean(kl_loss).item(), torch.mean(label_loss).item())
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                reconstruction_loss_epoch += reconstruction_loss_mean.item()
                kl_loss_epoch += kl_loss_mean.item()
                label_loss_epoch += label_loss_mean.item()
                n_batches += 1
            
            scheduler.step()
            epoch_train_time = time.time() - epoch_start_time
            logging.info('Epoch {}/{}\t Time: {:.5f}\t Total Loss: {:.5f}\t\
                        Reconstruction Loss {:.5f}\t KL Loss {:.5f} \t\
                        Lable Loss: {:.5f}'.format(
                            epoch+1, self.epoch, epoch_train_time, loss_epoch/n_batches, reconstruction_loss_epoch/n_batches, \
                            kl_loss_epoch/n_batches, label_loss_epoch/n_batches))
            print(epoch)
            #print('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(epoch+1, 100, epoch_train_time, loss_epoch/n_batches))
        return self.model
    
    def save_model(self, path = "./models"):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), os.path.join(path, "perception.pt"))
