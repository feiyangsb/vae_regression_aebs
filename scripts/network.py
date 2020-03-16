"""
@author Feiyang Cai
@email feiyang.cai@vanderbilt.edu
@create date 2020-03-15 20:36:34
@modify date 2020-03-15 20:36:34
@desc [description]
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class ConstrinedLinear(nn.Linear):
    def forward(self, input):
        return F.linear(input, self.weight/(self.weight.norm()+1e-7))

class VAEPerceptionNet(nn.Module):
    def __init__(self):
        super(VAEPerceptionNet, self).__init__()

        self.pool = nn.MaxPool2d(2,2)

        self.conv1 = nn.Conv2d(3, 32, 5, padding=2, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        self.conv2 = nn.Conv2d(32, 64, 5, padding=2, bias=False)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.conv3 = nn.Conv2d(64, 128, 5, padding=2, bias=False)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        self.conv4 = nn.Conv2d(128, 256, 5, padding=2, bias=False)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.bn2d4 = nn.BatchNorm2d(256, eps=1e-04, affine=False)

        self.fc1 = nn.Linear(256 * 14 * 14, 1568, bias=False)

        self.fc21 = nn.Linear(1568, 1024, bias=False)
        self.fc22 = nn.Linear(1568, 1024, bias=False)

        self.fc2 = nn.Linear(1568, 256, bias=False)
        self.fc33 = nn.Linear(256, 1, bias=False)
        self.fc34 = nn.Linear(256, 1, bias=False)

        self.fc41 = nn.Linear(1024, 1568, bias=False)
        self.fc42 = nn.Linear(1, 256, bias=False)
        self.fc5 = ConstrinedLinear(256, 1024, bias=False)

        self.deconv1 = nn.ConvTranspose2d(int(1568 / (14 * 14)), 256, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight)
        self.bn2d5 = nn.BatchNorm2d(256, eps=1e-04,affine=False)

        self.deconv2 = nn.ConvTranspose2d(256, 128, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight)
        self.bn2d6 = nn.BatchNorm2d(128, eps=1e-04,affine=False)

        self.deconv3 = nn.ConvTranspose2d(128, 64, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight)
        self.bn2d7 = nn.BatchNorm2d(64, eps=1e-04,affine=False)

        self.deconv4 = nn.ConvTranspose2d(64, 32, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight)
        self.bn2d8 = nn.BatchNorm2d(32, eps=1e-04,affine=False)

        self.deconv5 = nn.ConvTranspose2d(32, 3, 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv5.weight)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.elu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.elu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.elu(self.bn2d3(x)))
        x = self.conv4(x)
        x = self.pool(F.elu(self.bn2d4(x)))
        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1(x))

        z_mu = self.fc21(x)
        z_logvar = self.fc22(x)
        x = self.fc2(x)
        r_mu = self.fc33(x)
        r_logvar = self.fc34(x)

        z = self.reparameterize(z_mu, z_logvar)
        r = self.reparameterize(r_mu, r_logvar)

        x = F.elu(self.fc41(z))
        #x = F.elu(self.fc5(x))
        x = x.view(x.size(0), int(1568 / (14*14)), 14, 14)
        x = F.elu(x)

        x = self.deconv1(x)
        x = F.interpolate(F.elu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.elu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.elu(self.bn2d7(x)), scale_factor=2)
        x = self.deconv4(x)
        x = F.interpolate(F.elu(self.bn2d8(x)), scale_factor=2)
        x = self.deconv5(x)
        x = torch.sigmoid(x)

        pz_mu_pre = self.fc42(r)
        pz_mu = self.fc5(pz_mu_pre)
        
        return x, z_mu, z_logvar, pz_mu, r_mu, r_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std
        #std = logvar.mul(0.5).exp_()
        #eps = torch.FloatTensor(std.size()).normal_()
        #eps = Variable(eps)
        #return eps.mul(std).add_(mu)