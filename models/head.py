from typing import Tuple, Optional, List, Dict
import torch.nn as nn
import torch
from scipy import special


class LinearHead(nn.Module):
    def __init__(self, dim, num_class):
        super(LinearHead, self).__init__()
        self.head = nn.Linear(dim, num_class)

    def forward(self,x):
        x = self.head(x)
        return x
    

class PoissonHead(nn.Module):
    def __init__(self, dim, num_class):
        super(PoissonHead, self).__init__()
        self.num_classes = num_class
        self.dim = dim
        self.register_buffer('j', torch.arange(num_class))
        self.register_buffer('j_fac', torch.tensor(special.factorial(self.j)))
        self.register_parameter('tau', nn.Parameter(torch.zeros((1,1))))
        self.j = self.j.unsqueeze(0)
        self.j_fac = self.j_fac.unsqueeze(0)

        self.head = nn.Sequential(#dropout
                nn.BatchNorm1d(self.dim),
                nn.Linear(self.dim, 1),
                nn.Softplus()
            )

    def forward(self,x, return_lbd=False):
        bs = x.shape[0]
        x = self.head(x)
        lbd = x
        x = x.expand(bs,self.num_classes)
        x = self.j*torch.log(x)-x-torch.log(self.j_fac)
        x = x/torch.sigmoid(self.tau)
        if return_lbd:
            return x, lbd
        else:
            return x

class OEHead(nn.Module):
    def __init__(self, dim, num_class):
        super(OEHead, self).__init__()
        self.head = nn.Linear(dim, num_class)

    def forward(self,x):
        x = self.head(x)
        x = torch.sigmoid(x)
        #x = torch.cat([(1.-x[:,0]).unsqueeze(-1), x],1)
        return x