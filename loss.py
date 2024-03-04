import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import numpy as np
from scipy import special
import pdb

__all__ = ['EMDLoss', 'CrossEntropy', 'CrossEntropyMean', 'FocalLoss', 'PoissonFocal', 'Poisson', 'OrdinalFocal']

class Poisson(CrossEntropyLoss):
    def __init__(self, args):
        super(Poisson, self).__init__() 

class PoissonFocal(nn.Module):
    def __init__(self, args):
        super(PoissonFocal, self).__init__() 
        self.register_buffer('j', torch.arange(args.num_classes))
        self.register_buffer('j_fac', torch.tensor(special.factorial(self.j)))
    
        self.j = self.j.unsqueeze(0)
        self.j_fac = self.j_fac.unsqueeze(0)
        self.num_classes = args.num_classes
        self.t = 0.1
        self.smooth = 1e-4
        self.alpha=0.75 #optimal 0.75 
        self.gamma=2 #optimal 2 
    
    def forward(self, logit, target):
        bs = logit.shape[0]
    
        prob = F.softmax(logit, dim=1) + self.smooth
        logpt = torch.log(prob)

        target = target.unsqueeze(1)
        target_prob = target.expand(bs,self.num_classes) + 0.5
        target_prob = self.j*torch.log(target_prob)-target_prob-torch.log(self.j_fac)
        target_prob = torch.softmax(target_prob / self.t, -1)
        loss = torch.sum(-target_prob * logpt, dim=1)

        weight = (target_prob - prob).gather(1, target).view(-1)
        loss = self.alpha * torch.pow(weight, self.gamma) * loss
        return loss.mean()

class CrossEntropy(CrossEntropyLoss):
    def __init__(self, args):
        super(CrossEntropy, self).__init__()


class CrossEntropyMean(nn.Module):
    def __init__(self, args):
        super(CrossEntropyMean, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.l1 = nn.SmoothL1Loss()

    def forward(self, logits, label):
        loss_ce = self.ce(logits, label.long())
        n,c = logits.shape
        grid = torch.arange(c, device=logits.device).view(1,c)
        logits = (logits.softmax(1) * grid).sum(1)
        loss_mean = self.l1(logits, label.float())
        return loss_mean+loss_ce



class EMDLoss(nn.Module):
    def __init__(self, args):
        super(EMDLoss, self).__init__()
        self.num_classes = args.num_classes

    def forward(self, logits, label):
        logits = logits.softmax(1)
        label = F.one_hot(label, num_classes=self.num_classes)
        loss = torch.cumsum(label, -1) - torch.cumsum(logits, -1)
        loss = torch.square(loss)
        return loss.mean()

class FocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor shape=[num_class, ]
        gamma: hyper-parameter
        reduction: reduction type
    """

    def __init__(self, args, alpha=0.75, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.num_class = args.num_classes
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(args.num_classes, ) - 0.5
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * args.num_classes)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != args.num_classes:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)

        if prob.dim() > 2:
            # used for 3d-conv:  N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            N, C = logit.shape[:2]
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]

        ori_shp = target.shape
        target = target.view(-1, 1)

        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.squeeze(-1))
        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)

        return loss


class OrdinalFocal(nn.Module):
    def __init__(self, args):
        super(OrdinalFocal, self).__init__()
        self.enc = torch.triu(torch.ones(args.num_classes,args.num_classes),
                              diagonal=0).t().to(args.device)
        self.alpha = 0.75

    def forward(self, prob, target):
        target = self.enc[target]
        q = target*(1-prob)*(1-prob) + (1-target)*prob*prob
        q, _ = torch.max(q,dim=-1)
        p = -self.alpha * target * torch.log(prob) - (1-self.alpha)*(1-target)*torch.log(1-prob)
        p = torch.sum(p,-1)
        return torch.mean(p*q)