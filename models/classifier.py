from typing import Tuple, Optional, List, Dict
import torch.nn as nn
import torch
from .resnet import resnet18_3D
from .head import LinearHead, PoissonHead, OEHead
import pdb
import torch.nn.functional as F
import math


__all__ = ['Classifier']

backbone = {'resnet18':resnet18_3D}

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()

        self.backbone = backbone[args.model]()
        if args.pretrain is not None:
            checkpoint = torch.load(args.pretrain, map_location='cpu')
            msg = self.backbone.load_state_dict(checkpoint, strict=False)
            print(msg)

        if args.bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(self.backbone.output_dim, args.bottleneck_dim),
                nn.BatchNorm1d(args.bottleneck_dim),
                nn.ReLU()
            )
            self._features_dim = args.bottleneck_dim
        else:
            self.bottleneck = nn.Identity()
            self._features_dim = self.backbone.output_dim

        if 'Poisson' in args.loss:
            self.head = PoissonHead(self._features_dim, args.num_classes)
        elif args.loss == 'OrdinalFocal':
            self.head = OEHead(self._features_dim, args.num_classes)
        else:
            self.head = LinearHead(self._features_dim, args.num_classes)
             
    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.backbone(x)
        f = self.bottleneck(f)
        predictions = self.head(f)
        return predictions, f

class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Projector, self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return self.layers(x)
    

class ClassifierMemory(nn.Module):
    def __init__(self, args):
        super(ClassifierMemory, self).__init__()

        self.backbone = backbone[args.model]()
        if args.pretrain is not None:
            checkpoint = torch.load(args.pretrain, map_location='cpu')
            msg = self.backbone.load_state_dict(checkpoint, strict=False)
            print(msg)
       
            
        self._features_dim = self.backbone.output_dim

        self.projector = Projector(self._features_dim, args.hidden_dim, args.hidden_dim)
        self.memory = MomentumQueue(args.hidden_dim, args.memeory_size, 
                                    args.memeory_temp, args.memeory_k, args.num_classes, eps_ball=1.1)
        self.head = LinearHead(self._features_dim, args.num_classes)
             
    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.backbone(x)
        logit = self.head(f)
        proj = self.projector(f)
        pred_scores = self.memory(proj)
        return logit, pred_scores
    
class MomentumQueue(nn.Module):
    def __init__(self, feature_dim, queue_size, temperature, k, classes, eps_ball=1.1):
        super(MomentumQueue, self).__init__()
        self.queue_size = queue_size
        self.index = 0
        self.temperature = temperature
        self.k = k
        self.classes = classes
        self.eps_ball = eps_ball

        # noinspection PyCallingNonCallable
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(feature_dim / 3)
        memory = torch.rand(self.queue_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', memory)
        memory_label = torch.zeros(self.queue_size).long()
        self.register_buffer('memory_label', memory_label)
    
    def update_queue(self, k_all, k_label_all):
        with torch.no_grad():
            k_all = F.normalize(k_all)
            all_size = k_all.shape[0]
            out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index, self.queue_size)
            self.memory.index_copy_(0, out_ids, k_all)
            self.memory_label.index_copy_(0, out_ids, k_label_all)
            self.index = (self.index + all_size) % self.queue_size

    def forward(self, x, test=False):
        dist = torch.mm(F.normalize(x), self.memory.transpose(1, 0))    # B * Q, memory already normalized
        max_batch_dist, _ = torch.max(dist, 1)
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        if self.eps_ball <= 1 and max_batch_dist.min() > self.eps_ball:
            sim_weight = torch.where(dist >= self.eps_ball, dist, torch.tensor(float("-inf")).float().to(x.device))
            sim_labels = self.memory_label.expand(x.size(0), -1)
            sim_weight = F.softmax(sim_weight / self.temperature, dim=1)
            # counts for each class
            one_hot_label = torch.zeros(x.size(0) * self.memory_label.shape[0], self.classes, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.contiguous().view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
        else:
            sim_weight, sim_indices = torch.topk(dist, k=self.k)
            sim_labels = torch.gather(self.memory_label.expand(x.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = F.softmax(sim_weight / self.temperature, dim=1)

            one_hot_label = torch.zeros(x.size(0) * self.k, self.classes, device=sim_labels.device)
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(x.size(0), -1, self.classes) * sim_weight.unsqueeze(dim=-1), dim=1)
        pred_scores = (pred_scores + 1e-5).clamp(max=1.0)
        return pred_scores
    