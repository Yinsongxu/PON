import argparse
from dataset.dataset import build_loader
from functools import partial
import torch
import os
import logging
import numpy as np
from util.util import setup_logger, save_checkpoint
import loss as loss_builder
from metric import accuracy, ConfusionMatrix, kappa
import pdb
from util.meter import AverageMeter, ProgressMeter
import time
from models.classifierMemory import ClassifierMemoryReg
from util.lars import LARS
import util.lr_sched as lr_sched
from sklearn.metrics import roc_auc_score, top_k_accuracy_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default='/', type=str)
    parser.add_argument("--split", default=0, type=int)
    parser.add_argument("--snapshot_path", default="", type=str)
    parser.add_argument("--ckpt_path", default="", type=str)
    parser.add_argument("--crop_spatial_size", default=(128,128,32), type=tuple)
    parser.add_argument("--device", default="cuda", type=str)
    
    parser.add_argument("--loss", default="", type=str)
    parser.add_argument("--optim", default="adam", type=str)
    parser.add_argument("--num_classes", default=5, type=int)
    parser.add_argument("--model", default="", type=str)
    parser.add_argument('--bottleneck', action='store_true')
    parser.set_defaults(bottleneck=False)
    parser.add_argument('--bottleneck-dim', default=256, type=int, help='Dimension of bottleneck')
    parser.add_argument('--hidden_dim', default=256, type=int, help='Dimension of projector')
    parser.add_argument('--memeory_size', default=256, type=int)
    parser.add_argument('--memeory_temp', default=0.1, type=float)
    parser.add_argument('--memeory_k', default=20, type=int)
    parser.add_argument('--weightsampler', action='store_true')
    parser.set_defaults(weightsampler=False)

    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--min_lr', type=float, default=0, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument("-bs", "--batch_size", default=4, type=int)
    parser.add_argument("--max_epoch", default=100, type=int)
    parser.add_argument("--eval_interval", default=5, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pretrain", default=None, type=str)

    parser.add_argument('--analysis', action='store_true')
    parser.set_defaults(analysis=False)
    args = parser.parse_args()

    args.snapshot_path = os.path.join(args.snapshot_path)
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"train")
    logger.info(str(args))
    
    
    confmat = ConfusionMatrix(args.num_classes) 

    prob, gts = [], []

    for i in range(5):
        args.split = i
        train_loader, val_loader, test_loader = build_loader(args)
        model = ClassifierMemoryReg(args).to(args.device)
        model.eval()
        ckpt_path = os.path.join(args.ckpt_path, f'split_{args.split}',  'best.pth.tar')
        model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        
        with torch.no_grad():
            for idx, (img, gt, _) in enumerate(test_loader):
                img, gt = img.to(args.device), gt.to(args.device)
                output = model(img)
                logit = output['logit']
                #print(output['lbd'], logit.argmax(1), gt)
                prob.append(torch.softmax(logit, -1))
                gts.append(gt)
                # acc1 = accuracy(logit, gt, topk=(1,))
                # confmat.update(gt, logit.argmax(1))
                # top1.append(acc1[0].item())
    prob = torch.cat(prob, 0).cpu().numpy()
    gts = torch.cat(gts, 0).cpu().numpy()

    #score
    #acc
    score_acc = top_k_accuracy_score(gts, prob, k=1) * 100
    score_qwk = kappa(gts, np.argmax(prob, 1))
    score_auc = roc_auc_score(gts, prob, multi_class='ovr') * 100

    #screen
    screen_prob = np.sum(prob[:,1:],-1)
    screen_gts = (gts>0).astype(int)
    screen_acc = top_k_accuracy_score(screen_gts, screen_prob, k=1) * 100
    screen_auc = roc_auc_score(screen_gts, screen_prob, multi_class='ovr') * 100

    #signifcant
    sig_prob = np.sum(prob[:,3:],-1)
    sig_gts = (gts>3).astype(int)
    sig_acc = top_k_accuracy_score(sig_gts, sig_prob, k=1) * 100
    sig_auc = roc_auc_score(sig_gts, sig_prob) * 100

    print(f"score: acc {score_acc:.2f} \t auc {score_auc:.2f} \t qwk {score_qwk}" )
    print(f"screen: acc {screen_acc:.2f} \t auc {screen_auc:.2f}")
    print(f"signifcant: acc {sig_acc:.2f} \t auc {sig_auc:.2f}")


if __name__ == "__main__":
    main()