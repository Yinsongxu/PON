import argparse
from dataset.dataset import build_loader
from functools import partial
import torch
import os
import logging
import numpy as np
from util.util import setup_logger, save_checkpoint
from models.classifierMemory import ClassifierMemoryReg
from torch.optim import AdamW
import loss as build_loss
from metric import accuracy, ConfusionMatrix
import pdb
from util.meter import AverageMeter, ProgressMeter
import time
import util.lr_decay as lrd
import util.lr_sched as lr_sched
import timm.optim.optim_factory as optim_factory
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from torch.autograd import grad
import torch.nn.functional as F
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default='', type=str)
    parser.add_argument("--split", default=0, type=int)
    parser.add_argument("--snapshot_path", default="", type=str)
    parser.add_argument("--crop_spatial_size", default=(128,128,32), type=tuple)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

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

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"train")
    logger.info(str(args))
    
    train_loader, val_loader, test_loader = build_loader(args)
    cls_count = torch.from_numpy(args.cls_account).float().to(args.device)

    model = ClassifierMemoryReg(args).to(args.device)
    #model = torch.compile(model)
    loss_cal = build_loss.__dict__[args.loss](args).to(args.device)
    loss_scaler = NativeScaler()

    best_loss = np.inf
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)#, weight_decay=args.weight_decay)

    args.epochs = args.max_epoch
    if not args.analysis:
        for epoch_num in range(args.max_epoch):
            train(args, epoch_num, model, cls_count, train_loader, optimizer, loss_scaler, loss_cal, logger)
            if epoch_num>0 and (epoch_num+1)% args.eval_interval == 0:
                model.eval()
                confmat = ConfusionMatrix(args.num_classes) 
                with torch.no_grad():
                    loss_summary = []
                    for idx, (img, gt, _) in enumerate(val_loader):
                        img, gt = img.to(args.device), gt.to(args.device)
                        output = model(img)
                        logit = output['logit']
                        confmat.update(gt, logit.argmax(1))
                        
                        loss = loss_cal(logit, gt)
                        loss_summary.append(loss.detach().cpu().numpy())
                        logger.info(
                            'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(val_loader)) 
                            + " loss:" + str(loss_summary[-1].flatten()[0])
                            )
                        
                logger.info("- Val metrics: " + str(np.mean(loss_summary)))
                logger.info(confmat.format(list(range(args.num_classes))))
                logger.info(confmat.mat)
                is_best = False
                if np.mean(loss_summary) < best_loss:
                    best_loss = np.mean(loss_summary)
                    is_best = True

                save_checkpoint({"epoch": epoch_num,
                                "best_val_loss": best_loss,
                                "state_dict": model.state_dict()},
                                is_best=is_best,
                                checkpoint=args.snapshot_path)
                logger.info("- Val metrics best: " + str(best_loss))

    test(args, model, test_loader, logger)

    
def train(args, epoch_num, model, cls_count, train_loader, optimizer, loss_scaler, loss_cal, logger):
    loss_summary = []
    model.train()
    losses = AverageMeter('Loss', ':3.2f')
    reges = AverageMeter('Reg', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, reges, cls_accs],
        prefix="Epoch: [{}]".format(epoch_num))
    optimizer.zero_grad()
    end = time.time()

    for batch_idx, (img, gt, index) in enumerate(train_loader):
        lr_sched.adjust_learning_rate(optimizer, batch_idx / len(train_loader) + epoch_num, args)

        model.temperature = -(1.0 - 0.1) / (1.0 * args.epochs) * (batch_idx / len(train_loader) + epoch_num) + 1.0
        
        img, gt = img.to(args.device, non_blocking=True), gt.to(args.device, non_blocking=True)
        data_time.update(time.time() - end)
        with torch.cuda.amp.autocast():
            output = model(img)
            loss = loss_cal(output['logit'], gt)
            knn_loss = F.nll_loss(torch.log(output['knn_pred']), gt)
            loss_all = loss + knn_loss #+ 0.01*reg_loss
        loss_scaler(loss_all, optimizer, parameters=model.parameters())
        #torch.nn.utils.clip_grad_norm_(model.parameters(),  1.0)  
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            model.memory.update_queue(output['proj'], gt)

        cls_acc = accuracy(output['logit'], gt)[0]
        losses.update(loss.item(), img.size(0))
        #reges.update(reg_loss.item(), img.size(0))
        cls_accs.update(cls_acc, img.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        loss_summary.append(loss.detach().cpu().numpy())
        logger.info(progress.display(batch_idx))

    logger.info("- Train metrics: " + str(np.mean(loss_summary)))


def test(args, model, test_loader, logger):
    model.eval()
    filepath_best = os.path.join(args.snapshot_path, "best.pth.tar")
    model.load_state_dict(torch.load(filepath_best)['state_dict'])
    confmat = ConfusionMatrix(args.num_classes) 
    top1 = []

    with torch.no_grad():
        for idx, (img, gt, _) in enumerate(test_loader):
            img, gt = img.to(args.device), gt.to(args.device)
            output = model(img)
            logit = output['logit']
            acc1 = accuracy(logit, gt, topk=(1,))
            confmat.update(gt, logit.argmax(1))
            top1.append(acc1[0].item())

    print(confmat.format(list(range(args.num_classes))))
    print(confmat.mat)
    logger.info("- Test metrics acc: " + str(np.mean(top1)))


if __name__ == "__main__":
    main()