# coding=utf-8
import numpy as np
import torch
import os
import torch.utils.data as Data
import pickle
import argparse
import shutil
import time
import math

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim

from baseline_model import Baseline_model
from simple_model import Simple_model
from data_loader import Corrupted_dataset
from meta_weight_net import train_wrn
from models import resnet
from models import wrn_glc
from models import wideresnet
import numpy as np
import matplotlib.pyplot as plt
from data_loader import Corrupted_dataset


parser = argparse.ArgumentParser(description='uncertainty reweighting in noisy datasets')
parser.add_argument('--epochs', type=int, default=75,
                    help='number of epochs to train')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--log_interval', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--nesterov', type=bool, default=True,
                    help='nesterov momentum')
parser.add_argument('--milestones', type=list, default=[55,65],
                    help='milestones for lr')                    
parser.add_argument('--num_classes', type=int, default=10,
                    help='how many classes in dataset(CIFAR 10 or CIFAR 100)')
parser.add_argument('--corruption_prob', type=float, default=0,
                    help='the corrution ratio in the training set')
parser.add_argument('--corruption_type', type=str, default="unif",
                    help='the corrution type in the training set')
parser.add_argument('--dropout_prob', type=float, default=0,
                    help='the dropout ratio')
parser.add_argument('--model', type=str, default="wrn",
                    help='wrn, rn or dn')
parser.add_argument('--lr_scheduler', type=str, default="multistep",
                    help='lr scheduler')
parser.add_argument('--net_arg', type=int, default=2,
                    help='growth rate for densenet and widden factor for wrn')
parser.add_argument('--depth', type=int, default=40,
                    help='depth of densenet(resnet is fixed 32)')
best_prec1 = 0
log_file=str()


def main():
    global args, best_prec1, log_file
    args = parser.parse_args()
    data_dir="../data"
    results_dir="../results"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    elif args.model=="rn":
        Net = resnet.ResNet(depth=args.depth,num_classes=args.num_classes,n_var=args.n_var,block_name="BasicBlock")
    elif args.model=="wrn":
        Net = wrn_glc.WideResNet_glc(args.depth, args.num_classes, args.net_arg, args.dropout_prob)
    optimizer = optim.SGD(Net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay,nesterov=args.nesterov)
    if args.lr_scheduler=="multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones)
    elif args.lr_scheduler=="cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5,eta_min=4e-08)
    elif args.lr_scheduler=="cosine_m":
        pass
    else:
        assert False, "Invalid lr schedualer type"

    results_dir = os.path.join(results_dir,time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    log_file_dir = os.path.join(results_dir,"log.txt")
    log_file = str()
    log_file += "Hyperparameters:\n"+str(args)+"\n"
    total_params=0
    for x in filter(lambda p: p.requires_grad, Net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    log_file += " == total parameters: " + str(total_params) + "\n"

    model = torch.nn.DataParallel(Net)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cudnn.benchmark = True

    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    """train_set=Corrupted_dataset(args,root=data_dir,download=True,transform=None,target_transform=None,train=True)
    test_set=Corrupted_dataset(args,root=data_dir,download=True,transform=None,target_transform=None,train=False)
    train_loader=train_set.get_data_loader()
    val_loader=test_set.get_data_loader()"""

    train_loader, val_loader = train_wrn.build_dataset(args,data_dir)
    print(len(train_loader.dataset))
    print(len(val_loader.dataset))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    loss_curve=[]
    test_loss_curve=[]
    acc_curve=[]
    test_acc_curve=[]
    tt=0
    for epoch in range(args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        log_file+='current lr {:.5e}'.format(optimizer.param_groups[0]['lr'])
        tt=train(train_loader, val_loader, model, criterion, optimizer, epoch,loss_curve,test_loss_curve,acc_curve,test_acc_curve,device,tt)
        if args.lr_scheduler!="cosine_m":
            lr_scheduler.step()

        

    plt.clf()
    plt.plot(loss_curve)
    plt.yscale('log')
    plt.title('training loss curve')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir,"train_loss"))
    plt.clf()
    plt.plot(test_loss_curve)
    plt.yscale('log')
    plt.title('testing loss curve')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir,"test_loss"))
    plt.clf()
    plt.plot(acc_curve)
    plt.title('training acc curve')
    plt.savefig(os.path.join(results_dir,"train_acc"))
    plt.clf()
    plt.plot(test_acc_curve)
    plt.title('testing acc curve')
    plt.savefig(os.path.join(results_dir,"test_acc"))
    log_file="best test acc: "+str(best_prec1)+"\n"+log_file
    with open(log_file_dir,"w") as f:
        f.write(log_file)
    log_file=str()


def train(train_loader, test_loader, model, criterion, optimizer, epoch,loss_curve,test_loss_curve,acc_curve,test_acc_curve,device,tt):
    """
        Run one train epoch
    """
    global log_file
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device)
        input_var = input.to(device)
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output[:,:10].float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        loss_curve.append(loss.item())
        top1.update(prec1.item(), input.size(0))
        acc_curve.append(prec1.item())

        if args.lr_scheduler=="cosine_m":
            dt = math.pi/float(args.epochs)
            tt += float(dt)/(len(train_loader.dataset)/float(args.batch_size))
            if tt >= math.pi - 0.05:
                tt = math.pi - 0.05
            curT = math.pi/2.0 + tt
            new_lr = args.learning_rate * (1.0 + math.sin(curT))/2.0    # lr_min = 0, lr_max = lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        if i % args.log_interval == 0:
            log_file+='Epoch: [{0}][{1}/{2}]\t Loss {loss.val:.4f} ({loss.avg:.4f})\t Prec@1 {top1.val:.3f} ({top1.avg:.3f})\n'.format(epoch, i, len(train_loader), loss=losses, top1=top1)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), loss=losses, top1=top1))
    log_file+='====> Train Epoch: {} Average loss: {:.4f} Average acc: {:.4f}\n'.format(
        epoch, losses.avg, top1.avg)
    print('====> Train Epoch: {} Average loss: {:.4f} Average acc: {:.4f}'.format(
        epoch, losses.avg, top1.avg))
    test_res=validate(test_loader,model,criterion,device)
    test_loss_curve.append(test_res[0])
    test_acc_curve.append(test_res[1])
    return tt


def validate(val_loader, model, criterion,device):
    """
    Run evaluation
    """
    global best_prec1,log_file
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))


    log_file+='====> Test Average loss: {:.4f} Average acc: {:.4f}\n'.format(losses.avg, top1.avg)
    print('====> Tes Average loss: {:.4f} Average acc: {:.4f}'.format(losses.avg, top1.avg))
    best_prec1=max(best_prec1,top1.avg)
    return losses.avg,top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()