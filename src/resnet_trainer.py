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
import torch.nn.functional as F

from baseline_model import Baseline_model
from simple_model import Simple_model
from data_loader import Corrupted_dataset
#from meta_weight_net import train_wrn
from models import resnet
from models import wrn_glc
from models import wideresnet
import numpy as np
import matplotlib.pyplot as plt
from data_loader import Corrupted_dataset
import copy
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='uncertainty reweighting in noisy datasets')
parser.add_argument('--epochs', type=int, default=120,
                    help='number of epochs to train')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--n_var', type=int, default=10,
                    help='how many var to predict')
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
parser.add_argument('--milestones', type=list, default=[80,100],
                    help='milestones for lr')                    
parser.add_argument('--num_classes', type=int, default=10,
                    help='how many classes in dataset(CIFAR 10 or CIFAR 100)')
parser.add_argument('--corruption_prob', type=float, default=0.6,
                    help='the corrution ratio in the training set')
parser.add_argument('--corruption_type', type=str, default="unif",
                    help='the corrution type in the training set')
parser.add_argument('--dropout_prob', type=float, default=0,
                    help='the dropout ratio')
parser.add_argument('--model', type=str, default="rn",
                    help='wrn, rn or dn')
parser.add_argument('--lr_scheduler', type=str, default="multistep",
                    help='lr scheduler')
parser.add_argument('--net_arg', type=int, default=2,
                    help='growth rate for densenet and widden factor for wrn')
parser.add_argument('--depth', type=int, default=32,
                    help='depth of densenet(resnet is fixed 32)')
parser.add_argument('--size_ratio', type=float, default=1,
                    help='use size_ratio of the training set')
parser.add_argument('--train_class', type=list, default=range(0,10),
                    help='training set is composed of these classes')
parser.add_argument('--test_class', type=list, default=range(0,10),
                    help='testing set is composed of these classes')
parser.add_argument('--corrupted_classes', type=list, default=[0,1,2,3,4],
                    help='takes effect when corruption_prob=partial_unif')
parser.add_argument('--clean_only', type=bool, default=False,
                    help='train only on the clean part of the dataset')
parser.add_argument('--relabel_thre', type=float, default=0,
                    help='relabel threshold ratio on weights(when 128 the threshold will be 1/128*x')
parser.add_argument('--reweight_precision_thre', type=list, default=[0.1,0.3,0.5,0.7],
                    help='experiment parameter, calculate the precision of low weight sample in the first x*dataset_size sample with the lowest weight')
best_prec1 = 0
log_file=str()

def loss_dist(model,precision_loader,device,noised_sample,results_dir,epoch,args):
    model.eval()
    with torch.no_grad():
        dataset_size=len(precision_loader.dataset)
        all_weights=np.zeros(dataset_size)
        all_loss=np.zeros(dataset_size)
        index=0
        for batch_idx, (inputs, targets) in enumerate(precision_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            cost_w = F.cross_entropy(outputs, targets, reduce=False)
            all_loss[index:index+len(targets)]=cost_w.cpu().view(-1)
            index+=len(targets)
    
    clean_s=list(np.argwhere(noised_sample==0).reshape(-1))
    noised_s=list(np.argwhere(noised_sample==1).reshape(-1))
    clean_loss=all_loss[clean_s]
    noised_loss=all_loss[noised_s]
    print("drawing loss dist...")
    plt.clf()
    plt.hist(all_loss,bins=500,histtype="step",color="b",label="all sample loss")
    plt.hist(clean_loss,bins=500,histtype="step",color="r",label="clean sample loss")
    plt.hist(noised_loss,bins=500,histtype="step",color="g",label="noised sample loss")
    plt.ylabel("freq")
    plt.xlabel("loss")
    plt.legend()
    plt.title("loss distribution")
    plt.savefig(os.path.join(results_dir,"loss_dist"+str(epoch)+".png"))
    plt.close()
    return 0

def precision(model,precision_loader,device,noised_sample,args):
    model.eval()
    with torch.no_grad():
        dataset_size=len(precision_loader.dataset)
        all_weights=np.zeros(dataset_size)
        all_loss=np.zeros(dataset_size)
        index=0
        for batch_idx, (inputs, targets) in enumerate(precision_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            cost_w = F.cross_entropy(outputs, targets, reduce=False)
            all_loss[index:index+len(targets)]=cost_w.cpu().view(-1)
            index+=len(targets)

        precision_idx=[]
        a=[]
        b=[]
        precision=[]
        for i,thre in enumerate(args.reweight_precision_thre):
            precision_idx.append(np.zeros(dataset_size))
            precision_idx[i][(-all_loss).argsort()[0:int(dataset_size*thre)]]=1
            a.append(precision_idx[i]+noised_sample)#0,1,2
            b.append(precision_idx[i]-noised_sample)#-1,0,1
            if precision_idx[i].sum()==0:
                precision.append(0)
            else:
                precision.append(((a[i]+10*b[i])==2).astype(int).sum()/precision_idx[i].sum())#so a should be 2 and b should be 0, which means that relabel_idx is 1 and noise_sample is 1
        #log_file.write('====> Average precision list: {}\n'.format(str(precision)))
        print('====> Average precision list: {}\n'.format(str(precision)))
        #log_file.write('====> Average precision list: {}\n'.format(str(precision)))
    return precision

def main():
    global args, best_prec1, log_file
    args = parser.parse_args()
    data_dir="../data"
    results_dir="../results"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if args.model=="rn":
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

    train_set=Corrupted_dataset(args,root=data_dir,download=True,transform=None,target_transform=None,train=True)
    test_set=Corrupted_dataset(args,root=data_dir,download=True,transform=None,target_transform=None,train=False)
    train_loader,noised_sample=train_set.get_data_loader()
    test_loader=test_set.get_data_loader()
    relabel_dataset=copy.deepcopy(train_loader.dataset)#the unchangable dataset
    relabel_loader=DataLoader(dataset=relabel_dataset,batch_size=args.batch_size,shuffle=False)

    #train_loader, test_loader = train_wrn.build_dataset(args,data_dir)
    #print(len(train_loader.dataset))
    #print(len(test_loader.dataset))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    loss_curve=[]
    test_loss_curve=[]
    acc_curve=[]
    test_acc_curve=[]
    tt=0
    precision_curve=[]
    for epoch in range(args.epochs):
        precision_curve.append(precision(model,relabel_loader,device,noised_sample,args))
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        log_file+='current lr {:.5e}'.format(optimizer.param_groups[0]['lr'])
        tt=train(train_loader, test_loader, model, criterion, optimizer, epoch,loss_curve,test_loss_curve,acc_curve,test_acc_curve,device,tt)
        if args.lr_scheduler!="cosine_m":
            lr_scheduler.step()
        if epoch%20==0:
            loss_dist(model,relabel_loader,device,noised_sample,results_dir,epoch,args)
    loss_dist(model,relabel_loader,device,noised_sample,results_dir,args.epochs-1,args)
    pc=np.array(precision_curve)
    for i, thre in enumerate(args.reweight_precision_thre):
        plt.plot(pc[:,i],label=str(thre)+"% sample")
    plt.legend()
    plt.title('precision curve')
    plt.savefig(os.path.join(results_dir,"precision"))
    plt.close()

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


def validate(test_loader, model, criterion,device):
    """
    Run evaluation
    """
    global best_prec1,log_file
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
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