# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
# import sklearn.metrics as sm
# import pandas as pd
# import sklearn.metrics as sm
import random
import numpy as np

import copy
from torch.utils.data import DataLoader

# from wideresnet import WideResNet, VNet
from models.MWresnet import ResNet32,VNet
from data_loader import Corrupted_dataset

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--num_classes', type=int, default=10,
                    help='how many classes in dataset(CIFAR 10 or CIFAR 100)')
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
parser.add_argument('--corruption_prob', type=float, default=0.6,
                    help='label noise')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif',
                    help='Type of corruption ("unif" or "flip" or "flip2").')
parser.add_argument('--num_meta', type=int, default=0)
parser.add_argument('--epochs', default=120, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='ResNet-32', type=str,
                    help='name of experiment')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
parser.add_argument('--do_meta', type=bool, default=False, help='do meta or not')
parser.add_argument('--reweight_precision_thre', type=list, default=[0.1,0.3,0.5,0.7],
                    help='experiment parameter, calculate the precision of low weight sample in the first x*dataset_size sample with the lowest weight')
parser.add_argument('--visible_gpu', type=str, default="0,1,2,3",
                    help='visible gpu, in form "a,b,c,d"')
parser.set_defaults(augment=True)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
print("Let's use", torch.cuda.device_count(), "GPUs")
device = torch.device("cuda:0" if args.cuda else "cpu")
data_dir="../data"
results_dir="../results"
time_dir = os.path.join(results_dir,time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
results_dir = os.path.join(results_dir,time_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
print()
print(args)

def build_dataset():
    """
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    if args.augment:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'cifar10':
        train_data_meta = CIFAR10(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        train_data = CIFAR10(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR10(root='../data', train=False, transform=test_transform, download=True)


    elif args.dataset == 'cifar100':
        train_data_meta = CIFAR100(
            root='../data', train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True)
        train_data = CIFAR100(
            root='../data', train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, seed=args.seed)
        test_data = CIFAR100(root='../data', train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    train_meta_loader = torch.utils.data.DataLoader(
        train_data_meta, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True)"""
    train_set=Corrupted_dataset(args,root=data_dir,download=True,transform=None,target_transform=None,train=True,meta=False,do_meta=True)
    train_meta_set=Corrupted_dataset(args,root=data_dir,download=True,transform=None,target_transform=None,train=True,meta=True,do_meta=True)
    test_set=Corrupted_dataset(args,root=data_dir,download=True,transform=None,target_transform=None,train=False)
    train_loader,noised_sample=train_set.get_data_loader()
    train_meta_loader=train_meta_set.get_data_loader()
    test_loader=test_set.get_data_loader()
    

    return train_loader, train_meta_loader, test_loader, noised_sample


def build_model():
    model = ResNet32(args.dataset == 'cifar10' and 10 or 100)

    if torch.cuda.is_available():
        model.cuda()
        model = torch.nn.DataParallel(model).to(device)
        torch.backends.cudnn.benchmark = True

    return model

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


def adjust_learning_rate(optimizer, epochs):
    lr = args.lr * ((0.1 ** int(epochs >= 80)) * (0.1 ** int(epochs >= 100)))  # For WRN-28-10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss +=F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    log_file.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy

def relabel(epoch):
    model.eval()
    with torch.no_grad():
        dataset_size=len(relabel_loader.dataset)
        all_weights=np.zeros(dataset_size)
        all_loss=np.zeros(dataset_size)
        index=0
        for batch_idx, (inputs, targets) in enumerate(relabel_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            cost_w = F.cross_entropy(outputs, targets, reduce=False)
            cost_v = torch.reshape(cost_w, (len(cost_w), 1))
            w_new = vnet(cost_v)
            all_weights[index:index+len(targets)]=w_new.cpu().view(-1)
            all_loss[index:index+len(targets)]=cost_v.cpu().view(-1)
            index+=len(targets)
            if batch_idx%100==0:
                if args.do_meta:
                    print(w_new.cpu().view(-1))
                    log_file.write((str(w_new.cpu().view(-1))+"\n"))
                else:
                    print(cost_v.cpu().view(-1))
                    log_file.write((str(cost_v.cpu().view(-1))+"\n"))

        precision_idx=[]
        a=[]
        b=[]
        precision=[]
        for i,thre in enumerate(args.reweight_precision_thre):
            precision_idx.append(np.zeros(dataset_size))
            if args.do_meta:
                precision_idx[i][all_weights.argsort()[0:int(dataset_size*thre)]]=1
            else:
                precision_idx[i][(-all_loss).argsort()[0:int(dataset_size*thre)]]=1
            a.append(precision_idx[i]+noised_sample)#0,1,2
            b.append(precision_idx[i]-noised_sample)#-1,0,1
            if precision_idx[i].sum()==0:
                precision.append(0)
            else:
                precision.append(((a[i]+10*b[i])==2).astype(int).sum()/precision_idx[i].sum())#so a should be 2 and b should be 0, which means that relabel_idx is 1 and noise_sample is 1
        log_file.write('====> Average precision list: {}\n'.format(str(precision)))
        print('====> Average precision list: {}\n'.format(str(precision)))
    return precision

def train(train_loader,train_meta_loader,model,vnet,optimizer_model,optimizer_vnet,epoch):
    print('\nEpoch: %d' % epoch)
    log_file.write('\nEpoch: %d' % epoch)
    train_loss = 0
    meta_loss = 0
    train_meta_loader_iter = iter(train_meta_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model.train()
        inputs, targets = inputs.to(device), targets.to(device)
        if args.do_meta:
            meta_model = build_model().cuda()
            meta_model.load_state_dict(model.state_dict())
            outputs = meta_model(inputs)

            cost = F.cross_entropy(outputs, targets, reduce=False)
            cost_v = torch.reshape(cost, (len(cost), 1))
            v_lambda = vnet(cost_v.data)
            l_f_meta = torch.sum(cost_v * v_lambda)/len(cost_v)
            meta_model.zero_grad()
            grads = torch.autograd.grad(l_f_meta, (meta_model.module.params()), create_graph=True)
            meta_lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 100)))   # For ResNet32#这里是为了模仿Adam优化器吗，metalr一直固定0.001
            meta_model.module.update_params(lr_inner=meta_lr, source_params=grads)
            del grads

            try:
                inputs_val, targets_val = next(train_meta_loader_iter)
            except StopIteration:
                train_meta_loader_iter = iter(train_meta_loader)
                inputs_val, targets_val = next(train_meta_loader_iter)
            inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
            y_g_hat = meta_model(inputs_val)
            l_g_meta = F.cross_entropy(y_g_hat, targets_val.long())
            prec_meta = accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0]


            optimizer_vnet.zero_grad()
            l_g_meta.backward()
            optimizer_vnet.step()
            
        outputs = model(inputs)
        cost_w = F.cross_entropy(outputs, targets, reduce=False)
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))
        prec_train = accuracy(outputs.data, targets.data, topk=(1,))[0]

        if args.do_meta:
            with torch.no_grad():
                w_new = vnet(cost_v)
            loss = torch.sum(cost_v * w_new)/len(cost_v)
        else:
            loss = torch.sum(cost_v)/len(cost_v)
            prec_meta=0

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()


        train_loss += loss.item()
        #meta_loss += l_g_meta.item()
        meta_loss = 0


        if (batch_idx + 1) % 50 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % (
                      (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset)/args.batch_size, (train_loss / (batch_idx + 1)),
                      (meta_loss / (batch_idx + 1)), prec_train, prec_meta))
            log_file.write('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % (
                      (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset)/args.batch_size, (train_loss / (batch_idx + 1)),
                      (meta_loss / (batch_idx + 1)), prec_train, prec_meta))

def weight_dist(epoch):
    model.eval()
    dataset_size=len(relabel_loader.dataset)
    all_weights=np.zeros(dataset_size)
    all_loss=np.zeros(dataset_size)
    with torch.no_grad():
        index=0
        for batch_idx, (inputs, targets) in enumerate(relabel_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            cost_w = F.cross_entropy(outputs, targets, reduce=False)
            cost_v = torch.reshape(cost_w, (len(cost_w), 1))
            w_new = vnet(cost_v)
            all_weights[index:index+len(targets)]=w_new.cpu().view(-1)
            all_loss[index:index+len(targets)]=cost_v.cpu().view(-1)
            index+=len(targets)
    clean_s=list(np.argwhere(noised_sample==0).reshape(-1))
    noised_s=list(np.argwhere(noised_sample==1).reshape(-1))
    if args.do_meta:
        clean_weights=all_weights[clean_s]
        noised_weights=all_weights[noised_s]
        print("drawing weight dist...")
        log_file.write("drawing weight dist..."+"\n")
        plt.clf()
        plt.hist(all_weights,bins=500,histtype="step",color="b",label="all sample weights")
        plt.hist(clean_weights,bins=500,histtype="step",color="r",label="clean sample weights")
        plt.hist(noised_weights,bins=500,histtype="step",color="g",label="noised sample weights")
        plt.ylabel("freq")
        plt.xlabel("weight")
        plt.legend()
        plt.title("weight distribution")
        plt.savefig(os.path.join(results_dir,"weight_dist"+str(epoch)+".png"))
        plt.close()
    else:
        clean_loss=all_loss[clean_s]
        noised_loss=all_loss[noised_s]
        print("drawing loss dist...")
        log_file.write("drawing loss dist..."+"\n")
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


train_loader, train_meta_loader, test_loader, noised_sample = build_dataset()
# create model
relabel_dataset=copy.deepcopy(train_loader.dataset)#the unchangable dataset
relabel_loader=DataLoader(dataset=relabel_dataset,batch_size=args.batch_size,shuffle=False)
model = build_model()
vnet = VNet(1, 100, 1)
vnet = torch.nn.DataParallel(vnet).to(device)
log_file_dir = os.path.join(results_dir,"log.txt")
log_file = open(log_file_dir,"w",1)
log_file.write("Hyperparameters:\n"+str(args)+"\n")
if args.dataset == 'cifar10':
    num_classes = 10
if args.dataset == 'cifar100':
    num_classes = 100


optimizer_model = torch.optim.SGD(model.module.params(), args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay)
optimizer_vnet = torch.optim.Adam(vnet.module.params(), 1e-3,
                             weight_decay=1e-4)

def main():
    best_acc = 0
    precision_curve=[]
    for epoch in range(args.epochs):
        precision_curve.append(relabel(epoch))
        adjust_learning_rate(optimizer_model, epoch)
        train(train_loader,train_meta_loader,model,vnet,optimizer_model,optimizer_vnet,epoch)
        test_acc = test(model=model, test_loader=test_loader)
        if test_acc >= best_acc:
            best_acc = test_acc
        if epoch%20==0:
            weight_dist(epoch)
    weight_dist(args.epochs-1)
    pc=np.array(precision_curve)
    for i, thre in enumerate(args.reweight_precision_thre):
        plt.plot(pc[:,i],label=str(thre)+"% sample")
    plt.legend()
    plt.title('precision curve')
    plt.savefig(os.path.join(results_dir,"precision"))
    plt.close()
    print('best accuracy:', best_acc)
    log_file.write('best accuracy:%d'%best_acc)


if __name__ == '__main__':
    main()
