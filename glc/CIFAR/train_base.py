# -*- coding: utf-8 -*-

import argparse
import os
import time
import math
import json
import torch
from torch.autograd import Variable as V
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import wideresnet as wrn
import numpy as np
from load_corrupted_data import CIFAR10, CIFAR100
from PIL import Image
import socket


# note: nosgdr, schedule, and epochs are highly related settings

parser = argparse.ArgumentParser(description='Trains WideResNet on CIFAR',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Positional arguments
parser.add_argument('--data_path', type=str, default="../data", help='Root for the Cifar dataset.')
parser.add_argument('--dataset', type=str, default='cifar10',
    help='Choose between CIFAR-10, CIFAR-100.')
# Optimization options
parser.add_argument('--lr_scheduler', default="cosine_m", help='learning rate scheduler.')
parser.add_argument('--epochs', '-e', type=int, default=75, help='Number of epochs to train.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--gold_fraction', '-gf', type=float, default=0, help='What fraction of the data should be trusted?')
parser.add_argument('--corruption_prob', '-cprob', type=float, default=0.6, help='The label corruption probability.')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif', help='Type of corruption ("unif" or "flip").')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--test_bs', type=int, default=128)
parser.add_argument('--milestones', type=list, default=[55, 65],
                    help='Decrease learning rate at these epochs. Use when lr_scheduler=multistep.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Architecture
parser.add_argument('--depth', default=28, type=int, help='total number of depth (default: 28)')
parser.add_argument('--net_arg', default=10, type=int, help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--nonlinearity', type=str, default='relu', help='Nonlinearity (relu, elu, gelu).')
# Acceleration
parser.add_argument('--ngpu', type=int, default=4, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
# i/o
parser.add_argument('--result_dir', type=str, default='../results', help='Result folder.')
# random seed
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()


np.random.seed(args.seed)

print(args)
print()

# Init logger
if not os.path.isdir(args.result_dir):
    os.makedirs(args.result_dir)
settingstr=args.corruption_type+str(args.corruption_prob)+"_"+str(args.depth)+str(args.net_arg)+"_"+args.lr_scheduler+"_"+str(args.droprate)+"_"+"glc"
results_dir = os.path.join(args.result_dir,settingstr)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
log = open(os.path.join(results_dir,'log.txt'), 'w')
state = {k: v for k, v in args._get_kwargs()}
state['tt'] = 0      # SGDR variable
state['init_learning_rate'] = args.learning_rate
log.write(json.dumps(state) + '\n')

# Init dataset
if not os.path.isdir(args.data_path):
    os.makedirs(args.data_path)

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
     transforms.Normalize(mean, std)])
test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)])

if args.dataset == 'cifar10':
    train_data_gold = CIFAR10(
        args.data_path, True, True, args.gold_fraction, args.corruption_prob, args.corruption_type,
        transform=train_transform, download=True)
    train_data_silver = CIFAR10(
        args.data_path, True, False, args.gold_fraction, args.corruption_prob, args.corruption_type,
        transform=train_transform, download=True, shuffle_indices=train_data_gold.shuffle_indices, seed=args.seed)
    train_data_gold_deterministic = CIFAR10(
        args.data_path, True, True, args.gold_fraction, args.corruption_prob, args.corruption_type,
        transform=test_transform, download=True, shuffle_indices=train_data_gold.shuffle_indices)
    test_data = CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 10

elif args.dataset == 'cifar100':
    train_data_gold = CIFAR100(
        args.data_path, True, True, args.gold_fraction, args.corruption_prob, args.corruption_type,
        transform=train_transform, download=True)
    train_data_silver = CIFAR100(
        args.data_path, True, False, args.gold_fraction, args.corruption_prob, args.corruption_type,
        transform=train_transform, download=True, shuffle_indices=train_data_gold.shuffle_indices, seed=args.seed)
    train_data_gold_deterministic = CIFAR100(
        args.data_path, True, True, args.gold_fraction, args.corruption_prob, args.corruption_type,
        transform=test_transform, download=True, shuffle_indices=train_data_gold.shuffle_indices)
    test_data = CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 100


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, transform):
        # assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data_tensor[index], self.target_tensor[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return 50000

train_silver_loader = torch.utils.data.DataLoader(
    train_data_silver, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)
"""train_gold_deterministic_loader = torch.utils.data.DataLoader(
    train_data_gold_deterministic, batch_size=args.test_bs, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)
train_all_loader = torch.utils.data.DataLoader(
    TensorDataset(np.vstack((train_data_gold.train_data, train_data_silver.train_data)),
                  torch.from_numpy(np.array(train_data_gold.train_labels + train_data_silver.train_labels)),
                  train_transform),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)"""
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)

# Init model, criterion, and optimizer
net = wrn.WideResNet(args.depth, num_classes, args.net_arg, dropRate=args.droprate)
print(net)

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()

torch.manual_seed(args.seed)
if args.ngpu > 0:
    torch.cuda.manual_seed(args.seed)


optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'], weight_decay=state['decay'], nesterov=True)

if args.lr_scheduler=="multistep":
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones)
elif args.lr_scheduler=="cosine":
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5,eta_min=4e-08)
elif args.lr_scheduler=="cosine_m":
    pass
else:
    assert False, "invalid lr scheduler"

start_epoch = 0

cudnn.benchmark = True  # fire on all cylinders


def train_phase1():
    net.train()     # enter train mode
    loss_avg = 0.0
    for batch_idx, (data, target) in enumerate(train_silver_loader):
        data, target = V(data.cuda()), V(target.cuda())

        # forward
        output = net(data)

        # backward
        optimizer.zero_grad()
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + loss.data.item()* 0.2

        if args.lr_scheduler == "cosine_m":    # Use a cyclic learning rate
            dt = math.pi/float(args.epochs)
            state['tt'] += float(dt)/(len(train_silver_loader.dataset)/float(args.batch_size))
            if state['tt'] >= math.pi - 0.05:
                state['tt'] = math.pi - 0.05
            curT = math.pi/2.0 + state['tt']
            new_lr = args.learning_rate * (1.0 + math.sin(curT))/2.0    # lr_min = 0, lr_max = lr
            state['learning_rate'] = new_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['learning_rate']

    state['train_loss'] = loss_avg


# test function (forward only)
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = V(data.cuda()), V(target.cuda())

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum()

            # test loss average
            loss_avg += loss.data.item()

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = (float(correct) / float(len(test_loader.dataset)))

best_acc=0
# Main loop
for epoch in range(start_epoch, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()
    train_phase1()
    print('Epoch', epoch, '| Time Spent:', round(time.time() - begin_epoch, 2))

    test()
    torch.save(net.state_dict(), os.path.join(results_dir,'model.pytorch'))
    
    log.write('%s\n' % json.dumps(state))
    log.flush()
    print(state)
    best_acc=max(state["test_accuracy"],best_acc)
    if args.lr_scheduler!="cosine_m":
        lr_scheduler.step()
log.write("best acc: %s\n" % str(best_acc))