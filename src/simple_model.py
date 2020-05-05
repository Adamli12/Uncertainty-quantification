import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import time
import math
from models.wideresnet import WideResNet
from models.densenet import DenseNet
from models.resnet import ResNet
#thanks to Javier Antoran's implementation of mc dropout https://github.com/JavierAntoran/Bayesian-Neural-Networks
#thanks to the code by Shu et. al (https://github.com/xjtushujun/meta-weight-net)
#thanks to the code by Ribeiro et. al(https://github.com/fabio-deep/Deep-Bayesian-Self-Training)
class Simple_model:
    def __init__(self,args,results_dir,device):

        self.args=args
        self.epoch_num = args.epochs
        self.log_interval = args.log_interval
        self.device = device
        self.n_classes = args.num_classes
        self.n_var = args.n_var
        if args.model=="dn":
            self.module = DenseNet(dropout_prob=args.dropout_prob,growth_rate=args.net_arg,bottleneck=4,init_channels=args.growth_rate*2,trans_ratio=0.5,n_classes=self.n_classes,depth=args.depth,n_dense_blocks=3,n_var=args.n_var)
        elif args.model=="rn":
            self.module = ResNet(depth=args.depth,num_classes=self.n_classes,n_var=self.n_var,block_name="BasicBlock")
        elif args.model=="wrn":
            self.module = WideResNet(args.depth, args.num_classes, args.n_var, args.net_arg, args.dropout_prob)
        self.crossentropy_criterion=nn.CrossEntropyLoss().to(self.device)
        #self.optimizer = optim.Adam(self.module.parameters(), lr=args.learning_rate)
        self.optimizer = optim.SGD(self.module.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay,nesterov=args.nesterov)
        if args.lr_scheduler=="multistep":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestones)
        elif args.lr_scheduler=="cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5,eta_min=4e-08)
        elif args.lr_scheduler=="cosine_m":
            pass
        else:
            assert False, "invalid lr scheduler"

        #self.results_dir = os.path.join(results_dir,time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
        settingstr=args.corruption_type+str(args.corruption_prob)+"_"+args.model+str(args.depth)+str(args.net_arg)+"_"+args.lr_scheduler+"_"+"simple"
        self.results_dir = os.path.join(results_dir,settingstr)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.best_test_acc=0
        self.log_file_dir = os.path.join(self.results_dir,"log.txt")
        self.log_file = str()
        self.log_file += "Hyperparameters:\n"+str(args)+"\n"
        total_params=0
        for x in filter(lambda p: p.requires_grad, self.module.parameters()):
            total_params += np.prod(x.data.numpy().shape)
        self.log_file += " == total parameters: " + str(total_params) + "\n"

        self.module = torch.nn.DataParallel(self.module).to(device)

    def compute_acc(self,label,pred):#preds: T*N*n_classes
        batch_size = label.size(0)
        _, pred = pred.topk(1, -1, True, True)
        correct = pred.squeeze().eq(label)
        res=correct.view(-1).float().sum()/batch_size
        return res

    def train(self,train_loader,test_loader):
        begin=time.perf_counter()
        self.module.train()
        loss_curve=[]
        test_loss_curve=[]
        acc_curve=[]
        test_acc_curve=[]
        tt=0
        for epoch in range(1,self.epoch_num+1):
            self.module.train()
            train_loss = 0.
            train_acc = 0.
            print('current lr {:.5e}'.format(self.optimizer.param_groups[0]['lr']))
            self.log_file+='current lr {:.5e}\n'.format(self.optimizer.param_groups[0]['lr'])
            for batch_idx, (data,label) in enumerate(train_loader):
                data = data.to(self.device)
                label = label.to(self.device)
                #data = data.float()
                #label = label.float()
                pred=self.module(data)
                ac_label = label
                loss = self.crossentropy_criterion(pred,label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss+=loss.item()*len(data)
                loss_curve.append(loss.item())
                acc=self.compute_acc(ac_label,pred.detach())
                train_acc+=acc.item()*len(data)
                acc_curve.append(acc.item())
                if self.args.lr_scheduler=="cosine_m":
                    dt = math.pi/float(self.args.epochs)
                    tt += float(dt)/(len(train_loader.dataset)/float(self.args.batch_size))
                    if tt >= math.pi - 0.05:
                        tt = math.pi - 0.05
                    curT = math.pi/2.0 + tt
                    new_lr = self.args.learning_rate * (1.0 + math.sin(curT))/2.0    # lr_min = 0, lr_max = lr
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                if batch_idx % self.log_interval == 0:
                    self.log_file+='Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tAcc: {:.4f}\n'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx * len(data)/ len(train_loader.dataset),
                        loss.item(),acc.item())
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tAcc: {:.4f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx * len(data)/ len(train_loader.dataset),
                        loss.item(),acc.item()))
            self.log_file+='====> Train Epoch: {} Average loss: {:.4f} Average acc: {:.4f}\n'.format(
                epoch, train_loss/len(train_loader.dataset), train_acc/len(train_loader.dataset))
            print('====> Train Epoch: {} Average loss: {:.4f} Average acc: {:.4f}'.format(
                epoch, train_loss/len(train_loader.dataset), train_acc/len(train_loader.dataset)))
            test_res=self.test(test_loader)
            test_loss_curve.append(test_res[0])
            test_acc_curve.append(test_res[1])
            if max(test_acc_curve)==test_res[1] or len(test_acc_curve)==0:#early stop
                self.save()
                self.best_test_acc=test_res[1]
            if self.args.lr_scheduler!="cosine_m":
                self.lr_scheduler.step()
        plt.clf()
        plt.plot(loss_curve)
        plt.yscale('log')
        plt.title('training loss curve')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir,"train_loss"))
        plt.clf()
        plt.plot(test_loss_curve)
        plt.yscale('log')
        plt.title('testing loss curve')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir,"test_loss"))
        plt.clf()
        plt.plot(acc_curve)
        plt.title('training acc curve')
        plt.savefig(os.path.join(self.results_dir,"train_acc"))
        plt.clf()
        plt.plot(test_acc_curve)
        plt.title('testing acc curve')
        plt.savefig(os.path.join(self.results_dir,"test_acc"))
        end=time.perf_counter()
        self.log_file='simple model total training time: {:.4f}\n'.format(end-begin)+self.log_file
        self.log_file="best test acc: "+str(self.best_test_acc)+"\n"+self.log_file
        with open(self.log_file_dir,"w") as f:
            f.write(self.log_file)
        self.log_file=str()

    def test(self, test_loader):
        self.module.eval()
        eval_loss = 0.
        eval_acc = 0.
        with torch.no_grad():
            for data,label in test_loader:
                data = data.to(self.device)
                label = label.to(self.device)
                #data = data.float()
                #label = label.float()
                ac_label = label
                pred=self.module(data)
                loss = self.crossentropy_criterion(pred,label)
                eval_loss+=loss.item()*len(data)
                acc=self.compute_acc(ac_label,pred)
                eval_acc+=acc.item()*len(data)
        self.log_file+='====> Test Average loss: {:.4f} Average acc: {:.4f}\n'.format(eval_loss/len(test_loader.dataset), eval_acc/len(test_loader.dataset))
        print('====> Test Average loss: {:.4f} Average acc: {:.4f}'.format(eval_loss/len(test_loader.dataset), eval_acc/len(test_loader.dataset)))
        return eval_loss/len(test_loader.dataset), eval_acc/len(test_loader.dataset)


    def save(self,path="trained_model.pth"):
        torch.save(self.module.state_dict(), os.path.join(self.results_dir,path))
        return 0

    def load(self,path="trained_model.pth"):
        self.module.load_state_dict(torch.load(os.path.join(self.results_dir,path)))