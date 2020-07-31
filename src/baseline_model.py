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
import copy
from torch.utils.data import DataLoader
from models.densenet import DenseNet
from models.resnet import ResNet
from models.wideresnet import WideResNet

#thanks to Javier Antoran's implementation of mc dropout https://github.com/JavierAntoran/Bayesian-Neural-Networks
#thanks to the code by Shu et. al (https://github.com/xjtushujun/meta-weight-net)
#thanks to the code by Ribeiro et. al(https://github.com/fabio-deep/Deep-Bayesian-Self-Training)

class Simple_CNN_module(torch.nn.Module):
    """
    Simple CNN
    convolution-relu-convolution-relu-max pooling-dropout-dense-relu-dropout-dense-softmax, with 32 convolution kernels, 4x4 kernel size,2x2 pooling, dense layer with 128 units, and dropout probabilities 0.25 and 0.5
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25,inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25,inplace=True),
            nn.Flatten(),
            nn.Linear(4096,512),
            nn.ReLU(),
            nn.Dropout2d(0.5,inplace=True),
            nn.Linear(512,10),
        )

    def forward(self, input):
        return self.net(input)

class Baseline_model:
    def __init__(self,args,results_dir,device,train_loader,test_loader,noised_sample):

        self.args=args
        self.loss_type = args.loss_type
        self.dropout_prob = args.dropout_prob
        self.log_interval = args.log_interval
        self.device = device
        self.n_classes = args.num_classes
        self.mc_n_sample = args.MC_n_samples
        self.n_var = args.n_var
        self.reweight = args.reweight
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.noised_sample=noised_sample
        self.relabel_dataset=copy.deepcopy(self.train_loader.dataset)#the unchangable dataset
        self.current_dataset=copy.deepcopy(self.train_loader.dataset)#changable dataset
        self.dataset_size=len(self.relabel_dataset)
        self.relabel_loader=DataLoader(dataset=self.relabel_dataset,batch_size=self.args.batch_size,shuffle=False)

        if args.model=="dn":
            self.module = DenseNet(dropout_prob=args.dropout_prob,growth_rate=args.net_arg,bottleneck=4,init_channels=args.growth_rate*2,trans_ratio=0.5,n_classes=self.n_classes,depth=args.depth,n_dense_blocks=3,n_var=args.n_var)
        elif args.model=="rn":
            self.module = ResNet(depth=args.depth,num_classes=self.n_classes,n_var=self.n_var,block_name="BasicBlock")
        elif args.model=="wrn":
            self.module = WideResNet(args.depth, args.num_classes, args.n_var, args.net_arg, args.dropout_prob)
        self.nll_criterion=nn.NLLLoss(reduction="none").to(self.device)
        self.crossentropy_criterion=nn.CrossEntropyLoss(reduction="none").to(self.device)

        self.time_dir = os.path.join(results_dir,time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
        #settingstr=args.corruption_type+str(args.corruption_prob)+"_"+args.model+str(args.depth)+str(args.net_arg)+"_"+args.lr_scheduler+"_"+"base"
        self.settingstr=args.corruption_type+str(args.corruption_prob)+"_"+args.model+str(args.depth)+str(args.net_arg)+"_"+str(args.num_classes)+'_'+args.loss_type+"_"+str(args.size_ratio)+'_'+str(args.train_class)+'_'+str(args.test_class)+"_"+args.lr_scheduler+"_"+args.reweight+'_'+args.reweight_norm+'_'+args.reweight_function+'_'+str(args.dropout_prob)+"_"+str(args.p0epochs)+"_"+str(args.p1epochs)+"_"+str(args.clean_only)+"_"+args.relabel_f+"_"+str(args.relabel_thre)
        self.results_dir = os.path.join(results_dir,self.time_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.best_test_acc=0
        self.last_test_acc=0
        self.log_file_dir = os.path.join(self.results_dir,"log.txt")
        self.log_file = open(self.log_file_dir,"w",1)
        self.log_file.write("Hyperparameters:\n"+str(args)+"\n")
        total_params=0
        for x in filter(lambda p: p.requires_grad, self.module.parameters()):
            total_params += np.prod(x.data.numpy().shape)
        self.log_file.write(" == total parameters: " + str(total_params) + "\n")

        self.module = torch.nn.DataParallel(self.module).to(device)
        

    def heteroscedastic_crossentropy(self,y_true,logits_log_var,mode):
        def monte_carlo(T_h, gaussian):#gaussian:T*N*K
            sample = gaussian.rsample([T_h]) # rsample for all logits, sample:T_h*T*N*K
            T_softmax = torch.softmax(sample,-1)
            return T_softmax.mean((0,1))#N*K
        #logits_log_var: T*N*11
        n_classes = self.n_classes #10
        if mode=="multi":
            std = torch.sqrt(torch.exp(logits_log_var[:,:,n_classes:]))#T*N*1
            logits=logits_log_var[:,:,:n_classes]#T*N*K(in CIFAR10 K=10)
        if mode=="single":
            std = torch.sqrt(torch.exp(logits_log_var[[0],:,n_classes:]))#1*N*1
            logits=logits_log_var[[0],:,:n_classes]#1*N*K(in CIFAR10 K=10)
        if std.shape[-1]==1:
            gaussian=torch.distributions.Normal(loc=logits, scale=std.expand_as(logits))#T*N*K
        else:
            gaussian=torch.distributions.Normal(loc=logits, scale=std)#T*N*K

        # get T_h softmax monte carlo simulations
        y_hat = monte_carlo(T_h=100, # number of simulations
            gaussian=gaussian)

        y_hat = torch.clamp(y_hat, 1e-11, 1 - 1e-11) # prevent nans
        nll = self.nll_criterion(torch.log(y_hat),y_true)
        return nll

    def calculate_uncertainty(self,MC_preds):
        def aleatoric(MC_log_vars):#T*N*1
            '''aleatoric uncertainty per datapoint'''
            # (N,)
            return torch.exp(MC_log_vars).mean((0,-1))# mean uncertainty of all MC sample log variances

        def epistemic(MC_logits):#T*N*K
            '''epistemic uncertainty per datapoint'''
            # (T, N, K)
            y_hat = torch.softmax(MC_logits,-1).mean(0) # mean softmax of all MC sample logits
            y_hat = torch.clamp(y_hat, 1e-11, 1 - 1e-11) # prevent nans
            # (N,)
            return -torch.sum(y_hat * torch.log(y_hat), dim=1)

        def KG_epistemic(MC_logits):
            logit_vars = (torch.std(MC_logits,0)**2).mean(-1)
            return logit_vars

        def variational_ratio(MC_logits):
            y_hat = torch.softmax(MC_logits,-1).mean(0) # mean softmax of all MC sample logits
            y_hat = torch.clamp(y_hat, 1e-11, 1 - 1e-11) # prevent nans,(N,K)
            y_max,_ = torch.max(y_hat, -1)
            # (N,)
            #print(y_max)
            """nan=torch.isnan(y_max)
            for i in range(len(y_max)):
                if nan[i]==True:
                    print(MC_logits)
                assert nan[i]==False"""
            return torch.sub(1,y_max)

        MC_preds_avatar=MC_preds.detach()#T*N*(K+1)
        return aleatoric(MC_preds_avatar[...,self.n_classes:]), epistemic(MC_preds_avatar[...,:self.n_classes]), KG_epistemic(MC_preds_avatar[...,:self.n_classes]), variational_ratio(MC_preds_avatar[...,:self.n_classes])

    def sample_predict(self, x, T):#x: N*C*H*W
        ''' ---------------------- MONTE CARLO DROPOUT --------------------------'''
        #predictions: T*N*(n_classes+n_var)
        for i in range(T):
            pred=self.module(x)
            if i==0:
                predictions = torch.zeros((T,x.shape[0],self.n_classes+self.n_var),device=x.device)
            predictions[i]=pred
        return predictions

    def compute_weights(self,aleatorics,epistemics):
        overalls=aleatorics+epistemics
        if self.args.reweight_function=="exp":
            weights=1/torch.exp(overalls)
        if self.args.reweight_function=="inverse":
            weights=1/overalls
        if self.args.reweight_norm=="batch":
            weights=torch.div(weights,(torch.sum(weights)))
        elif self.args.reweight_norm=="sig_m":
            weights=torch.sigmoid(weights)
            weights=weights*2-1
            weights=weights/len(weights)
        elif self.args.reweight_norm=="none":#DBST
            weights=weights/len(weights)
        return weights

    def compute_acc(self,label,pred):#preds: T*N*n_classes
        batch_size = label.size(0)
        probs = torch.softmax(pred,-1)
        pred=probs.mean(0)#N*K
        _, pred = pred.topk(1, -1, True, True)
        correct = pred.squeeze().eq(label)
        res=correct.view(-1).float().sum()/batch_size
        return res

    def weight_dist_comp(self,epoch,phase):
        self.module.eval()
        all_weights=np.zeros(self.dataset_size)
        all_var_weights=np.zeros(self.dataset_size)
        #all_norms=np.zeros(self.dataset_size)
        index=0
        for batch_idx, (data,label) in enumerate(self.relabel_loader):
            data = data.to(self.device)
            label = label.to(self.device)
            #data = data.float()
            #label = label.float()
            if self.dropout_prob==0:
                pred = self.sample_predict(data,1)#MC_preds
            else:
                pred = self.sample_predict(data,self.mc_n_sample)#MC_preds
            aleatorics,epistemics,logits_var,variational_ratio=self.calculate_uncertainty(pred)
            """if self.reweight=="none":
                weights=torch.ones(pred.shape[1])
                weights/=len(data)"""
            if self.reweight=="what":
                weights = self.compute_weights(epistemics,logits_var)#这weight是归一化的
            elif self.reweight=="none":
                weights = F.cross_entropy(pred.detach()[:,:,:self.n_classes].mean(0),label,reduce=False)
                #norms = torch.norm(pred.detach()[:,:,:self.n_classes].squeeze(),dim=1)
                var_weights = self.compute_weights(variational_ratio,0)#这weight是归一化的
            elif self.reweight=="dbst":
                weights = self.compute_weights(aleatorics,epistemics)#这weight是归一化的
            elif self.reweight=="entropy":
                weights = self.compute_weights(epistemics,0)#这weight是归一化的
            elif self.reweight=="variance":
                weights = self.compute_weights(logits_var,0)#这weight是归一化的
            elif self.reweight=="a_pred":
                weights = self.compute_weights(aleatorics,0)#这weight是归一化的
            elif self.reweight=="var_ratio":
                weights = self.compute_weights(variational_ratio,0)#这weight是归一化的
            all_weights[index:index+len(label)]=weights.cpu()
            all_var_weights[index:index+len(label)]=var_weights.cpu()
            #all_norms[index:index+len(label)]=norms.cpu()
            index+=len(label)
        clean_s=list(np.argwhere(self.noised_sample==0).reshape(-1))
        noised_s=list(np.argwhere(self.noised_sample==1).reshape(-1))
        clean_weights=all_weights[clean_s]
        noised_weights=all_weights[noised_s]
        print("drawing weight dist...")
        self.log_file.write("drawing weight dist..."+"\n")
        plt.clf()
        plt.close()
        plt.hist(all_weights,bins=500,histtype="step",color="b",label="all sample weights")
        plt.hist(clean_weights,bins=500,histtype="step",color="r",label="clean sample weights")
        plt.hist(noised_weights,bins=500,histtype="step",color="g",label="noised sample weights")
        plt.ylabel("freq")
        plt.xlabel("weight")
        plt.legend()
        plt.title("weight distribution")
        plt.savefig(os.path.join(self.results_dir,"weight_dist"+'p'+str(phase)+'_'+str(epoch)+".png"))
        plt.close()
        if self.reweight=="none":
            clean_var_weights=all_var_weights[clean_s]
            noised_var_weights=all_var_weights[noised_s]
            print("drawing varratio weight dist...")
            self.log_file.write("drawing varratio weight dist..."+"\n")
            plt.clf()
            plt.close()
            plt.hist(all_var_weights,bins=500,histtype="step",color="b",label="all sample")
            plt.hist(clean_var_weights,bins=500,histtype="step",color="r",label="clean sample")
            plt.hist(noised_var_weights,bins=500,histtype="step",color="g",label="noised sample")
            plt.ylabel("freq")
            plt.xlabel("weight")
            plt.legend()
            plt.title("varratio weight distribution")
            plt.savefig(os.path.join(self.results_dir,"var_weight_dist"+'p'+str(phase)+'_'+str(epoch)+".png"))
            plt.close()
            a=[]
            b=[]
            va=[]
            vb=[]
            precision=[]
            vprecision=[]
            common_sample_rate=[]
            precision_idx=[]
            vprecision_idx=[]
            for i,thre in enumerate(self.args.reweight_precision_thre):
                precision_idx.append(np.zeros(self.dataset_size))
                vprecision_idx.append(np.zeros(self.dataset_size))
                precision_idx[i][(-all_weights).argsort()[0:int(self.dataset_size*thre)]]=1
                vprecision_idx[i][all_var_weights.argsort()[0:int(self.dataset_size*thre)]]=1
                if i == 0:
                    self.log_file.write(str(np.argwhere(precision_idx[i]==1)))
                    self.log_file.write(str(np.argwhere(vprecision_idx[i]==1)))
                a.append(precision_idx[i]+self.noised_sample)#0,1,2
                b.append(precision_idx[i]-self.noised_sample)#-1,0,1
                va.append(vprecision_idx[i]+self.noised_sample)#0,1,2
                vb.append(vprecision_idx[i]-self.noised_sample)#-1,0,1
                if precision_idx[i].sum()==0:
                    precision.append(0)
                else:
                    precision.append(((a[i]+10*b[i])==2).astype(int).sum()/precision_idx[i].sum())#so a should be 2 and b should be 0, which means that relabel_idx is 1 and noise_sample is 1
                if vprecision_idx[i].sum()==0:
                    vprecision.append(0)
                else:
                    vprecision.append(((va[i]+10*vb[i])==2).astype(int).sum()/vprecision_idx[i].sum())#so a should be 2 and b should be 0, which means that relabel_idx is 1 and noise_sample is 1
                c=precision_idx[i]+vprecision_idx[i]
                d=precision_idx[i]-vprecision_idx[i]
                common_sample_rate.append(((c+10*d)==2).astype(int).sum()/int(self.dataset_size*thre))
            self.log_file.write('====> Average precision list: {}\n'.format(str(precision)))
            print('====> Average precision list: {}\n'.format(str(precision)))
            self.log_file.write('====> Average var_precision list: {}\n'.format(str(vprecision)))
            print('====> Average var_precision list: {}\n'.format(str(vprecision)))
            self.log_file.write('====> common sample rate list: {}\n'.format(str(common_sample_rate)))
            print('====> common sampe rate list: {}\n'.format(str(common_sample_rate)))
        return 0

    def weight_dist(self,epoch,phase):
        self.module.eval()
        all_weights=np.zeros(self.dataset_size)
        #all_norms=np.zeros(self.dataset_size)
        index=0
        for batch_idx, (data,label) in enumerate(self.relabel_loader):
            data = data.to(self.device)
            label = label.to(self.device)
            #data = data.float()
            #label = label.float()
            if self.dropout_prob==0:
                pred = self.sample_predict(data,1)#MC_preds
            else:
                pred = self.sample_predict(data,self.mc_n_sample)#MC_preds
            aleatorics,epistemics,logits_var,variational_ratio=self.calculate_uncertainty(pred)
            """if self.reweight=="none":
                weights=torch.ones(pred.shape[1])
                weights/=len(data)"""
            if self.reweight=="what":
                weights = self.compute_weights(epistemics,logits_var)#这weight是归一化的
            elif self.reweight=="none":
                weights = F.cross_entropy(pred.detach()[:,:,:self.n_classes].mean(0),label,reduce=False)
                #norms = torch.norm(pred.detach()[:,:,:self.n_classes].squeeze(),dim=1)
            elif self.reweight=="dbst":
                weights = self.compute_weights(aleatorics,epistemics)#这weight是归一化的
            elif self.reweight=="entropy":
                weights = self.compute_weights(epistemics,0)#这weight是归一化的
            elif self.reweight=="variance":
                weights = self.compute_weights(logits_var,0)#这weight是归一化的
            elif self.reweight=="a_pred":
                weights = self.compute_weights(aleatorics,0)#这weight是归一化的
            elif self.reweight=="var_ratio":
                weights = self.compute_weights(variational_ratio,0)#这weight是归一化的
            all_weights[index:index+len(label)]=weights.cpu()
            #all_norms[index:index+len(label)]=norms.cpu()
            index+=len(label)
        clean_s=list(np.argwhere(self.noised_sample==0).reshape(-1))
        noised_s=list(np.argwhere(self.noised_sample==1).reshape(-1))
        clean_weights=all_weights[clean_s]
        noised_weights=all_weights[noised_s]
        #clean_norms=all_norms[clean_s]
        #noised_norms=all_norms[noised_s]
        print("drawing weight dist...")
        self.log_file.write("drawing weight dist..."+"\n")
        plt.clf()
        plt.close()
        plt.hist(all_weights,bins=500,histtype="step",color="b",label="all sample weights")
        plt.hist(clean_weights,bins=500,histtype="step",color="r",label="clean sample weights")
        plt.hist(noised_weights,bins=500,histtype="step",color="g",label="noised sample weights")
        plt.ylabel("freq")
        plt.xlabel("weight")
        plt.legend()
        plt.title("weight distribution")
        plt.savefig(os.path.join(self.results_dir,"weight_dist"+'p'+str(phase)+'_'+str(epoch)+".png"))
        plt.close()
        """print("drawing norm dist...")
        self.log_file.write("drawing norm dist..."+"\n")
        plt.clf()
        plt.close()
        plt.hist(all_norms,bins=500,histtype="step",color="b",label="all sample norms")
        plt.hist(clean_norms,bins=500,histtype="step",color="r",label="clean sample norms")
        plt.hist(noised_norms,bins=500,histtype="step",color="g",label="noised sample norms")
        plt.ylabel("freq")
        plt.xlabel("norm")
        plt.legend()
        plt.title("norm distribution")
        plt.savefig(os.path.join(self.results_dir,"norm_dist"+'p'+str(phase)+'_'+str(epoch)+".png"))
        plt.close()"""
        return 0

    def relabel(self):
        self.module.eval()
        data_labels=self.relabel_dataset.targets
        pred_labels=np.zeros(self.dataset_size)
        all_weights=np.zeros(self.dataset_size)
        index=0
        for batch_idx, (data,label) in enumerate(self.relabel_loader):
            data = data.to(self.device)
            label = label.to(self.device)
            #data = data.float()
            #label = label.float()
            if self.dropout_prob==0:
                pred = self.sample_predict(data,1)#MC_preds
            else:
                pred = self.sample_predict(data,self.mc_n_sample)#MC_preds
            aleatorics,epistemics,logits_var,variational_ratio=self.calculate_uncertainty(pred)
            """if self.reweight=="none":
                weights=torch.ones(pred.shape[1])
                weights/=len(data)"""
            if self.reweight=="what":
                weights = self.compute_weights(epistemics,logits_var)#这weight是归一化的
            elif self.reweight=="dbst":
                weights = self.compute_weights(aleatorics,epistemics)#这weight是归一化的
            elif self.reweight=="entropy":
                weights = self.compute_weights(epistemics,0)#这weight是归一化的
            elif self.reweight=="variance":
                weights = self.compute_weights(logits_var,0)#这weight是归一化的
            elif self.reweight=="a_pred":
                weights = self.compute_weights(aleatorics,0)#这weight是归一化的
            elif self.reweight=="var_ratio":
                weights = self.compute_weights(variational_ratio,0)#这weight是归一化的
            elif self.reweight=="none":
                weights = F.cross_entropy(pred.detach()[:,:,:self.n_classes].mean(0),label,reduce=False)
            all_weights[index:index+len(label)]=weights.cpu()
            probs = torch.softmax(pred.detach()[:,:,:self.n_classes],-1)
            final_pred=probs.mean(0)#N*K
            _, batch_pred_label = final_pred.topk(1, -1, True, True)
            pred_labels[index:index+len(label)]=batch_pred_label.reshape(-1).cpu()
            index+=len(label)
            if batch_idx%self.args.log_interval==0:
                print(weights)
                self.log_file.write((str(weights)+"\n"))
        print("relabeling...")
        self.log_file.write("relabeling..."+"\n")


        #relabel_idx=(all_weights<self.args.relabel_thre/self.args.batch_size).astype(int)
        relabel_idx=np.zeros(self.dataset_size)
        relabel_idx[all_weights.argsort()[0:int(self.dataset_size*self.args.relabel_thre)]]=1
        a=[]
        b=[]
        precision=[]
        precision_idx=[]
        for i,thre in enumerate(self.args.reweight_precision_thre):
            precision_idx.append(np.zeros(self.dataset_size))
            if self.reweight=="none":
                precision_idx[i][(-all_weights).argsort()[0:int(self.dataset_size*thre)]]=1
            else:
                precision_idx[i][all_weights.argsort()[0:int(self.dataset_size*thre)]]=1
            a.append(precision_idx[i]+self.noised_sample)#0,1,2
            b.append(precision_idx[i]-self.noised_sample)#-1,0,1
            if precision_idx[i].sum()==0:
                precision.append(0)
            else:
                precision.append(((a[i]+10*b[i])==2).astype(int).sum()/precision_idx[i].sum())#so a should be 2 and b should be 0, which means that relabel_idx is 1 and noise_sample is 1
        self.log_file.write('====> Average precision list: {}\n'.format(str(precision)))
        print('====> Average precision list: {}\n'.format(str(precision)))
        if self.args.relabel_thre!=0:
            relabeled_labels=relabel_idx*pred_labels+(1-relabel_idx)*data_labels
            self.current_dataset.targets=torch.tensor(relabeled_labels).long()
            self.train_loader=DataLoader(dataset=self.current_dataset,batch_size=self.args.batch_size,shuffle=True)
        """
        上面if里面的东西会造成不可预料的对数据集的影响，先在还未查明
        """
        return precision

    def train(self):
        begin=time.perf_counter()
        self.module.train()
        self.loss_curve=[]
        self.test_loss_curve=[]
        self.acc_curve=[]
        self.test_acc_curve=[]
        self.precision_curve=[]
        if self.args.corruption_type=="partial_unif":
            self.uncertainty_curve=[[],[],[],[]]
            self.co_test_uncertainty_curve=[[],[],[],[]]
            self.cl_test_uncertainty_curve=[[],[],[],[]]
        else:
            self.uncertainty_curve=[[],[],[],[]]
            self.test_uncertainty_curve=[[],[],[],[]]
        self.uncertainty_index=0
        def train_phase(phase,epoch_num):
            tt=0
            #self.optimizer = optim.Adam(self.module.parameters(), lr=self.args.learning_rate)
            #self.optimizer = optim.SGD(self.module.parameters(), lr=self.args.learning_rate)
            self.optimizer = optim.SGD(self.module.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay,nesterov=self.args.nesterov)
            if self.args.lr_scheduler=="multistep":
                self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.milestones)
            elif self.args.lr_scheduler=="cosine":
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5,eta_min=4e-08)
            elif self.args.lr_scheduler=="cosine_m":
                pass
            else:
                print(self.args.lr_scheduler)
                assert False, "invalid lr scheduler"
            for epoch in range(epoch_num):
                if self.args.relabel_f!="none" and phase!=0:
                    if epoch%(int(self.args.relabel_f))==0:
                        with torch.no_grad():
                            self.precision_curve.append(self.relabel())
                self.module.train()
                train_loss = 0.
                train_acc = 0.
                print('current lr {:.5e}'.format(self.optimizer.param_groups[0]['lr']))
                self.log_file.write('current lr {:.5e}\n'.format(self.optimizer.param_groups[0]['lr']))
                for batch_idx, (data,label) in enumerate(self.train_loader):
                    data = data.to(self.device)
                    label = label.to(self.device)
                    #data = data.float()
                    #label = label.float()
                    if self.reweight!="none":
                        pred = self.sample_predict(data,self.mc_n_sample)#MC_preds:T*N*(K+1)
                    else:
                        if self.loss_type=="hs":
                            pred = self.sample_predict(data,1)
                        elif self.loss_type=="hm":
                            pred = self.sample_predict(data,self.mc_n_sample)#MC_preds
                        elif self.loss_type=="s":
                            pred = self.sample_predict(data,1)
                    aleatorics,epistemics,logits_var,variational_ratio=self.calculate_uncertainty(pred)
                    self.uncertainty_curve[0].append(aleatorics.mean().item())
                    self.uncertainty_curve[1].append(epistemics.mean().item())
                    self.uncertainty_curve[2].append(logits_var.mean().item())
                    self.uncertainty_curve[3].append(variational_ratio.mean().item())
                    if self.reweight=="none" or phase==0:
                        weights=torch.ones(pred.shape[1]).to(self.device)
                        weights/=len(data)
                    elif self.reweight=="what":
                        F.nll_loss()
                        weights = self.compute_weights(epistemics,logits_var)#这weight是归一化的
                    elif self.reweight=="dbst":
                        weights = self.compute_weights(aleatorics,epistemics)#这weight是归一化的
                    elif self.reweight=="entropy":
                        weights = self.compute_weights(epistemics,0)#这weight是归一化的
                    elif self.reweight=="variance":
                        weights = self.compute_weights(logits_var,0)#这weight是归一化的
                    elif self.reweight=="a_pred":
                        weights = self.compute_weights(aleatorics,0)#这weight是归一化的
                    elif self.reweight=="var_ratio":
                        weights = self.compute_weights(variational_ratio,0)#这weight是归一化的

                    if self.loss_type=="hm":
                        loss = self.heteroscedastic_crossentropy(label,pred,"multi")#这里用了mcdropout的prob_mean，应该没人用过，这里用是因为算不确定性已经把T个随机前传的logits全算了，但可能会导致时间复杂度很高
                    elif self.loss_type=="hs":
                        loss = self.heteroscedastic_crossentropy(label,pred,"single")#这里用了与Kendall & Gal2017中相同的，train过程中是不用MC采样的，只会进行heteroscedastic采样
                    elif self.loss_type=="s":
                        if self.args.loss_debug=="n":
                            pred_prob=torch.softmax(pred[:,:,:self.n_classes],-1).mean(0)#pred T*N*K, pred_prob N*K
                            loss = self.nll_criterion(torch.log(torch.clamp(pred_prob, 1e-11, 1 - 1e-11)),label)
                        elif self.args.loss_debug=="c":
                            loss = self.crossentropy_criterion(pred[0,:,:self.n_classes],label)
                    loss=torch.mul(weights,loss)
                    loss=torch.sum(loss)#这里不reweight则是平均的loss，reweight则是加权平均，权重和为1
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    train_loss+=loss.item()*len(data)
                    self.loss_curve.append(loss.item())
                    acc=self.compute_acc(label,pred.detach()[:,:,:self.n_classes])
                    train_acc+=acc.item()*len(data)
                    self.acc_curve.append(acc.item())
                    if self.args.lr_scheduler=="cosine_m":
                        dt = math.pi/float(epoch_num)
                        tt += float(dt)/(len(self.train_loader.dataset)/float(self.args.batch_size))
                        if tt >= math.pi - 0.05:
                            tt = math.pi - 0.05
                        curT = math.pi/2.0 + tt
                        new_lr = self.args.learning_rate * (1.0 + math.sin(curT))/2.0    # lr_min = 0, lr_max = lr
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                    if batch_idx % self.log_interval == 0:


                        print(weights)
                        self.log_file.write((str(weights)+"\n"))
                        
                        
                        self.log_file.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tAcc: {:.4f}\n'.format(
                            epoch, batch_idx * len(data), len(self.train_loader.dataset),
                            100. * batch_idx * len(data)/ len(self.train_loader.dataset),
                            loss.item(),acc.item()))
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tAcc: {:.4f}'.format(
                            epoch, batch_idx * len(data), len(self.train_loader.dataset),
                            100. * batch_idx * len(data)/ len(self.train_loader.dataset),
                            loss.item(),acc.item()))
                mean_uncertainties=[np.mean(self.uncertainty_curve[0][self.uncertainty_index:]),np.mean(self.uncertainty_curve[1][self.uncertainty_index:]),np.mean(self.uncertainty_curve[2][self.uncertainty_index:]),np.mean(self.uncertainty_curve[3][self.uncertainty_index:])]
                self.uncertainty_index=len(self.uncertainty_curve[0])
                self.log_file.write("phase "+str(phase))
                print("phase "+str(phase))
                self.log_file.write('====> Train Epoch: {} Average loss: {:.4f} Average acc: {:.4f}\n'.format(
                    epoch, train_loss/len(self.train_loader.dataset), train_acc/len(self.train_loader.dataset)))
                print('====> Train Epoch: {} Average loss: {:.4f} Average acc: {:.4f}'.format(
                    epoch, train_loss/len(self.train_loader.dataset), train_acc/len(self.train_loader.dataset)))
                self.log_file.write('====> Train Epoch: {} Average uncertainties: {:.4f}, {:.4f}, {:.4f}, {:.4f}\n'.format(epoch,mean_uncertainties[0],mean_uncertainties[1],mean_uncertainties[2],mean_uncertainties[3]))
                print('====> Train Epoch: {} Average uncertainties: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(epoch,mean_uncertainties[0],mean_uncertainties[1],mean_uncertainties[2],mean_uncertainties[3]))
                test_res=self.test(self.test_loader)
                self.test_loss_curve.append(test_res[0])
                self.test_acc_curve.append(test_res[1])
                if self.args.corruption_type!="partial_unif":
                    self.test_uncertainty_curve[0].append(test_res[2][0].mean().item())
                    self.test_uncertainty_curve[1].append(test_res[2][1].mean().item())
                    self.test_uncertainty_curve[2].append(test_res[2][2].mean().item())
                    self.test_uncertainty_curve[3].append(test_res[2][3].mean().item())
                else:
                    self.co_test_uncertainty_curve[0].append(test_res[2][0].mean().item())
                    self.co_test_uncertainty_curve[1].append(test_res[2][1].mean().item())
                    self.co_test_uncertainty_curve[2].append(test_res[2][2].mean().item())
                    self.co_test_uncertainty_curve[3].append(test_res[2][3].mean().item())
                    self.cl_test_uncertainty_curve[0].append(test_res[3][0].mean().item())
                    self.cl_test_uncertainty_curve[1].append(test_res[3][1].mean().item())
                    self.cl_test_uncertainty_curve[2].append(test_res[3][2].mean().item())
                    self.cl_test_uncertainty_curve[3].append(test_res[3][3].mean().item())
                if max(self.test_acc_curve)==test_res[1] or len(self.test_acc_curve)==0:#early stop
                    self.save()
                    self.best_test_acc=test_res[1]
                if self.args.lr_scheduler!="cosine_m":
                    self.lr_scheduler.step()
                self.last_test_acc=test_res[1]
                if epoch%20==0:
                    self.weight_dist(epoch,phase)
        train_phase(0,self.args.p0epochs)
        train_phase(1,self.args.p1epochs)
        
        if self.reweight=="none":
            self.weight_dist_comp((self.args.p0epochs+self.args.p1epochs),2)
        else:
            self.weight_dist((self.args.p0epochs+self.args.p1epochs),2)


        plt.clf()
        fig, axes = plt.subplots(2, 2, sharex="all")

        axes[0,0].plot(self.uncertainty_curve[0])
        axes[0,0].set_title('aleatoric pred')
        axes[0,0].set_ylabel("uncertainty")
        axes[0,0].set_xlabel("Iteration")

        axes[1,0].plot(self.uncertainty_curve[1])
        axes[1,0].set_title('entropy')
        axes[1,0].set_ylabel("uncertainty")
        axes[1,0].set_xlabel("Iteration")

        axes[0,1].plot(self.uncertainty_curve[2])
        axes[0,1].set_title('logits var')
        axes[0,1].set_ylabel("uncertainty")
        axes[0,1].set_xlabel("Iteration")

        axes[1,1].plot(self.uncertainty_curve[3])
        axes[1,1].set_title('variational ratio')
        axes[1,1].set_ylabel("uncertainty")
        axes[1,1].set_xlabel("Iteration")
        plt.savefig(os.path.join(self.results_dir,"training_uncertainty.png"))
        plt.close()

        plt.clf()
        fig, axes = plt.subplots(2, 2, sharex="all")
        if self.args.corruption_type!="partial_unif":
            axes[0,0].plot(self.test_uncertainty_curve[0])
            axes[0,0].set_title('aleatoric pred')
            axes[0,0].set_ylabel("uncertainty")
            axes[0,0].set_xlabel("Iteration")

            axes[1,0].plot(self.test_uncertainty_curve[1])
            axes[1,0].set_title('entropy')
            axes[1,0].set_ylabel("uncertainty")
            axes[1,0].set_xlabel("Iteration")

            axes[0,1].plot(self.test_uncertainty_curve[2])
            axes[0,1].set_title('logits var')
            axes[0,1].set_ylabel("uncertainty")
            axes[0,1].set_xlabel("Iteration")

            axes[1,1].plot(self.test_uncertainty_curve[3])
            axes[1,1].set_title('variational ratio')
            axes[1,1].set_ylabel("uncertainty")
            axes[1,1].set_xlabel("Iteration")
            plt.savefig(os.path.join(self.results_dir,"testing_uncertainty.png"))
            plt.close()
        else:
            axes[0,0].plot(self.co_test_uncertainty_curve[0],label="corrupted")
            axes[0,0].plot(self.cl_test_uncertainty_curve[0],label="clean")
            axes[0,0].set_title('aleatoric pred')
            axes[0,0].set_ylabel("uncertainty")
            axes[0,0].set_xlabel("Iteration")

            axes[1,0].plot(self.co_test_uncertainty_curve[1],label="corrupted")
            axes[1,0].plot(self.cl_test_uncertainty_curve[1],label="clean")
            axes[1,0].set_title('entropy')
            axes[1,0].set_ylabel("uncertainty")
            axes[1,0].set_xlabel("Iteration")

            axes[0,1].plot(self.co_test_uncertainty_curve[2],label="corrupted")
            axes[0,1].plot(self.cl_test_uncertainty_curve[2],label="clean")
            axes[0,1].set_title('logits var')
            axes[0,1].set_ylabel("uncertainty")
            axes[0,1].set_xlabel("Iteration")

            axes[1,1].plot(self.co_test_uncertainty_curve[3],label="corrupted")
            axes[1,1].plot(self.cl_test_uncertainty_curve[3],label="clean")
            axes[1,1].set_title('variational ratio')
            axes[1,1].set_ylabel("uncertainty")
            axes[1,1].set_xlabel("Iteration")
            plt.legend()
            plt.savefig(os.path.join(self.results_dir,"testing_uncertainty.png"))
            plt.close()
        plt.clf()
        plt.plot(self.loss_curve)
        plt.yscale('log')
        plt.title('training loss curve')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir,"train_loss"))
        plt.close()
        plt.clf()
        plt.plot(self.test_loss_curve)
        plt.yscale('log')
        plt.title('testing loss curve')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir,"test_loss"))
        plt.close()
        plt.clf()
        plt.plot(self.acc_curve)
        plt.title('training acc curve')
        plt.savefig(os.path.join(self.results_dir,"train_acc"))
        plt.close()
        plt.clf()
        plt.plot(self.test_acc_curve)
        plt.title('testing acc curve')
        plt.savefig(os.path.join(self.results_dir,"test_acc"))
        plt.close()
        plt.clf()
        pc=np.array(self.precision_curve)
        for i, thre in enumerate(self.args.reweight_precision_thre):
            plt.plot(pc[:,i],label=str(thre)+"% sample")
        plt.legend()
        plt.title('precision curve')
        plt.savefig(os.path.join(self.results_dir,"precision"))
        plt.close()
        end=time.perf_counter()
        self.log_file.write('base total training time: {:.4f}\n'.format(end-begin))
        self.log_file.write("best test acc: "+str(self.best_test_acc)+"\n")
        with open(os.path.join(self.results_dir,str(self.best_test_acc)+"_"+str(self.last_test_acc)+".txt"),"w") as f:
            f.write("acc sign")
        self.log_file.close()

    def test(self, test_loader):
        self.module.eval()
        eval_loss = 0.
        eval_acc = 0.
        if self.args.corruption_type!="partial_unif":
            uncertainty_curve=[[],[],[],[]]
        else:
            co_ucurve=[[],[],[],[]]#corrupted uncertainty curve
            cl_ucurve=[[],[],[],[]]#clean uncertainty curve
        with torch.no_grad():
            for data,label in test_loader:
                data = data.to(self.device)
                label = label.to(self.device)
                #data = data.float()
                #label = label.float()
                if self.dropout_prob==0:
                    pred = self.sample_predict(data,1)#MC_preds
                else:
                    pred = self.sample_predict(data,self.mc_n_sample)#MC_preds
                if self.loss_type=="hm":
                    loss = self.heteroscedastic_crossentropy(label,pred,"multi")#这里用了mcdropout的prob_mean，应该没人用过，这里用是因为算不确定性已经把T个随机前传的logits全算了，但可能会导致时间复杂度很高
                elif self.loss_type=="hs":
                    loss = self.heteroscedastic_crossentropy(label,pred,"single")#这里用了与Kendall & Gal2017中相同的，train过程中是不用MC采样的，只会进行heteroscedastic采样
                elif self.loss_type=="s":
                    pred_prob=torch.softmax(pred[0,:,:self.n_classes],-1)#pred 1*N*K, pred_prob N*K
                    loss = self.nll_criterion(torch.log(torch.clamp(pred_prob,1e-11,1-1e-11)),label)
                    #loss = self.crossentropy_criterion(pred[0,:,:self.n_classes],label)
                aleatorics,epistemics,logits_var,variational_ratio=self.calculate_uncertainty(pred)
                if self.reweight!="none":
                    weights = self.compute_weights(aleatorics,epistemics)#这weight是归一化的
                    loss = torch.mul(weights,loss)
                else:
                    loss/=len(data)
                loss=torch.sum(loss)#这里不reweight则是平均的loss，reweight则是加权平均，权重和为1
                eval_loss+=loss.item()*len(data)
                acc=self.compute_acc(label,pred[:,:,:self.n_classes])
                eval_acc+=acc.item()*len(data)
                if self.args.corruption_type!="partial_unif":
                    uncertainty_curve[0].append(aleatorics.mean().item())
                    uncertainty_curve[1].append(epistemics.mean().item())
                    uncertainty_curve[2].append(logits_var.mean().item())
                    uncertainty_curve[3].append(variational_ratio.mean().item())
                else:
                    co_u=[[],[],[],[]]
                    cl_u=[[],[],[],[]]
                    for i in range(len(label)):
                        if label[i] in self.args.corrupted_classes:
                            co_u[0].append(aleatorics[i].item())
                            co_u[1].append(epistemics[i].item())
                            co_u[2].append(logits_var[i].item())
                            co_u[3].append(variational_ratio[i].item())
                        else:
                            cl_u[0].append(aleatorics[i].item())
                            cl_u[1].append(epistemics[i].item())
                            cl_u[2].append(logits_var[i].item())
                            cl_u[3].append(variational_ratio[i].item())
                    co_ucurve[0].append(np.mean(co_u[0]))
                    co_ucurve[1].append(np.mean(co_u[1]))
                    co_ucurve[2].append(np.mean(co_u[2]))
                    co_ucurve[3].append(np.mean(co_u[3]))
                    cl_ucurve[0].append(np.mean(cl_u[0]))
                    cl_ucurve[1].append(np.mean(cl_u[1]))
                    cl_ucurve[2].append(np.mean(cl_u[2]))
                    cl_ucurve[3].append(np.mean(cl_u[3]))
        if self.args.corruption_type!="partial_unif":
            mean_uncertainties=[np.mean(uncertainty_curve[0]),np.mean(uncertainty_curve[1]),np.mean(uncertainty_curve[2]),np.mean(uncertainty_curve[3])]
        else:
            co_mean_uncertainties=[np.mean(co_ucurve[0]),np.mean(co_ucurve[1]),np.mean(co_ucurve[2]),np.mean(co_ucurve[3])]
            cl_mean_uncertainties=[np.mean(cl_ucurve[0]),np.mean(cl_ucurve[1]),np.mean(cl_ucurve[2]),np.mean(cl_ucurve[3])]
        self.log_file.write('====> Test Average loss: {:.4f} Average acc: {:.4f}\n'.format(eval_loss/len(test_loader.dataset), eval_acc/len(test_loader.dataset)))
        print('====> Test Average loss: {:.4f} Average acc: {:.4f}'.format(eval_loss/len(test_loader.dataset), eval_acc/len(test_loader.dataset)))
        if self.args.corruption_type!="partial_unif":
            self.log_file.write('====> Test Average uncertainties: {:.4f}, {:.4f}, {:.4f}, {:.4f}\n'.format(mean_uncertainties[0],mean_uncertainties[1],mean_uncertainties[2],mean_uncertainties[3]))
            print('====> Test Average uncertainties: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(mean_uncertainties[0],mean_uncertainties[1],mean_uncertainties[2],mean_uncertainties[3]))
            return eval_loss/len(test_loader.dataset), eval_acc/len(test_loader.dataset), mean_uncertainties
        else:
            self.log_file.write('====> Test Average corrupted class pics uncertainties: {:.4f}, {:.4f}, {:.4f}, {:.4f}\n'.format(co_mean_uncertainties[0],co_mean_uncertainties[1],co_mean_uncertainties[2],co_mean_uncertainties[3]))
            print('====> Test Average uncertainties: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(co_mean_uncertainties[0],co_mean_uncertainties[1],co_mean_uncertainties[2],co_mean_uncertainties[3]))
            self.log_file.write('====> Test Average clean class pics uncertainties: {:.4f}, {:.4f}, {:.4f}, {:.4f}\n'.format(cl_mean_uncertainties[0],cl_mean_uncertainties[1],cl_mean_uncertainties[2],cl_mean_uncertainties[3]))
            print('====> Test Average uncertainties: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(cl_mean_uncertainties[0],cl_mean_uncertainties[1],cl_mean_uncertainties[2],cl_mean_uncertainties[3]))
            return eval_loss/len(test_loader.dataset), eval_acc/len(test_loader.dataset), co_mean_uncertainties, cl_mean_uncertainties

    def save(self,path="trained_model.pth"):
        torch.save(self.module.state_dict(), os.path.join(self.results_dir,path))
        return 0

    def load(self,path="trained_model.pth"):
        self.module.load_state_dict(torch.load(os.path.join(self.results_dir,path)))