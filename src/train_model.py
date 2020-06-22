# coding=utf-8
import numpy as np
import torch
from baseline_model import Baseline_model
from simple_model import Simple_model
from data_loader import Corrupted_dataset
import os
import torch.utils.data as Data
import pickle
import argparse

parser = argparse.ArgumentParser(description='uncertainty reweighting in noisy datasets')
parser.add_argument('--p0epochs', type=int, default=30,
                    help='number of epochs for phase 0(no reweight) to train')
parser.add_argument('--p1epochs', type=int, default=45,
                    help='number of epochs for phase 1(reweight) to train')              
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
parser.add_argument('--milestones', type=list, default=[40,50],
                    help='milestones')  
parser.add_argument('--MC_n_samples', type=int, default=10,
                    help='number of sample in mc dropout')
parser.add_argument('--loss_debug', type=str, default="n",
                    help='n for nll, c for cross entropy')
parser.add_argument('--n_var', type=int, default=10,
                    help='how many var to predict')
parser.add_argument('--model', type=str, default="wrn",
                    help='wrn, rn or dn')
parser.add_argument('--net_arg', type=int, default=10,
                    help='growth rate for densenet and widden factor for wrn')
parser.add_argument('--depth', type=int, default=28,
                    help='depth of densenet(resnet is fixed 32)')
parser.add_argument('--dropout_prob', type=float, default=0.5,
                    help='the dropout ratio')
parser.add_argument('--lr_scheduler', type=str, default="cosine_m",
                    help='lr scheduler')
parser.add_argument('--num_classes', type=int, default=10,
                    help='how many classes in dataset(CIFAR 10 or CIFAR 100)')
parser.add_argument('--corruption_type', type=str, default="unif",
                    help='the corrution type in the training set')
parser.add_argument('--size_ratio', type=float, default=1,
                    help='use size_ratio of the training set')
parser.add_argument('--train_class', type=list, default=range(0,10),
                    help='training set is composed of these classes')
parser.add_argument('--test_class', type=list, default=range(0,10),
                    help='testing set is composed of these classes')
parser.add_argument('--corrupted_classes', type=list, default=[0,1,2,3,4],
                    help='takes effect when corruption_prob=partial_unif')  

parser.add_argument('--corruption_prob', type=float, default=0.6,
                    help='the corrution ratio in the training set')
parser.add_argument('--reweight', type=str, default="var_ratio",
                    help='how to reweight')
parser.add_argument('--reweight_function', type=str, default="exp",
                    help='reweight function')
parser.add_argument('--reweight_norm', type=str, default="sig_m",
                    help='reweight normalization, sig_m or batch')
parser.add_argument('--loss_type', type=str, default="s",
                    help='type of loss, hm, hs, s')
parser.add_argument('--relabel_f', type=str, default="3",
                    help='frequency of relabeling')
parser.add_argument('--clean_only', type=bool, default=False,
                    help='train only on the clean part of the dataset')
parser.add_argument('--noise class', type=bool, default=False,
                    help='add a new class(noise class)')               


parser.add_argument('--relabel_thre', type=float, default=0.3,
                    help='relabel threshold ratio on weights(when 128 the threshold will be 1/128*x')


os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0,1,2,3]))
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
print("Let's use", torch.cuda.device_count(), "GPUs")
device = torch.device("cuda:0" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#directory
data_dir="../data"
results_dir="../results"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

train_set=Corrupted_dataset(args,root=data_dir,download=True,transform=None,target_transform=None,train=True)
test_set=Corrupted_dataset(args,root=data_dir,download=True,transform=None,target_transform=None,train=False)
train_loader,noised_sample=train_set.get_data_loader()
test_loader=test_set.get_data_loader()
#train_loader,test_loader=train_wrn.build_dataset(args)
#noised_sample=None

brm=Baseline_model(args,results_dir,device,train_loader,test_loader,noised_sample)
brm.train()
brm.save()
torch.cuda.empty_cache()

args.reweight_norm="none"
'''
args.reweight="var_ratio"
brm=Baseline_model(args,results_dir,device,train_loader,test_loader,noised_sample)
brm.train()
brm.save()
torch.cuda.empty_cache()

args.reweight="entropy"
brm=Baseline_model(args,results_dir,device,train_loader,test_loader,noised_sample)
brm.train()
brm.save()
torch.cuda.empty_cache()

args.reweight_norm="sig_m"

args.reweight="entropy"
brm=Baseline_model(args,results_dir,device,train_loader,test_loader,noised_sample)
brm.train()
brm.save()
torch.cuda.empty_cache()
'''


args.reweight_norm="batch"

"""args.reweight="var_ratio"
brm=Baseline_model(args,results_dir,device,train_loader,test_loader,noised_sample)
brm.train()
brm.save()
torch.cuda.empty_cache()

args.reweight="entropy"
brm=Baseline_model(args,results_dir,device,train_loader,test_loader,noised_sample)
brm.train()
brm.save()
torch.cuda.empty_cache()

args.reweight="none"
brm=Baseline_model(args,results_dir,device,train_loader,test_loader,noised_sample)
brm.train()
brm.save()
torch.cuda.empty_cache()"""