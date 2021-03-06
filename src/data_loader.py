import torch
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import pickle
import numpy as np
import argparse
import copy
import os

#thanks to the code by Shu et. al (https://github.com/xjtushujun/meta-weight-net)

def uniform_mix_C(mixing_ratio, num_classes):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)

def flip_labels_C(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C

def flip_labels_C_two(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in two other entries for each row
    '''
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_prob / 2
    return C

def partial_unif_C(corruption_prob,num_classes,args):
    '''
    returns a matrix with only some classes corrupted(only corrupt into these classes), and other classes remain unaffected
    '''
    corrupted_classes=args.corrupted_classes
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = corrupted_classes
    for i in range(num_classes):
        if i not in row_indices:
            C[i][i]=1
        else:
            for j in row_indices:
                C[i][j] += corruption_prob/len(row_indices)
    return C

class Corrupted_dataset():
    def __init__(self,args,root='../../data',download=False,transform=None,target_transform=None,train=True,meta=False,do_meta=False):
        if do_meta==True:
            self.num_meta=1000
        else:
            self.num_meta=0
        self.meta=meta#for meta-weight-net training
        self.corruption_type=args.corruption_type
        self.corruption_prob=args.corruption_prob
        self.num_classes=args.num_classes
        self.train=train
        self.batch_size=args.batch_size
        self.seed=args.seed
        self.size_ratio=args.size_ratio
        self.train_class=args.train_class
        self.test_class=args.test_class

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        if transform==None:
            if train==True:
                transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize])


        if self.num_classes==10:
            self.cifar = datasets.CIFAR10(root,train=train,transform=transform,target_transform=target_transform,download=download)
        elif self.num_classes==100:
            self.cifar = datasets.CIFAR100(root,train=train,transform=transform,target_transform=target_transform,download=download)

        if self.train:
            if self.corruption_type == 'unif':
                C = uniform_mix_C(self.corruption_prob, self.num_classes)
                print(C)
                self.corruption_matrix = C
            elif self.corruption_type == 'flip':
                C = flip_labels_C(self.corruption_prob, self.num_classes)
                print(C)
                self.corruption_matrix = C
            elif self.corruption_type == 'flip2':
                C = flip_labels_C_two(self.corruption_prob, self.num_classes)
                print(C)
                self.corruption_matrix = C
            elif self.corruption_type == 'partial_unif':
                C = partial_unif_C(self.corruption_prob,self.num_classes,args)
            else:
                assert False, "Invalid corruption type '{}' given. Must be in {'unif', 'flip', 'flip2'}".format(self.corruption_type)

            np.random.seed(self.seed)
            masks=[]
            shuffled_list=np.arange(len(self.cifar))
            np.random.shuffle(shuffled_list)
            self.meta_idx=np.sort(shuffled_list[:self.num_meta])
            self.train_idx=np.sort(shuffled_list[self.num_meta:])
            #print(self.meta_idx[:100])
            #print(self.train_idx[:100])
            if self.meta:
                d=np.array(self.cifar.data)
                t=np.array(self.cifar.targets)
                self.cifar.data=list(d[self.meta_idx])
                self.cifar.targets=list(t[self.meta_idx])
            else:
                d=np.array(self.cifar.data)
                t=np.array(self.cifar.targets)
                self.cifar.data=list(d[self.train_idx])
                self.cifar.targets=list(t[self.train_idx])
                self.noised_sample=np.zeros(len(self.cifar))
                for i in range(len(self.cifar)):
                    if self.cifar.targets[i] not in self.train_class:
                        masks.append(i)
                    else:
                        label=self.cifar.targets[i]
                        noised=np.random.choice(self.num_classes, p=C[self.cifar.targets[i]])
                        self.cifar.targets[i] = noised
                        if args.clean_only==True:
                            masks.append(i)
                        if label!=noised:
                            self.noised_sample[i]=1
                targets_array=np.array(self.cifar.targets)
                targets_array=np.delete(targets_array,masks,0)
                self.noised_sample=np.delete(self.noised_sample,masks,0)
                self.cifar.data=np.delete(self.cifar.data,masks,0)
                self.cifar.targets=targets_array.tolist()
                """self.cifar, _ = torch.utils.data.random_split(self.cifar, [int(self.size_ratio*len(self.cifar)), len(self.cifar)-int(self.size_ratio*len(self.cifar))])#not able for recording noised sample?????"""
        else:
            masks=[]
            for i in range(len(self.cifar)):
                if self.cifar.targets[i] not in self.test_class:
                    masks.append(i)
            targets_array=np.array(self.cifar.targets)
            targets_array=np.delete(targets_array,masks,0)
            self.cifar.data=np.delete(self.cifar.data,masks,0)
            self.cifar.targets=targets_array.tolist()

    def get_data_loader(self):
        shuffle = False
        if self.train:
            shuffle = True
        data_loader = DataLoader(dataset=self.cifar,batch_size=self.batch_size,shuffle=shuffle)
        if self.train and self.meta==False:
            return data_loader,self.noised_sample
        else:
            return data_loader

if __name__ == "__main__":
    def build_dataset():
        train_set=Corrupted_dataset(args,root=data_dir,download=True,transform=None,target_transform=None,train=True,meta=False,do_meta=True)
        train_meta_set=Corrupted_dataset(args,root=data_dir,download=True,transform=None,target_transform=None,train=True,meta=True,do_meta=True)
        test_set=Corrupted_dataset(args,root=data_dir,download=True,transform=None,target_transform=None,train=False)
        train_loader,noised_sample=train_set.get_data_loader()
        train_meta_loader=train_meta_set.get_data_loader()
        test_loader=test_set.get_data_loader()
        return train_loader, train_meta_loader, test_loader, noised_sample

    parser = argparse.ArgumentParser(description='uncertainty reweighting in noisy datasets')              
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--visible_gpu', type=str, default="0,1,2,3",
                        help='visible gpu, in form "a,b,c,d"')  
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

    parser.add_argument('--p0epochs', type=int, default=0,
                        help='number of epochs for phase 0(no reweight) to train')
    parser.add_argument('--p1epochs', type=int, default=80,
                        help='number of epochs for phase 1(reweight) to train')
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
    parser.add_argument('--relabel_f', type=str, default="1",
                        help='frequency of relabeling')
    parser.add_argument('--clean_only', type=bool, default=False,
                        help='train only on the clean part of the dataset')
    parser.add_argument('--noise class', type=bool, default=False,
                        help='add a new class(noise class)')               


    parser.add_argument('--relabel_thre', type=float, default=0,
                        help='relabel threshold ratio on weights(when 128 the threshold will be 1/128*x')
    parser.add_argument('--reweight_precision_thre', type=list, default=[0.1,0.3,0.5,0.7],
                        help='experiment parameter, calculate the precision of low weight sample in the first x*dataset_size sample with the lowest weight')


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    print("Let's use", torch.cuda.device_count(), "GPUs")
    device = torch.device("cuda:0" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    #directory
    data_dir="../data"
    results_dir="../results"
    

    train_loader, train_meta_loader, test_loader, noised_sample = build_dataset()
    relabel_dataset=copy.deepcopy(train_loader.dataset)#the unchangable dataset
    relabel_loader=DataLoader(dataset=relabel_dataset,batch_size=args.batch_size,shuffle=False)
    to_pil=transforms.ToPILImage()
    for idx,(data,label) in enumerate(train_loader):
        pic=to_pil(data[0])
        print(label[0])
        pic.show()
        input()
