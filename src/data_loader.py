import torch
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import pickle
import numpy as np
import argparse

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
    def __init__(self,args,root='../../data',download=False,transform=None,target_transform=None,train=True):
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
            for i in range(len(self.cifar)):
                if self.cifar.targets[i] not in self.train_class:
                    masks.append(i)
                else:
                    self.cifar.targets[i] = np.random.choice(self.num_classes, p=C[self.cifar.targets[i]])
            targets_array=np.array(self.cifar.targets)
            targets_array=np.delete(targets_array,masks,0)
            self.cifar.data=np.delete(self.cifar.data,masks,0)
            self.cifar.targets=targets_array.tolist()
            self.cifar, _ = torch.utils.data.random_split(self.cifar, [int(self.size_ratio*len(self.cifar)), len(self.cifar)-int(self.size_ratio*len(self.cifar))])
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
        return data_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='uncertainty reweighting in noisy datasets')
    parser.add_argument('--epochs', type=int, default=40,
                        help='number of epochs to train')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--log_interval', type=int, default=60,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='how many classes in dataset(CIFAR 10 or CIFAR 100)')
    parser.add_argument('--corruption_prob', type=float, default=0.5,
                        help='the corrution ratio in the training set')
    parser.add_argument('--corruption_type', type=str, default="flip",
                        help='the corrution type in the training set')
    parser.add_argument('--dropout_prob', type=float, default=0,
                        help='the dropout ratio')
    parser.add_argument('--MC_n_samples', type=int, default=1,
                        help='number of sample in mc dropout')
    parser.add_argument('--loss_type', type=str, default="simple",
                        help='type of loss')
    parser.add_argument('--reweight', type=bool, default=False,
                        help='reweight or not')
    parser.add_argument('--n_var', type=int, default=10,
                        help='how many var to predict')
    parser.add_argument('--size_ratio', type=float, default=1,
                        help='use size_ratio of the training set')
    parser.add_argument('--train_class', type=list, default=range(2),
                        help='training set is composed of these classes')
    parser.add_argument('--test_class', type=list, default=range(10),
                        help='testing set is composed of these classes')
    parser.add_argument('--model', type=str, default="wrn",
                        help='wrn, rn or dn')
    parser.add_argument('--growth_rate', type=int, default=48,
                        help='growth rate')
    parser.add_argument('--dense_depth', type=int, default=40,
                        help='depth of densenet(resnet is fixed 32)')
    args = parser.parse_args()
    transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor()])
    train_set=Corrupted_dataset(args,root='../../data',download=True,transform=transform,target_transform=None,train=False)
    train_loader=train_set.get_data_loader()
    to_pil=transforms.ToPILImage()
    for idx,(data,label) in enumerate(train_loader):
        pic=to_pil(data[0])
        print(label[0])
        pic.show()
        input()
