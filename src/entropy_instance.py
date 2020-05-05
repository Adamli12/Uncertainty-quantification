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

steps=100
classes=10
scale=0.05

def variational_ratio(y_hat):
    y_hat = torch.clamp(y_hat, 1e-11, 1 - 1e-11) # prevent nans,(N,K)
    y_max,_ = torch.max(y_hat, -1)
    return torch.sub(1,y_max)

def compute_entropy(y_hat):
    y_hat=y_hat.clamp(1e-11,1-1e-11)
    return -torch.sum(y_hat * torch.log(y_hat), dim=1)

#unif entropy
probs=torch.zeros(steps,classes)
probs[:,0]=torch.from_numpy(np.linspace(1,1/classes,num=steps))
probs[:,1:]=((-probs[:,0]+1)/(classes-1)).view(100,-1)
entropies=compute_entropy(probs)
x=probs[:,0]
plt.plot(x,entropies,label="one hot->uniform, entropy")

#flip entropy
probs=torch.zeros(steps,classes)
probs[:,0]=torch.from_numpy(np.linspace(1,0.5,num=steps))
probs[:,1]=torch.from_numpy(np.linspace(0,0.5,num=steps))
entropies=compute_entropy(probs)
x=probs[:,0]
plt.plot(x,entropies,label="one hot->0.5, entropy")

#unif var_ratio
probs=torch.zeros(steps,classes)
probs[:,0]=torch.from_numpy(np.linspace(1,1/classes,num=steps))
probs[:,1:]=((-probs[:,0]+1)/(classes-1)).view(100,-1)
vrs=variational_ratio(probs)
x=probs[:,0]
plt.plot(x,vrs,label="one hot->uniform, var_ratio")

#flip var_ratio
probs=torch.zeros(steps,classes)
probs[:,0]=torch.from_numpy(np.linspace(1,0.5,num=steps))
probs[:,1]=torch.from_numpy(np.linspace(0,0.5,num=steps))
vrs=variational_ratio(probs)
x=probs[:,0]
plt.plot(x,vrs,label="one hot->0.5, var_ratio")


#pertubation
probs=torch.zeros(steps,classes)
probs[:,:]=1/classes
gaussian=torch.distributions.Normal(0,scale)
pertubation=gaussian.sample(probs.shape)
probs=probs+pertubation
probs=probs.clamp(1e-11,1-1e-11)
probs=probs/(torch.sum(probs,dim=1)).view(100,-1)

entropies=compute_entropy(probs)
plt.boxplot(entropies,widths=0.1)
plt.legend()
plt.xlabel("max prob of classes")
plt.ylabel("uncertainty")
plt.show()
plt.close()