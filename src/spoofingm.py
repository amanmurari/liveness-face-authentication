import torch
from torch import nn
from torch.nn import functional as F

class SpoofNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn=nn.Sequential(
            nn.Conv2d(3,8,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,512,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512,1024,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(1024,16,3,padding=1),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.ffn=nn.Sequential(
            nn.Linear(16,512),
            nn.Linear(512,1),
        )
    def forward(self,x):
        return self.ffn(self.nn(x))