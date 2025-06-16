import torch
import torch.nn.functional as F
from torch import nn

from src.models import BaseModel


class Model(BaseModel):
    def __init__(self, channels=32, num_classes=10, **kwargs):
        super(Model, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(          #(1,28,28)
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2   #padding=(kernelsize-stride)/2
            ),#(16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)#(16,14,14)
 
        )
        self.conv2=nn.Sequential(#(16,14,14)
            nn.Conv2d(16,32,5,1,2),#(32,14,14)
            nn.ReLU(),#(32,14,14)
            nn.MaxPool2d(2)#(32,7,7)
        )
        self.out=nn.Linear(32*7*7,10)


    def forward(self, x):
        x = self.conv1( x )
        x = self.conv2( x ) #(batch,32,7,7)
        x=x.view(x.size(0),-1) #(batch,32*7*7)
        output=self.out(x)
        return output



