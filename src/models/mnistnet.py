import torch
import torch.nn as nn
from src.models import BaseModel
import torch.nn.functional as F


class Model(BaseModel):
    def __init__(self, feature_dimension, num_classes, **kwargs):
        super(Model, self).__init__(feature_dimension=feature_dimension, num_classes=num_classes)
        # LSoftmax regression is equivalent to using a fully connected layer and a softmax layer
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output
