import torch
import torch.nn as nn
from src.models import BaseModel


class Model(BaseModel):
    def __init__(self, feature_dimension, num_classes, **kwargs):
        super(Model, self).__init__(feature_dimension=feature_dimension, num_classes=num_classes)
        # Linear regression is equivalent to using a fully connected layer
        self.linear = nn.Linear(in_features=feature_dimension, out_features=num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        out = self.linear(x)
        return out
