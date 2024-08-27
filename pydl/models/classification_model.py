import torch
import torch.nn as nn
from nncore.nn import MODELS


@MODELS.register()
class Mymodel(nn.Module):
    def __init__(self, net, loss=None):
        super(Mymodel, self).__init__()
        self.net = net
        self.loss = loss if loss is not None else nn.CrossEntropyLoss()

    def forward(self, data, **kwargs):
        x, y = data[0], data[1]
        x = self.net(x)
        out = torch.argmax(x, dim=1)
        acc = torch.eq(out, y).sum().float() / x.size(0)
        loss = self.loss(x, y)

        return dict(_avg_factor=x.size(0), acc=acc, loss=loss)
