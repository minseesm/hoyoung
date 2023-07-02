import torch
import torch.nn as nn
import numpy as np

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(15, 200, bias=True),   # [n, 15] -> [n, 40]
            nn.ReLU(),
            nn.Linear(200, 200, bias=True),   # [n, 40] -> [n, 40]
            nn.ReLU(),
            nn.Linear(200, 1, bias=True),     # [n, 40] -> [n, 1]
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer(x)     # [n, 15]   n : batch size
        return out
    

class Linear_custom(nn.Module):
    def __init__(self, node_num):
        super(Linear_custom, self).__init__()

        self.node_num = node_num

        self.layer = nn.Sequential(
            nn.Linear(15, self.node_num, bias=True),   # [n, 15] -> [n, 40]
            nn.ReLU(),
            nn.Linear(self.node_num, self.node_num, bias=True),   # [n, 40] -> [n, 40]
            nn.ReLU(),
            nn.Linear(self.node_num, 1, bias=True),     # [n, 40] -> [n, 1]
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer(x)     # [n, 15]   n : batch size
        return out

in_data = torch.randn([15])
model = Linear()
print(model)
out = model(in_data)
print(out)