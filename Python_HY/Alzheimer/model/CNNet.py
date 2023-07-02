import torch
import torch.nn as nn
import numpy as np

class CNNet(nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),     # [1, 32, 128, 128]
            nn.MaxPool2d(2),                    # [1, 32, 64, 64]
            nn.Conv2d(32, 64, 3, padding=1),    # [1, 64, 64, 64]
            nn.MaxPool2d(2),                    # [1, 64, 32, 32]
            nn.Conv2d(64, 128, 3, padding=1),   # [1, 128, 32, 32]
            nn.MaxPool2d(2),                    # [1, 128, 16, 16] 
            nn.Conv2d(128, 256, 3, padding=1),  # [1, 256, 16, 16]
            nn.MaxPool2d(2),                    # [1, 256, 8, 8]
        )
        self.fc = nn.Sequential(
            nn.Linear(16384, 4)     # 4 : No, Verymild, mild, moderate
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(out.size(0),-1)          # Flatten [1, 256 * 8 * 8]
        out = self.fc(out)
        return out

class CNNet_convtrans(nn.Module):
    def __init__(self):
        super(CNNet_convtrans, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),        # [1, 32, 64, 64]
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),        # [1, 32, 64, 64]
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),       # [1, 64, 32, 32]
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0), 

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),   # [1, 128, 16, 16] 
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0), 

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [1, 256, 8, 8]
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0), 
        )
        self.fc = nn.Sequential(
            nn.Linear(16384, 4)     # 4 : No, Verymild, mild, moderate
        )


    def forward(self, x):
        out = self.layer(x)
        out = out.view(out.size(0),-1)          # Flatten [1, 256 * 8 * 8]
        out = self.fc(out)
        return out