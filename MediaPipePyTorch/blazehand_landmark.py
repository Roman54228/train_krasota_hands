import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from MediaPipePyTorch.blazebase import BlazeLandmark, BlazeBlock
# from blazebase import BlazeLandmark, BlazeBlock


class ClassificationHead(nn.Module):
    def __init__(self, in_channels=288, num_classes=5, dropout=0.3):
        super().__init__()
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.head = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            # nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            # nn.Dropout(dropout),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.gap(x)           # (B, 288, 1, 1)
        x = x.view(x.size(0), -1) # (B, 288)
        x = self.head(x)          # (B, 5)
        return x
    
    
class BlazeHandLandmark(BlazeLandmark):
    """The hand landmark model from MediaPipe.
    
    """
    def __init__(self):
        super(BlazeHandLandmark, self).__init__()

        # size of ROIs used for input
        self.resolution = 256

        self._define_layers()

    def _define_layers(self):
        self.backbone1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),

            BlazeBlock(24, 24, 5),
            BlazeBlock(24, 24, 5),
            BlazeBlock(24, 48, 5, 2),
        )

        self.backbone2 = nn.Sequential(
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 96, 5, 2),
        )

        self.backbone3 = nn.Sequential(
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5, 2),
        )

        self.backbone4 = nn.Sequential(
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5, 2),
        )

        self.blaze5 = BlazeBlock(96, 96, 5)
        self.blaze6 = BlazeBlock(96, 96, 5)
        self.conv7 = nn.Conv2d(96, 48, 1, bias=True)

        self.backbone8 = nn.Sequential(
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 96, 5, 2),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 288, 5, 2),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5, 2),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5, 2),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5, 2),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
        )

        #self.hand_flag = nn.Conv2d(288, 1, 2, bias=True)
        # self.hand_cls = nn.Conv2d(288, 5, 2, bias=True)
        self.handed = nn.Conv2d(288, 1, 2, bias=True)
        self.landmarks = nn.Conv2d(288, 63, 2, bias=True)
        self.cls_head = ClassificationHead()
        # self.cls5_layer = nn.Conv2d(288, 5, 2, bias=True)
        
        # self.lin1 = nn.Linear(5, 128)
        # self.lin2 = nn.Linear(128,60)
        # self.lin3 = nn.Linear(60,5)
        # self.silu = nn.SiLU()
        # self.landmarks2 = nn.Conv2d(288, 42, 2, bias=True)


    def forward(self, x):
        if x.shape[0] == 0:
            return torch.zeros((0,)), torch.zeros((0,)), torch.zeros((0, 21, 3))

        x = F.pad(x, (0, 1, 0, 1), "constant", 0)

        x = self.backbone1(x)
        y = self.backbone2(x)
        z = self.backbone3(y)
        w = self.backbone4(z)

        z = z + F.interpolate(w, scale_factor=2, mode='bilinear')
        z = self.blaze5(z)

        y = y + F.interpolate(z, scale_factor=2, mode='bilinear')
        y = self.blaze6(y)
        y = self.conv7(y)

        x = x + F.interpolate(y, scale_factor=2, mode='bilinear')

        x = self.backbone8(x)

        # hand_flag = self.cls5_layer(x).view(-1, 5)#.sigmoid()
        # hand_flag = self.silu(hand_flag)
        # hand_flag = self.lin1(hand_flag)
        # hand_flag = self.silu(hand_flag)
        # hand_flag = self.lin2(hand_flag)
        # hand_flag = self.silu(hand_flag)
        # hand_flag = self.lin3(hand_flag)
        hand_flag = self.cls_head(x)
        # handed = self.handed(x).view(-1).sigmoid()
        # breakpoint()
        landmarks = self.landmarks(x).view(-1, 21, 3) / 256
        

        # return hand_flag, handed, landmarks
        return hand_flag, landmarks
        # return landmarks