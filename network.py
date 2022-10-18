import torch
from torch.nn import Conv2d, Linear
from torchvision.models import resnet50
from util.dark_enhance import DarkEnhance
from util.frame_diff import Sandevistan

class MotionPrompt(torch.nn.Module):
    def __init__(self, n_trunk:int=8, out_features:int=2048, thres:float=4.0) -> None:
        super().__init__()
        self.feed_forward = resnet50()
        self.feed_forward.conv1 = Conv2d(n_trunk, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.feed_forward.fc = Linear(in_features=2048, out_features=out_features, bias=True)

        self.dark_enhancer = DarkEnhance()
        self.frame_diff = Sandevistan(n_trunks=n_trunk, thres=thres)
    
    def forward(self, x):
        with torch.no_grad():
            # B T H W C
            x = self.dark_enhancer(x)
            # B T C H W
            x = self.frame_diff(x.transpose(-1,-3))
        x = self.feed_forward(x)
        return x
