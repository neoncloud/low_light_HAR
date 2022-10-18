from typing import Optional
import torch
import torch.nn.functional as F
from einops import rearrange, reduce

class Sandevistan(torch.nn.Module):
    def __init__(self, n_frames:int=5, thres:float= 4.0, n_trunks:Optional[int]=None) -> None:
        super().__init__()
        if n_trunks is None:
            self.n_frames = n_frames
            self.n_trunks = None
        else:
            self.n_frames = None
            self.n_trunks = n_trunks
        self.thres = thres
    def forward(self, x:torch.Tensor):
        # B T C H W
        if self.n_trunks is None:
            n = x.shape[1]//self.n_frames
            frames = n*self.n_frames
        else:
            n = self.n_trunks
            frames = n*(x.shape[1]//n)

        # drop some tailing frames
        x = x[:,:frames,...]
        x = rearrange(x,'b (n t) c h w -> b n t c h w',n=n)
        x = (x[:,:,1:,...] - x[:,:,0, None, ...]).abs()
        x = reduce(x, 'b n t c h w -> b n c h w', 'sum')
        x = reduce(x, 'b n c h w -> b n h w', 'mean')
        x = F.relu(x-self.thres)
        # x_list = torch.split(x,self.n_frames,dim=1)
        # output=[]
        # for trunk in x_list:
        #     start_frame = trunk[:,0,...]
        #     trunk -= start_frame[:,None,...].clone()
        #     trunk = trunk[:,1:,...].abs().sum(1).mean(1)
        #     trunk = F.relu(trunk-self.thres)
        #     output.append(trunk)
        return x