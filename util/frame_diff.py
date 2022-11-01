from typing import Optional
import torch
from torch.nn.functional import relu
from einops import rearrange, reduce


class Sandevistan(torch.nn.Module):
    def __init__(self, n_frames: int = 5, thres: float = 4.0, n_trunks: Optional[int] = None) -> None:
        super().__init__()
        if n_trunks is None:
            self.n_frames = n_frames
            self.n_trunks = None
        else:
            self.n_frames = None
            self.n_trunks = n_trunks
        self.thres = thres

    def forward(self, x: torch.Tensor):
        # B T C H W
        if self.n_trunks is None:
            n = x.shape[1]//self.n_frames
            frames = n*self.n_frames
        else:
            n = self.n_trunks
            frames = n*(x.shape[1]//n)

        # drop some tailing frames
        x = x[:, :frames, ...]
        x = rearrange(x, 'b (n t) c h w -> b n t c h w', n=n)
        # leading frames of each chunk
        y = x[:, :, 0, ...]
        x = (x[:, :, 1:, ...] - y[:, :, None, ...]).abs()
        x = reduce(x, 'b n t c h w -> b n c h w', 'sum')
        x = reduce(x, 'b n c h w -> b n 1 h w', 'max')
        x = relu(x-self.thres)
        return x, y
