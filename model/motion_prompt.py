import torch
import torch.nn as nn
from util.frame_diff import Sandevistan
from clip.model import ResidualAttentionBlock, VisionTransformer, LayerNorm, Transformer
from einops import rearrange

class ResidualCrossAttentionBlock(ResidualAttentionBlock):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__(d_model, n_head, attn_mask)
        self.ln_1_y = LayerNorm(d_model)
    
    def attention(self, x: torch.Tensor, y: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(y, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x,y = torch.chunk(x,2,-1)
        x = x + self.attention(self.ln_1(x),self.ln_1_y(y))
        x = x + self.mlp(self.ln_2(x))
        return x

class CrossTransformer(Transformer):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__(width, layers, heads, attn_mask)
        self.resblocks = nn.Sequential(*([ResidualCrossAttentionBlock(width, heads, attn_mask)]+[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers-1)]))

class MotionPrompt(VisionTransformer):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, T: int=8, thres:float=4.0):
        super().__init__(input_resolution, patch_size, width, layers, heads, output_dim)
        # patching layer for Sandevistan motion features
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=width*2, kernel_size=patch_size, stride=patch_size, bias=False)
        self.transformer = CrossTransformer(width, layers, heads)
        self.frame_diff = Sandevistan(n_trunks=T, thres=thres)
        self.ln_pre = LayerNorm(width*2)

    def encode_video(self, vid: torch.Tensor):
        with torch.no_grad():
            x,y = self.frame_diff(vid)
            x = torch.cat((x,y),2)
            x = rearrange(x, 'b n c h w -> (b n) c h w')
        return x

    def forward(self, x: torch.Tensor):
        ######
        x = self.encode_video(x)
        ######
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        ######
        cls = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1]//2, dtype=x.dtype, device=x.device)
        cls = torch.cat([cls, cls], dim=2)
        x = torch.cat([cls, x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        pos = torch.cat([self.positional_embedding.to(x.dtype), self.positional_embedding.to(x.dtype)], dim=1)
        print(pos.shape, x.shape)
        x = x + pos
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        cls_x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            cls_x = cls_x @ self.proj
        
        return cls_x, x[:,1:,:]

# class MotionPrompt(nn.Module):
#     def __init__(self, n_trunk:int=8, out_features:int=2048, thres:float=4.0) -> None:
#         super().__init__()

#         self.frame_diff = Sandevistan(n_trunks=n_trunk, thres=thres)
    
#     def forward(self, x):
#         with torch.no_grad():
#             # B T C H W
#             x = self.frame_diff(x)
#         return x
