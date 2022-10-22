import torch
import torch.nn as nn
from clip.model import ResidualAttentionBlock, VisionTransformer, LayerNorm, Transformer

class ResidualCrossAttentionBlock(ResidualAttentionBlock):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__(d_model, n_head, attn_mask)
        self.ln_1_y = LayerNorm(d_model)
    
    def attention(self, x: torch.Tensor, y: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(y, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = x + self.attention(self.ln_1(x),self.ln_1_y(y))
        x = x + self.mlp(self.ln_2(x))
        return x

class CrossTransformer(Transformer):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__(width, layers, heads, attn_mask)
        self.resblocks = nn.ModuleList([ResidualCrossAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
    
    def forward(self, x: torch.Tensor, y: list):
        for blk, y_ in zip(self.resblocks, y[:len(self.resblocks)]):
            x = blk(x, y_)
        return x

class MotionPrompt(VisionTransformer):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__(input_resolution, patch_size, width, layers, heads, output_dim)
        # patching layer for Sandevistan motion features
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.transformer = CrossTransformer(width, layers, heads)

    def forward(self, x: torch.Tensor, y: list):
        # x:video tensor, y: list of features of frames from CLIP ViT
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        print(x.shape)
        x = self.transformer(x,y)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

# class MotionPrompt(nn.Module):
#     def __init__(self, n_trunk:int=8, out_features:int=2048, thres:float=4.0) -> None:
#         super().__init__()

#         self.frame_diff = Sandevistan(n_trunks=n_trunk, thres=thres)
    
#     def forward(self, x):
#         with torch.no_grad():
#             # B T C H W
#             x = self.frame_diff(x)
#         return x