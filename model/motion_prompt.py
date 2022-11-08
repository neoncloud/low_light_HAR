import torch
import torch.nn as nn
from clip.model import ResidualAttentionBlock, VisionTransformer, LayerNorm, Transformer
from einops import rearrange

class ResidualCrossAttentionBlock(ResidualAttentionBlock):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__(d_model, n_head, attn_mask)
        self.ln_1_y = LayerNorm(d_model)
    
    def attention(self, x: torch.Tensor, y: torch.Tensor=None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, y: torch.Tensor=None):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CrossTransformer(Transformer):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__(width, layers, heads, attn_mask)
        self.resblocks = nn.ModuleList([ResidualCrossAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
    
    def forward(self, x: torch.Tensor, y: list=None):
        for blk in self.resblocks:
            x = torch.utils.checkpoint.checkpoint(blk, x)
        return x

class MotionPrompt(VisionTransformer):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, T: int):
        super().__init__(input_resolution, patch_size, width, layers, heads, output_dim)
        # patching layer for Sandevistan motion features
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=width,
                               kernel_size=patch_size, stride=patch_size, bias=False)
        self.transformer = CrossTransformer(width, layers, heads)
        self.temporal_embeddings = nn.Embedding(
            T, (input_resolution // patch_size) ** 2)
        self.T = T

    def forward(self, x: torch.Tensor, y: list):
        # x:video tensor, y: list of features of frames from CLIP ViT
        x = torch.utils.checkpoint.checkpoint(
            self.conv1, x)  # shape = [*, width, grid, grid]
        # x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = rearrange(x, '(b t) c w1 w2 -> b c t (w1 w2)', t=self.T)
        position_ids = torch.arange(
            self.T, dtype=torch.long, device=x.device)
        x = x + self.temporal_embeddings(
            position_ids).to(x.dtype)
        #x = x + rearrange(temp_embeddings, 't w -> 1 t 1 w')
        x = rearrange(x, 'b c t w -> (b t) w c')
        # x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                      dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = torch.utils.checkpoint.checkpoint(self.ln_pre, x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x
