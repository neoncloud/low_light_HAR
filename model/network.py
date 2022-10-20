from typing import Tuple, Union
import torch
import torch.nn as nn
from clip.model import CLIP, ResidualAttentionBlock, VisionTransformer, LayerNorm, Transformer
from util.frame_diff import Sandevistan
from einops import rearrange
from model.motion_prompt import MotionPrompt

class Transformer_(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, return_all_feat: bool=False):
        if return_all_feat:
            all_feat = []
            for blk in self.resblocks:
                x = blk(x)
                all_feat.append(x)
                #x = x['out']
            return all_feat
        else:
            for blk in self.resblocks:
                x = blk(x)
            return x

class VisionTransformer_(VisionTransformer):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__(input_resolution, patch_size, width, layers, heads, output_dim)
        self.transformer = Transformer_(width, layers, heads)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        ######
        all_feat = self.transformer(x, True)
        x = all_feat[-1]
        ######
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x, all_feat

class SandevistanCLIP(CLIP):
    def __init__(self, embed_dim: int, image_resolution: int, vision_layers: Union[Tuple[int, int, int, int], int], vision_width: int, vision_patch_size: int, context_length: int, vocab_size: int, transformer_width: int, transformer_heads: int, transformer_layers: int, T: int=8, thres: float=4.0):
        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer_(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.motion = MotionPrompt(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(1 / 0.07).values)
        self.T = T
        self.frame_diff = Sandevistan(n_trunks=T, thres=thres)
        self.initialize_parameters()
    
    def encode_motion(self, motion_feat: torch.Tensor, frame_feat: torch.Tensor):
        return self.motion(motion_feat, frame_feat)

    def encode_image(self, video: torch.Tensor):
        b,t,c,h,w = video.shape
        with torch.no_grad():
            motion_feat, frames = self.frame_diff(video)
            motion_feat = torch.cat((motion_feat,frames),2)
            motion_feat = rearrange(motion_feat, 'b t c h w -> (b t) c h w')
            frames = rearrange(frames, 'b t c h w -> (b t) c h w')
        class_features, frame_features = self.visual(frames.type(self.dtype))
        motion_features = self.encode_motion(motion_feat,frame_features)
        
        class_features = class_features.view(b, -1, *class_features.shape[1:]).mean(1)
        motion_features = motion_features.view(b, -1, *motion_features.shape[1:]).mean(1)

        video_features = (class_features+motion_features)/2
        return video_features