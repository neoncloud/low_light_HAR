from typing import Tuple, Union
import torch
import torch.nn as nn
from clip.model import CLIP, ResidualAttentionBlock, VisionTransformer, convert_weights
from util.frame_diff import Sandevistan
from einops import rearrange
from model.motion_prompt import MotionPrompt

class Transformer_(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, num_feats: int=0):
        all_feats = []
        for i, blk in enumerate(self.resblocks):
            x = blk(x)
            if i < num_feats:
                all_feats.append(x)
            #x = x['out']
        return all_feats, x


class VisionTransformer_(VisionTransformer):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, num_feats:int):
        super().__init__(input_resolution, patch_size, width, layers, heads, output_dim)
        self.transformer = Transformer_(width, layers, heads)
        self.num_feats = num_feats

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        ######
        all_feats, x = self.transformer(x, self.num_feats)
        ######
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x, all_feats

class SandevistanCLIP(CLIP):
    def __init__(self, embed_dim: int, image_resolution: int, vision_layers: Union[Tuple[int, int, int, int], int], motion_layers: Union[Tuple[int, int, int, int], int], vision_width: int, vision_patch_size: int, context_length: int, vocab_size: int, transformer_width: int, transformer_heads: int, transformer_layers: int, T: int=8, thres: float=4.0):
        super().__init__(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)
        
        vision_heads = vision_width // 64
        self.visual = VisionTransformer_(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            num_feats=motion_layers
        )

        self.motion = MotionPrompt(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=motion_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.T = T
        self.frame_diff = Sandevistan(n_trunks=T, thres=thres)
    
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
        del frame_features
        class_features = class_features.view(b, -1, *class_features.shape[1:]).mean(1)
        motion_features = motion_features.view(b, -1, *motion_features.shape[1:]).mean(1)

        video_features = (class_features+motion_features)/2
        return video_features

def build_model(state_dict: dict, T=8, pretrain=True, motion_layers=None):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        print('img transformer layers:',vision_layers)
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
        
        if motion_layers == None:
            motion_layers = vision_layers
        
        if "motion.proj" not in state_dict:
            motion = {}
            for k,v in state_dict.items():
                if 'visual.' in k:
                    if k == 'visual.conv1.weight':
                        continue
                    k_ = k.replace('visual.','motion.')
                    motion[k_] = v
            state_dict.update(motion)
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    model = SandevistanCLIP(
        embed_dim,
        image_resolution, vision_layers, motion_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, T=T
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    if pretrain:
        print('loading clip pretrained model!')
        model.load_state_dict(state_dict,strict=False)
    else:
        print('not using full clip pretrained model, only visual!')
        
        for k in list(state_dict.keys()):
            if not k.find("visual")>-1: 
                state_dict.pop(k)

        model.load_state_dict(state_dict,strict=False)
    return model.eval()
