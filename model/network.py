from collections import OrderedDict
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from clip.model import CLIP, ResidualAttentionBlock, VisionTransformer, QuickGELU, Transformer, LayerNorm
from util.frame_diff import Sandevistan
from einops import rearrange
from model.motion_prompt import MotionPrompt, ResidualCrossAttentionBlock


class Transformer_(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(
            width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, num_feats: int = 0):
        all_feats = []
        # if not x.requires_grad:
        #     x = x.requires_grad_(True)
        for i, blk in enumerate(self.resblocks):
            x = torch.utils.checkpoint.checkpoint(blk, x)
            if i < num_feats:
                all_feats.append(x)
            #x = x['out']
        return all_feats, x


class CrossTransformer_(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.cross_resblock = ResidualCrossAttentionBlock(
            width, heads, attn_mask)
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers-1)])

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = self.resblocks(x)
        return self.cross_resblock(x, y)

class FusionModel(nn.Module):
    def __init__(self, width:int, fusion_type:str='transf', transformer_heads:int=8) -> None:
        super().__init__()
        if fusion_type == 'transf':
            self.fusion = Transformer(
                width=width,
                layers=2,
                heads=transformer_heads
            )
            # self.ln_pre = LayerNorm(width)
            # self.ln_post = LayerNorm(width)
            # self.proj = nn.Parameter(torch.randn(width, width))
        elif fusion_type == 'mlp':
            self.fusion = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(transformer_heads+1, 2*transformer_heads)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(2*transformer_heads, 1))
            ]))
        elif fusion_type == 'mean':
            self.fusion = None
        else:
            raise NotImplementedError

        self.fusion_type = fusion_type

    def forward(self, x:torch.Tensor):
        b, c, f = x.shape
        if self.fusion_type == 'transf':
            #x = self.ln_pre(x)
            x = (x+self.fusion(x)).mean(0)
            #x = self.ln_post(x[0,:,:])
            #x = x @ self.proj
        elif self.fusion_type == 'mlp':
            x = x.permute(0, 2, 1)
            x = self.fusion(x)
            x = x.squeeze()
        elif self.fusion_type == 'mean':
            x = x.mean(1)
        return x


class VisionTransformer_(VisionTransformer):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, num_feats: int):
        super().__init__(input_resolution, patch_size, width, layers, heads, output_dim)
        self.transformer = Transformer_(width, layers, heads)
        self.num_feats = num_feats

    def forward(self, x: torch.Tensor):
        x = torch.utils.checkpoint.checkpoint(
            self.conv1, x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                      dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = torch.utils.checkpoint.checkpoint(self.ln_pre, x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        ######
        all_feats, x = self.transformer(x, self.num_feats)
        ######
        x = x[0, :, :]  # LND -> NLD
        x = torch.utils.checkpoint.checkpoint(self.ln_post, x)

        if self.proj is not None:
            x = x @ self.proj

        return x, all_feats


class SandevistanCLIP(CLIP):
    def __init__(self, embed_dim: int, image_resolution: int, vision_layers: Union[Tuple[int, int, int, int], int], motion_layers: Union[Tuple[int, int, int, int], int], vision_width: int, vision_patch_size: int, context_length: int, vocab_size: int, transformer_width: int, transformer_heads: int, transformer_layers: int, T: int = 8, thres: float = 4.0, alpha: float = 0.3, fusion_type: str = 'transf'):
        super().__init__(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)
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
            output_dim=embed_dim,
            T=T
        )

        self.fusion = FusionModel(
            width=embed_dim,
            fusion_type=fusion_type,
            transformer_heads=transformer_heads
        )

        # self.frame_position_embeddings = nn.Embedding(
        #     context_length, embed_dim)
        #self.T = T
        self.frame_diff = Sandevistan(n_trunks=T, thres=thres)
        #self.alpha = nn.parameter.Parameter(torch.tensor([alpha]))

    def encode_motion(self, motion_feat: torch.Tensor, frame_feat: list):
        # frame_feat_ = []
        # for y in frame_feat:
        #     print(y.shape)
        #     b, n, c = y.size()
        #     seq_length = n
        #     position_ids = torch.arange(
        #         seq_length, dtype=torch.long, device=y.device)
        #     position_ids = position_ids.unsqueeze(0).expand(y.size(0), -1)
        #     frame_position_embeddings = self.frame_position_embeddings(
        #         position_ids)
        #     y = y + frame_position_embeddings
        #     frame_feat_.append(y)
        return self.motion(motion_feat, frame_feat)

    # def encode_video(self, x, y):
    #     b, n, c = x.size()
    #     x_original = x
    #     seq_length = n
    #     position_ids = torch.arange(
    #         seq_length, dtype=torch.long, device=x.device)
    #     position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
    #     frame_position_embeddings = self.frame_position_embeddings(
    #         position_ids)
    #     x = x + frame_position_embeddings

    #     x = x.permute(1, 0, 2)  # NLD -> LND
    #     y = y.permute(1, 0, 2)
    #     x = self.temporal(x, y)
    #     x = x.permute(1, 0, 2)  # LND -> NLD
    #     x = x.type(x_original.dtype) + x_original
    #     return x

    def encode_image(self, video: torch.Tensor):
        b, t, c, h, w = video.shape
        motion, frames = self.frame_diff(video)
        motion = rearrange(
            motion, 'b t c h w -> (b t) c h w').detach()
        frames = rearrange(
            frames, 'b t c h w -> (b t) c h w').detach()
        class_features, frame_features = self.visual(
            frames.type(self.dtype).requires_grad_(True))
        video_features = self.encode_motion(
            motion.type(self.dtype).requires_grad_(True),
            [f.requires_grad_(True) for f in frame_features])

        #del frame_features
        # class_features = class_features.view(
        #     b, -1, class_features.shape[-1]).mean(1)
        # motion_features = motion_features.view(
        #     b, -1, motion_features.shape[-1]).mean(1)
        # video_features = self.encode_video(
        #     class_features, motion_features).mean(1)
        #video_features = (self.alpha*class_features+(1-self.alpha)*motion_features).mean(1)
        class_features = rearrange(class_features, '(b t) c -> t b c', b=b).mean(0, keepdim=True)
        video_features = rearrange(video_features, '(b t) c -> t b c', b=b)

        #video_features = reduce(video_features, '(b t) f -> b f', 'mean', b=b)
        # b t+1 f
        video_features = torch.cat((class_features, video_features),dim=0)
        video_features = self.fusion(video_features)
        return video_features

    def forward(self, image: torch.Tensor, text: Optional[torch.Tensor] = None, text_features: Optional[torch.Tensor] = None):
        if text_features is None:
            text_features = self.encode_text(text)
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)

        image_features = self.encode_image(image)
        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    def inference(self, image: torch.Tensor, text: Optional[torch.Tensor] = None, text_features: Optional[torch.Tensor] = None):
        if text_features is None:
            text_features = self.encode_text(text)
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)

        image_features = self.encode_image(image)
        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        # shape = [global_batch_size, global_batch_size]
        return text_probs


def build_model(state_dict: dict, T: int = 8, thres: float = 2.0, pretrain: bool = True, motion_layers: Optional[int] = None, motion_layers_init: bool = True, train_visual: bool = False, train_text: bool = False, alpha: float = 0.3, fusion_type: str = 'transf'):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith(
            "visual.") and k.endswith(".attn.in_proj_weight")])
        print('img transformer layers:', vision_layers)
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        if motion_layers == None:
            motion_layers = vision_layers

        if motion_layers_init:
            motion = {}
            for k, v in state_dict.items():
                if 'visual.' in k:
                    if k == 'visual.conv1.weight':
                        continue
                    k_ = k.replace('visual.', 'motion.')
                    motion[k_] = v
            state_dict.update(motion)
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(
            f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)

        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + \
            1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(
        k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = SandevistanCLIP(
        embed_dim,
        image_resolution, vision_layers, motion_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, T, thres, alpha, fusion_type
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)
    if pretrain:
        print('loading clip pretrained model!')
        not_loaded = model.load_state_dict(state_dict, strict=False)
        print('The following sub-module is not loaded')
        print(not_loaded)
    else:
        print('not using full clip pretrained model, only visual!')

        for k in list(state_dict.keys()):
            if not k.find("visual") > -1:
                state_dict.pop(k)

        model.load_state_dict(state_dict, strict=False)

    if not train_visual:
        for param in model.visual.parameters():
            param.requires_grad = False
    if not train_text:
        for param in model.transformer.parameters():
            param.requires_grad = False
        for param in model.token_embedding.parameters():
            param.requires_grad = False
        for param in model.ln_final.parameters():
            param.requires_grad = False
        model.text_projection.requires_grad = False
        model.positional_embedding.requires_grad = False
    return model.eval().float()


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()
