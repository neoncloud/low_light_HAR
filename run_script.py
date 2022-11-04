import torch
from torchvision.io import read_video
import numpy as np
from numpy.random import randint
import os
from model.network import build_model
from model.text_prompt import text_prompt
import yaml
import argparse
from dotmap import DotMap
from util.dark_enhance import DarkEnhance
from einops import rearrange
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg')
    parser.add_argument('--path', '-p')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = DotMap(cfg)
    
    state_dict = torch.load(cfg.resume, map_location='cuda')
    model = build_model(
            state_dict=state_dict['model_state_dict'],
            pretrain=True,
            motion_layers=cfg.network.motion.num_layers,
            motion_layers_init=cfg.network.motion.init,
            train_visual=cfg.visual.train,
            T=cfg.data.seg_length,
            thres=cfg.network.motion.thres,
            alpha=cfg.network.other.alpha,
            fusion_type=cfg.network.fusion.type
        ).cuda()

    name_list = ['Drink','Jump','Pick','Pour','Push','Run','Sit','Stand','Turn','Walk']
    num_text_aug, text_tokenized = text_prompt(name_list)
    all_text_features = model.encode_text(
            rearrange(text_tokenized, 'c n d -> (c n) d').cuda())
    transform = Compose([
        DarkEnhance(),
        Resize(256, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    all_test_result = []
    for file in os.listdir(args.path):
        if file.endswith(".mp4"):
            frames, _, _  = read_video(os.path.join(args.path,file))
            video_len = frames.shape[0]
            step = video_len//cfg.data.num_segments
            end = step*cfg.data.num_segments
            # T H W C
            #frames = frames[:end:step,...].permute(0,3,1,2).cuda()
            frames = frames.permute(0,3,1,2).cuda()
            # T C H W
            frames = transform(frames).unsqueeze(0)
            # 1(B) T C H W
            
            similarity = model.inference(frames/255.0, text_features=all_text_features)
            similarity = rearrange(similarity, 'b (n d) -> b n d', n=num_text_aug)
            similarity = similarity.mean(dim=1, keepdim=False)
            _, indices_1 = similarity.topk(1, dim=-1)

            test_result = (file.strip('.mp4'), indices_1[0].item())
            print(test_result)
            all_test_result.append(test_result)
            torch.cuda.empty_cache()
    with open(os.path.join('test_result.txt'), 'w') as file:
        file.write('\n'.join('%s	%d' % x for x in all_test_result))