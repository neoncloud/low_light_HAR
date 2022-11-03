import torch
from torchvision.io import read_video
import numpy as np
from numpy.random import randint
import os

def sample_indices(num_frames, total_length=8):
    if num_frames <= total_length:
        offsets = np.concatenate((
            np.arange(num_frames),
            randint(num_frames,
                    size=total_length - num_frames)))
        return np.sort(offsets)
    offsets = list()
    ticks = [i * num_frames // total_length
              for i in range(total_length + 1)]

    for i in range(total_length):
        tick_len = ticks[i + 1] - ticks[i]
        tick = ticks[i]
        if tick_len >= 1:
            tick += randint(tick_len)
        offsets.extend([j for j in range(tick, tick + 1)])
    return np.array(offsets)

model = torch.jit.load('/home/neoncloud/ActionCLIP/full_model_scripted_ob.pt')
torch.backends.cudnn.benchmark = False
path = "/mnt/e/EE6222/validate/"
for file in os.listdir(path):
    if file.endswith(".mp4"):
        # model_ = model.cuda()
        frames, _, _  = read_video(os.path.join(path,file))
        video_len = frames.shape[0]
        indices = torch.tensor(sample_indices(video_len,8),dtype=torch.long)
        frames = frames[indices,...]
        # torch.cuda.empty_cache()
        #print(frames.shape)
        pred = model(frames).cpu()
        print(file, pred[0].item())
        torch.cuda.empty_cache()