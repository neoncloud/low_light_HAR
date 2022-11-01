from torchvision.io import read_video
from torch.utils.data import Dataset
import os
import csv
from PIL import Image
import torch
from clip.clip import _transform

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _transform_video(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


class VideoFramesDataset(Dataset):
    def __init__(self, n_classes: int, n_frames: int, frame_size: int, video_list_path: str, video_label_path: str, image_tmpl: str = "{:06d}.jpg") -> None:
        """Video dataset

        Args:
            n_classes (int): num of video classes
            n_frames (int): num of returned frames from each video clip
            frame_size (int): frame resolution. Imgs will be resized to this size.
            video_list_path (str): path to the dataset file list
            video_label_path (str): path to the label to name mapping csv file
            image_tmpl (str): file name template, e.g. "frame_{:06d}.jpg"
        """
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.video_list_path = video_list_path
        self.video_label_path = video_label_path
        self.image_tmpl = image_tmpl
        self.video_files = []
        self.name_list = []
        self.transform = _transform(frame_size)

        self.read_labels()
        self.read_file_list()

    def read_file_list(self):
        root, _ = os.path.split(self.video_list_path)
        with open(self.video_list_path, 'r') as file:
            for line in file:
                path, n_frames, label = line.split(' ')
                path = os.path.join(root, path)
                n_frames = int(n_frames)
                label = int(label)
                self.video_files.append((path, n_frames, label))

    def read_labels(self):
        with open(self.video_label_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.name_list.append(row['name'])

    def read_frames(self, path: str, n_frames: int):
        step = n_frames//self.n_frames
        n_frames = step*self.n_frames
        if step > 1:
            start = torch.randint(1, step+1, (1,)).item()
        else:
            start = 1
        return [self.transform(Image.open(os.path.join(path, self.image_tmpl.format(idx))))for idx in range(start, n_frames+1, step)]

    def one_hot(self, x, on_value=1., off_value=0., device='cpu'):
        x = x.long().view(-1, 1)
        return torch.full((x.size()[0], self.n_classes), off_value, device=device).scatter_(1, x, on_value)

    def __getitem__(self, idx):
        while True:
            path, n_frames, label = self.video_files[idx]
            if n_frames < self.n_frames:
                idx += 1
                continue
            else:
                break
        frames = self.read_frames(path, n_frames)
        frames = torch.stack(frames, dim=0)
        label = torch.tensor(label, dtype=torch.long)
        #name = self.name_list[label]
        return {'frames': frames, 'label': label}

    def __len__(self):
        return len(self.video_files)


class VideoDataset(VideoFramesDataset):
    def __init__(self, n_classes: int, n_frames: int, frame_size: int, video_list_path: str, video_label_path: str, image_tmpl: str = "{:06d}.jpg") -> None:
        self.name_dict = {}
        super().__init__(n_classes, n_frames, frame_size,
                         video_list_path, video_label_path, image_tmpl)
        self.transform = _transform_video(frame_size)

    def read_labels(self):
        with open(self.video_label_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.name_list.append(row['name'])
                self.name_dict[row['name']] = int(row['id'])

    def read_file_list(self):
        root, _ = os.path.split(self.video_list_path)
        with open(self.video_list_path, 'r') as file:
            for line in file:
                path, n_frames = line.split(' ')[0:2]
                path = os.path.join(root, path)
                name = path.split('/')[-2].replace('_', ' ')
                label = self.name_dict[name]
                self.video_files.append((path, label))

    def read_frames(self, path: str):
        frames, _, _ = read_video(path)
        step = frames.shape[0]//self.n_frames
        if step == 0:
            return None
        n_frames = step*self.n_frames
        if step > 1:
            start = torch.randint(0, step, (1,)).item()
        else:
            start = 0
        return self.transform(frames[start:n_frames:step, ...].float().transpose(3, 1))

    def __getitem__(self, idx):
        while True:
            path, label = self.video_files[idx]
            frames = self.read_frames(path)
            if frames is None:
                idx += 1
                continue
            else:
                label = torch.tensor(label, dtype=torch.long)
                return {'frames': frames, 'label': label}
