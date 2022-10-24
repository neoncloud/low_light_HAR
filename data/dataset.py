from torch.utils.data import Dataset
import os
import csv
from PIL import Image, ImageOps
import torch
from clip.clip import _transform


class VideoDataset(Dataset):
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

        self.read_file_list()
        self.read_labels()

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
        start = torch.randint(1,step+1,(1,)).item()
        return [self.transform(Image.open(os.path.join(path, self.image_tmpl.format(idx))))for idx in range(start, n_frames+1, step)]

    def one_hot(self, x, on_value=1., off_value=0., device='cpu'):
        x = x.long().view(-1, 1)
        return torch.full((x.size()[0], self.n_classes), off_value, device=device).scatter_(1, x, on_value)

    def __getitem__(self, idx):
        path, n_frames, label = self.video_files[idx]
        frames = self.read_frames(path, n_frames)
        frames = torch.stack(frames, dim=0)
        label = torch.tensor(label,dtype=torch.long)
        #name = self.name_list[label]
        return {'frames': frames, 'label': label}

    def __len__(self):
        return len(self.video_files)