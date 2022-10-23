from torch.utils.data import Dataset
import os, csv
from PIL import Image, ImageOps

class VideoDataset(Dataset):
    def __init__(self, total_frames:int, frame_size:int, video_list_path:str, video_label_path:str, image_tmpl:str="{:06d}.jpg") -> None:
        """Video dataset

        Args:
            total_frames (int): num of returned frames from each video clip
            frame_size (int): frame resolution. Imgs will be resized to this size.
            video_list_path (str): path to the dataset file list
            video_label_path (str): path to the label to name mapping csv file
            image_tmpl (str): file name template, e.g. "frame_{:06d}.jpg"
        """
        super().__init__()
        self.total_frames = total_frames
        self.frame_size = frame_size
        self.video_list_path = video_list_path
        self.video_label_path = video_label_path
        self.image_tmpl = image_tmpl
        self.video_files = []
        self.labels = {}

        self.read_file_list()
        self.read_labels()

    def read_file_list(self):
        root, csv_file = os.path.split(self.video_list_path)
        with open(self.video_list_path,'r') as file:
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
                self.labels[int(row['id'])] = row['name']
    
    def read_frames(self, path: str, n_frames: int):
        step = n_frames//self.total_frames
        return [Image.open(os.path.join(path, self.image_tmpl.format(idx))).convert('RGB').resize((self.frame_size, self.frame_size)) for idx in range(1, n_frames+1, step)]

    def __getitem__(self, idx):
        path, n_frames, label = self.video_files[idx]
        frames = self.read_frames(path, n_frames)
        return frames, self.labels[label]

    def __len__(self):
        return len(self.video_files)