from torch.utils.data import DataLoader
from .dataset import VideoDataset
from dotmap import DotMap


def get_dataloder(cfg: DotMap):
    train_dataset = VideoDataset(
        n_classes=cfg.data.num_classes,
        n_frames=cfg.data.num_segments,
        frame_size=cfg.data.input_size,
        video_list_path=cfg.data.train_list,
        video_label_path=cfg.data.label_list,
        image_tmpl=cfg.data.image_tmpl
    )

    validate_dataset = VideoDataset(
        n_classes=cfg.data.num_classes,
        n_frames=cfg.data.num_segments,
        frame_size=cfg.data.input_size,
        video_list_path=cfg.data.val_list,
        video_label_path=cfg.data.label_list,
        image_tmpl=cfg.data.image_tmpl
    )

    train_dataloader = DataLoader(
        dataset = train_dataset,
        shuffle = True,
        batch_size = cfg.data.batch_size,
        num_workers = cfg.data.worker,
        pin_memory = True
    )

    validate_dataloader = DataLoader(
        dataset = validate_dataset,
        shuffle = True,
        batch_size = cfg.data.batch_size,
        num_workers = cfg.data.worker,
        pin_memory = True
    )

    return train_dataloader, validate_dataloader, train_dataset.name_list
