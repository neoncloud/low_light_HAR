from torch.utils.data import DataLoader
from .dataset import VideoFramesDataset, VideoDataset
from dotmap import DotMap


def get_dataloder(cfg: DotMap):
    dataset = VideoDataset if cfg.data.type=='video' else VideoFramesDataset
    if cfg.data.train_list is not None:
        train_dataset = dataset(
            n_classes=cfg.data.num_classes,
            n_frames=cfg.data.num_segments,
            frame_size=cfg.data.input_size,
            video_list_path=cfg.data.train_list,
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
    else:
        train_dataset = None
        train_dataloader = None

    validate_dataset = dataset(
        n_classes=cfg.data.num_classes,
        n_frames=cfg.data.num_segments,
        frame_size=cfg.data.input_size,
        video_list_path=cfg.data.val_list,
        video_label_path=cfg.data.label_list,
        image_tmpl=cfg.data.image_tmpl
    )

    validate_dataloader = DataLoader(
        dataset = validate_dataset,
        shuffle = True,
        batch_size = cfg.data.val_batch_size,
        num_workers = cfg.data.worker,
        pin_memory = True
    )

    return train_dataloader, validate_dataloader, validate_dataset.name_list
