import torch
from dotmap import DotMap
from .lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR


def get_optimizer(cfg: DotMap, model):
    #params = filter(lambda p: p.requires_grad, model.parameters())
    vision_params = list(map(id, model.visual.parameters()))
    motion_params = list(map(id, model.motion.parameters()))
    fusion_params = list(map(id, model.fusion.parameters()))
    other_params = filter(lambda p: id(p) not in vision_params +
                          motion_params+fusion_params and p.requires_grad, model.parameters())

    params = []
    if cfg.network.visual.train:
        params += [{'params': model.visual.parameters(),
                    'lr': cfg.network.visual.lr}]
    if cfg.network.motion.train:
        params += [{'params': model.motion.parameters(),
                    'lr': cfg.network.motion.lr}]
    if cfg.network.fusion.train:
        params += [{'params': model.fusion.parameters(),
                    'lr': cfg.network.fusion.lr}]
    if cfg.network.other.train:
        params += [{'params': other_params, 'lr': cfg.network.other.lr}]

    if cfg.optim.optim == 'adam':
        optimizer = torch.optim.Adam(params,
                                     weight_decay=cfg.optim.weight_decay)
        print('Adam')
    elif cfg.optim.optim == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    cfg.optim.lr,
                                    momentum=cfg.optim.momentum,
                                    weight_decay=cfg.optim.weight_decay)
        print('SGD')
    elif cfg.optim.optim == 'adamw':
        optimizer = torch.optim.AdamW(params,
                                      weight_decay=cfg.optim.weight_decay)
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        print('AdamW')
    else:
        raise ValueError('Unknown optimizer: {}'.format(cfg.optim.optim))
    return optimizer


def get_lr_scheduler(cfg: DotMap, optimizer):
    if cfg.optim.type == 'cosine':
        lr_scheduler = WarmupCosineAnnealingLR(
            optimizer,
            cfg.optim.epochs,
            warmup_epochs=cfg.optim.lr_warmup_step
        )
    elif cfg.optim.type == 'multistep':
        if isinstance(cfg.optim.lr_decay_step, list):
            milestones = cfg.optim.lr_decay_step
        elif isinstance(cfg.optim.lr_decay_step, int):
            milestones = [
                cfg.optim.lr_decay_step * (i + 1)
                for i in range(cfg.optim.epochs //
                               cfg.optim.lr_decay_step)]
        else:
            raise ValueError("error learning rate decay step: {}".format(
                type(cfg.optim.lr_decay_step)))
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            milestones,
            warmup_epochs=cfg.optim.lr_warmup_step
        )
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(cfg.optim.type))
    return lr_scheduler
