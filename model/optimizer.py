import torch
from dotmap import DotMap
from .lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR


def get_optimizer(cfg: DotMap, model):
    #params = filter(lambda p: p.requires_grad, model.parameters())
    vision_params = list(map(id, model.visual.parameters()))
    temporal_params = list(map(id, model.temporal.parameters()))
    other_params = filter(lambda p: id(p) not in vision_params and id(
        p) not in temporal_params and p.requires_grad, model.parameters())
    params = [{'params': other_params},
              {'params': model.visual.parameters(), 'lr': cfg.optim.lr *
               cfg.optim.f_ratio} if cfg.network.visual.train else {},
              {'params': model.temporal.parameters(), 'lr': cfg.optim.lr * cfg.optim.f_ratio} if cfg.network.temporal.train else {}]
    if cfg.optim.optim == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr=cfg.optim.lr,
                                     weight_decay=0.001)
        print('Adam')
    elif cfg.optim.optim == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    cfg.optim.lr,
                                    momentum=cfg.optim.momentum,
                                    weight_decay=cfg.optim.weight_decay)
        print('SGD')
    elif cfg.optim.optim == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=cfg.optim.lr,
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
