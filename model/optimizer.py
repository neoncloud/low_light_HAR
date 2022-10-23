import torch
from dotmap import DotMap

def get_optimizer(cfg:DotMap, model):
    params = list(model.motion.parameters())
    if cfg.visual.train:
        params += list(model.visual.parameters())
    if cfg.optim.optim == 'adam':
        optimizer = torch.optim.Adam(params,
                               lr=cfg.optim.lr,
                               weight_decay=0.2)
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

def get_lr_scheduler(cfg:DotMap, optimizer):
    if cfg.solver.type == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.solver.epochs,
            T_mult=cfg.solver.lr_warmup_step
        )
    elif cfg.solver.type == 'multistep':
        if isinstance(cfg.solver.lr_decay_step, list):
            milestones = cfg.solver.lr_decay_step
        elif isinstance(cfg.solver.lr_decay_step, int):
            milestones = [
                cfg.solver.lr_decay_step * (i + 1)
                for i in range(cfg.solver.epochs //
                               cfg.solver.lr_decay_step)]
        else:
            raise ValueError("error learning rate decay step: {}".format(type(cfg.solver.lr_decay_step)))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones,
            warmup_epochs=cfg.solver.lr_warmup_step
        )
    else:
        raise ValueError('Unknown lr scheduler: {}'.format(cfg.solver.type))
    return lr_scheduler