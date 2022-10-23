import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from model.network import build_model
from model.kl_loss import KLLoss
from model.text_prompt import text_prompt
from model.optimizer import get_optimizer
from data.dataloder import get_dataloder
from util.save import best_saving, epoch_saving
import yaml
import argparse
from dotmap import DotMap
from contextlib import nullcontext

def train(cfg: DotMap):
    train_dataloader, validate_dataloader, name_list = get_dataloder(cfg)
    num_text_aug, text_tokenized = text_prompt(name_list)
    print(text_tokenized.shape)
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = KLLoss()
    if cfg.resume is not None:
        state_dict = torch.load(cfg.resume)
        start_epoch = state_dict['epoch']
        model = build_model(
            state_dict=state_dict['model_state_dict'],
            pretrain=True,
            motion_layers=cfg.network.motion.num_layers,
            motion_layers_init=False,
            train_visual=cfg.visual.train
        ).train().cuda()
        optimizer = get_optimizer(cfg, model)
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    elif cfg.pretrain is not None:
        state_dict = torch.load(cfg.pretrain)
        start_epoch = 1
        model = build_model(
            state_dict=state_dict['model_state_dict'],
            pretrain=True,
            motion_layers=cfg.network.motion.num_layers,
            motion_layers_init=True,
            train_visual=cfg.visual.train
        ).train().cuda()
        optimizer = get_optimizer(cfg, model)
    else:
        raise NotImplementedError
    
    if cfg.optim.amp:
        scaler = GradScaler()
        amp_ctx = autocast()
    else:
        scaler = None
        amp_ctx = nullcontext()

    for epoch in range(start_epoch, cfg.optim.epochs):
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            text_id = torch.randint(high=num_text_aug,size=(data['label'].shape[0],))
            texts = text_tokenized[text_id, data['label'], :]
            with amp_ctx:
                logits_per_image, _ = model(data['frames'].cuda(), texts.cuda())
                label = data['label'].unsqueeze(-1)
                ground_truth = torch.eq(label, label.T).to(torch.float16)
                loss = criterion(logits_per_image, ground_truth.cuda())
                if cfg.optim.amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true')
    group.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = DotMap(cfg)
    
    if args.train:
        train(cfg)
