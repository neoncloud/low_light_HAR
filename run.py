import os
import torch
from torch.cuda.amp import GradScaler, autocast
#from torch.nn.utils import clip_grad_norm_
from model.network import convert_models_to_fp32
from model.kl_loss import KLLoss
from model.text_prompt import text_prompt
from model.optimizer import get_optimizer, get_lr_scheduler
from data.dataloder import get_dataloder
from util.save import best_saving, epoch_saving
from model.network import load, build_model
import yaml
import argparse
from dotmap import DotMap
from contextlib import nullcontext
from einops import rearrange
import time
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
import gc

# import debugpy
# debugpy.listen(("localhost", 5678))


def load_model():
    if cfg.resume is not None:
        state_dict = torch.load(cfg.resume, map_location='cuda')
        start_epoch = state_dict['epoch']
        model_ = build_model(
            state_dict=state_dict['model_state_dict'],
            pretrain=True,
            motion_layers=cfg.network.motion.num_layers,
            motion_layers_init=cfg.network.motion.init,
            train_visual=cfg.visual.train,
            T=cfg.data.seg_length,
            thres=cfg.network.motion.thres,
            alpha=cfg.network.other.alpha,
            fusion_type=cfg.network.fusion.type
        ).cuda()
        if args.train:
            optimizer = get_optimizer(cfg, model_)
            optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        else:
            optimizer = None
    elif cfg.pretrain is not None:
        state_dict = torch.load(cfg.pretrain)
        start_epoch = 1
        model_, _ = load(
            name=cfg.network.arch,
            jit=False,
            pretrain=True,
            motion_layers=cfg.network.motion.num_layers,
            motion_layers_init=cfg.network.motion.init,
            train_visual=cfg.visual.train,
            T=cfg.data.seg_length,
            thres=cfg.network.motion.thres,
            alpha=cfg.network.other.alpha,
            fusion_type=cfg.network.fusion.type
        )
        model_.load_state_dict(state_dict['model_state_dict'],False)
        if args.train:
            optimizer = get_optimizer(cfg, model_)
        else:
            optimizer = None
    else:
        raise NotImplementedError
    del state_dict
    if args.train:
        lr_scheduler = get_lr_scheduler(cfg, optimizer)
    else:
        lr_scheduler = None
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max')
    train_dataloader, validate_dataloader, name_list = get_dataloder(cfg)
    num_text_aug, text_tokenized = text_prompt(name_list)
    with torch.no_grad():
        all_text_features = model_.encode_text(
            rearrange(text_tokenized, 'c n d -> (c n) d').cuda())
        all_text_features = rearrange(
            all_text_features, '(c n) d -> c n d', n=num_text_aug)
    if cfg.optim.distributed:
        from torch.nn.parallel import DistributedDataParallel
        print(f"[{os.getpid()}] (local_rank = {local_rank}) training...")
        model_ = DistributedDataParallel(model_.cuda(), device_ids=[local_rank],output_device=local_rank)
    else:
        model_ = model_.cuda()

    return model_, optimizer, lr_scheduler, start_epoch, train_dataloader, validate_dataloader, num_text_aug, text_tokenized, all_text_features


def train():
    #criterion = torch.nn.CrossEntropyLoss()
    criterion_img = KLLoss()
    criterion_txt = KLLoss()

    if cfg.optim.amp:
        scaler = GradScaler()
        amp_ctx = autocast()
    else:
        scaler = None
        amp_ctx = nullcontext()
        convert_models_to_fp32(model)
    best_prec1 = 0.0
    for epoch in trange(start_epoch, cfg.optim.epochs):
        for i, data in enumerate(tqdm(train_dataloader)):

            # sample a prompt
            text_id = torch.randint(
                high=num_text_aug, size=(data['label'].shape[0],))
            if cfg.network.text.train:
                text_token = text_tokenized[text_id, data['label'], :].cuda()
            else:
                text_features = all_text_features[data['label'], text_id, :]

            grad_accu_ctx = model_.no_sync() if cfg.optim.distributed and (i+1) % cfg.optim.grad_accu != 0 else nullcontext()

            ########## FORWARD #############
            with amp_ctx, grad_accu_ctx:
                if cfg.network.text.train:
                    logits_per_image, logits_per_text = model(
                        data['frames'].cuda(), text=text_token.cuda())
                else:
                    logits_per_image, logits_per_text = model(
                        data['frames'].cuda(), text_features=text_features)
            ########## FORWARD #############

                # Make ground Truth (confussion matrix)
                label = data['label'].unsqueeze(-1)
                ground_truth = (0.99+0.01*torch.randn(label.shape[0]))*(torch.eq(label, label.T).to(torch.float16))
                #ground_truth += torch.rand_like(ground_truth)*0.01

                loss_img = criterion_img(logits_per_image, ground_truth.cuda())
                loss_txt = criterion_txt(logits_per_text, ground_truth.cuda())
                loss = (loss_img+loss_txt)/2

                if cfg.optim.amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (i+1) % cfg.optim.grad_accu == 0:
                    if cfg.optim.amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

            if local_rank == 0:
                total_steps = i*cfg.data.batch_size+epoch*len(train_dataloader)
                if total_steps % cfg.logging.write_freq == 0:
                    print('loss:', loss_img.cpu().item())
                    writer.add_scalar(
                        'loss', loss_img.cpu().item(), total_steps)

        if local_rank == 0:
            if epoch % cfg.logging.eval_freq == 0:
                with amp_ctx:
                    torch.cuda.empty_cache()
                    top_1,top_5 = eval()
                    writer.add_scalar('Top 1', top_1, epoch*len(train_dataloader))
                    writer.add_scalar('Top 5', top_5, epoch*len(train_dataloader))
                    print('Epoch: [{}/{}]: Top1: {}, Top5: {}'.format(epoch, cfg.optim.epochs, top_1, top_5))
                    torch.cuda.empty_cache()
                    gc.collect()
                is_best = top_1 > best_prec1
                best_prec1 = max(top_1, best_prec1)
                print('Testing: {}/{}'.format(top_1, best_prec1))
                lr_scheduler.step(best_prec1)
                if is_best:
                    print('Saving:')
                    best_saving(working_dir, epoch, model,
                                optimizer, lr_scheduler)

            if epoch % cfg.logging.save_freq == 0:
                print('Saving:')
                filename = "{}/last_model.pt".format(working_dir)
                epoch_saving(epoch, model, optimizer, lr_scheduler, filename)


@torch.no_grad()
def eval():
    model.eval()
    if cfg.network.text.train:
        all_class_features = model.encode_text(
            rearrange(text_tokenized, 'c n d -> (c n) d').cuda())
    else:
        all_class_features = rearrange(all_text_features, 'c n d -> (c n) d')
    num = 0
    corr_1 = 0.0
    corr_5 = 0.0
    for data in tqdm(validate_dataloader):
        b, t, c, h, w = data['frames'].size()
        label = data['label'].cuda().unsqueeze(-1)
        similarity = model.inference(
            data['frames'].cuda(), text_features=all_class_features)
        similarity = rearrange(
            similarity, 'b (n d) -> b n d', n=num_text_aug)
        similarity = similarity.mean(dim=1, keepdim=False)
        _, indices_1 = similarity.topk(1, dim=-1)
        _, indices_5 = similarity.topk(5, dim=-1)
        num += b
        corr_1 += torch.eq(indices_1, label).to(torch.float32).sum()
        corr_5 += torch.any(torch.eq(indices_5, label),
                            dim=-1).to(torch.float32).sum()
    top_1 = (corr_1 / num * 100).cpu().item()
    top_5 = (corr_5 / num * 100).cpu().item()
    return top_1, top_5


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

    if cfg.optim.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
    torch.manual_seed(cfg.seed+local_rank)
    torch.backends.cudnn.benchmark = True

    working_dir = os.path.join(cfg.logging.chpt_dir, cfg.network.arch,
                               cfg.data.dataset, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    writer = SummaryWriter(working_dir)

    model_, optimizer, lr_scheduler, start_epoch, train_dataloader, validate_dataloader, num_text_aug, text_tokenized, all_text_features = load_model()
    gc.collect()
    torch.cuda.empty_cache()
    if cfg.optim.distributed:
        model = model_.module
    else:
        model = model_
    if args.train:
        train()
    elif args.eval:
        top_1, top_5 = eval()
        print('Epoch: [{}/{}]: Top1: {}, Top5: {}'.format(0, cfg.optim.epochs, top_1, top_5))
