import torch

def epoch_saving(epoch, model, optimizer, lr_scheduler, filename):
    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict()
                    }, filename) #just change to your preferred folder/filename

def best_saving(working_dir, epoch, model, optimizer, lr_scheduler):
    best_name = '{}/model_best.pt'.format(working_dir)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict()
    }, best_name)  # just change to your preferred folder/filename