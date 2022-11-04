import cv2, os, sys
from torchvision.io import read_video
from dark_enhance import DarkEnhance
import torch

enhancer = DarkEnhance().cuda()
def dump_frames(vid_path, out_path):
    #video = cv2.VideoCapture(vid_path)
    frames, _, _  = read_video(vid_path)
    #print('frames',frames.shape)
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    frames = enhancer.forward(frames.cuda()).to(dtype=torch.uint8)
    fcount = int(frames.shape[0])
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    file_list = []
    for i in range(1,fcount+1):
        cv2.imwrite('{}/{:06d}.jpg'.format(out_full_path, i), frames[i-1,...].cpu().numpy()[...,::-1], [cv2.IMWRITE_JPEG_QUALITY, 75])
        access_path = '{}/{:06d}.jpg'.format(vid_name, i)
        file_list.append(access_path)
    print('{} done'.format(vid_name))
    sys.stdout.flush()
    return fcount,out_full_path

if __name__ == '__main__':
    dataset_path = '/home/neoncloud/low_light_video'
    train_path = os.path.join(dataset_path,'train')
    val_path = os.path.join(dataset_path,'validate')
    train_img_path = os.path.join(dataset_path,'train_img')
    if not os.path.exists(train_img_path):
        os.mkdir(train_img_path)
    train_vid_file = []
    with open(os.path.join(dataset_path,'train.txt'),'r') as file:
        for line in file:
            cls = line.split('	')[1].strip('\n')
            vid = os.path.join(train_path,line.split('	')[2].strip('\n'))
            print(vid)
            num_frames,out_full_path = dump_frames(vid,train_img_path)
            train_vid_file.append((out_full_path,num_frames,cls))

    with open(os.path.join(dataset_path,'train_.txt'), 'w') as file:
        file.write('\n'.join('%s %s %s' % x for x in train_vid_file))

    val_img_path = os.path.join(dataset_path,'val_img')
    if not os.path.exists(val_img_path):
        os.mkdir(val_img_path)
    val_vid_file = []
    with open(os.path.join(dataset_path,'validate.txt'),'r') as file:
        for line in file:
            cls = line.split('	')[1].strip('\n')
            vid = os.path.join(val_path,line.split('	')[2].strip('\n'))
            num_frames,out_full_path = dump_frames(vid,val_img_path)
            val_vid_file.append((out_full_path,num_frames,cls))

    with open(os.path.join(dataset_path,'validate_.txt'), 'w') as file:
        file.write('\n'.join('%s %s %s' % x for x in val_vid_file))