import cv2, os, sys
from torchvision.io import read_video
# from .dark_enhance import DarkEnhance

# enhancer = DarkEnhance().cuda()
def dump_frames(vid_path, out_path):
    #video = cv2.VideoCapture(vid_path)
    #frames, _, _  = read_video(vid_path)
    cap = cv2.VideoCapture(vid_path)
    fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1
    #print('frames',frames.shape)
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    #frames = enhancer.forward(frames.cuda())
    #fcount = int(frames.shape[0])
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    # file_list = []
    # for i in range(1,fcount+1):
    #     cv2.imwrite('{}/{:06d}.jpg'.format(out_full_path, i), frames[i-1,...].cpu().numpy()[...,::-1], [cv2.IMWRITE_JPEG_QUALITY, 75])
    #     access_path = '{}/{:06d}.jpg'.format(vid_name, i)
    #     file_list.append(access_path)
    print('{} done'.format(vid_name))
    sys.stdout.flush()
    return fcount,out_full_path

if __name__ == '__main__':
    dataset_path = '/home/neoncloud/hmdb51_org'
    name_id = {
        'drink':0,
        'jump' :1,
        'pick' :2,
        'pour' :3,
        'push' :4,
        'run'  :5,
        'sit'  :6,
        'stand':7,
        'turn' :8,
        'walk' :9,
    }
    split_path = os.path.join(dataset_path,'HMDB_split')
    train_img_path = os.path.join(split_path,'train_img')
    val_img_path = os.path.join(split_path,'val_img')
    test_img_path = os.path.join(split_path,'test_img')
    if not os.path.exists(train_img_path):
        os.mkdir(train_img_path)
    if not os.path.exists(val_img_path):
        os.mkdir(val_img_path)
    if not os.path.exists(test_img_path):
        os.mkdir(test_img_path)
    train_vid_file = []
    val_vid_file = []
    test_vid_file = []

    for k,v in name_id.items():
        with open(os.path.join(split_path,k+'.txt'),'r') as file:
            for line in file:
                file_name, cls = line.strip('\n').split(' ')[0:2]
                print(cls)
                if cls == '1':
                    img_path = train_img_path
                elif cls == '2':
                    img_path = val_img_path
                else:
                    img_path = test_img_path
                vid = os.path.join(dataset_path,k,file_name)
                num_frames,out_full_path = dump_frames(vid,img_path)
                file_entry = (out_full_path,num_frames,v)
                if cls == '1':
                    train_vid_file.append(file_entry)
                elif cls == '2':
                    val_vid_file.append(file_entry)
                else:
                    test_vid_file.append(file_entry)

    with open(os.path.join(dataset_path,'train_.txt'), 'w') as file:
        file.write('\n'.join('%s %s %s' % x for x in train_vid_file))
    with open(os.path.join(dataset_path,'validate_.txt'), 'w') as file:
        file.write('\n'.join('%s %s %s' % x for x in val_vid_file))
    with open(os.path.join(dataset_path,'test_.txt'), 'w') as file:
        file.write('\n'.join('%s %s %s' % x for x in test_vid_file))

    # val_img_path = os.path.join(dataset_path,'val_img')
    # if not os.path.exists(val_img_path):
    #     os.mkdir(val_img_path)
    # val_vid_file = []
    # with open(os.path.join(dataset_path,'validate.txt'),'r') as file:
    #     for line in file:
    #         cls = line.split('	')[1].strip('\n')
    #         vid = os.path.join(val_path,line.split('	')[2].strip('\n'))
    #         num_frames,out_full_path = dump_frames(vid,val_img_path)
    #         val_vid_file.append((out_full_path,num_frames,cls))

    # with open(os.path.join(dataset_path,'validate_.txt'), 'w') as file:
    #     file.write('\n'.join('%s %s %s' % x for x in val_vid_file))