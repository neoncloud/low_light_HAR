import cv2
import os
import sys
from torchvision.io import read_video
def dump_frames(vid_path, out_path):
    #video = cv2.VideoCapture(vid_path)
    #frames, _, _  = read_video(vid_path)
    cap = cv2.VideoCapture(vid_path)
    fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1
    # print('frames',frames.shape)
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
    #     ret, frame = cap.read()
    #     assert ret
    #     cv2.imwrite('{}/{:06d}.jpg'.format(out_full_path, i), frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    #     access_path = '{}/{:06d}.jpg'.format(vid_name, i)
    #     file_list.append(access_path)
    print('{} done'.format(vid_name))
    sys.stdout.flush()
    return fcount, out_full_path


if __name__ == '__main__':
    dataset_path = '/home/neoncloud/hmdb51_org'
    # name_id = {
    #     'drink':0,
    #     'jump' :1,
    #     'pick' :2,
    #     'pour' :3,
    #     'push' :4,
    #     'run'  :5,
    #     'sit'  :6,
    #     'stand':7,
    #     'turn' :8,
    #     'walk' :9,
    # }
    name_id = {
        'brush hair': 0,
        'cartwheel': 1,
        'catch': 2,
        'chew': 3,
        'clap': 4,
        'climb': 5,
        'climb stairs': 6,
        'dive': 7,
        'draw sword': 8,
        'dribble': 9,
        'drink': 10,
        'eat': 11,
        'fall floor': 12,
        'fencing': 13,
        'flic flac': 14,
        'golf': 15,
        'handstand': 16,
        'hit': 17,
        'hug': 18,
        'jump': 19,
        'kick ball': 20,
        'kick': 21,
        'kiss': 22,
        'laugh': 23,
        'pick': 24,
        'pour': 25,
        'pullup': 26,
        'punch': 27,
        'push': 28,
        'pushup': 29,
        'ride bike': 30,
        'ride horse': 31,
        'run': 32,
        'shake hands': 33,
        'shoot ball': 34,
        'shoot bow': 35,
        'shoot gun': 36,
        'sit': 37,
        'situp': 38,
        'smile': 39,
        'smoke': 40,
        'somersault': 41,
        'stand': 42,
        'swing baseball': 43,
        'sword exercise': 44,
        'sword': 45,
        'talk': 46,
        'throw': 47,
        'turn': 48,
        'walk': 49,
        'wave': 50
    }
    split_path = os.path.join(dataset_path, 'HMDB_split')
    train_img_path = os.path.join(split_path, 'train_img')
    val_img_path = os.path.join(split_path, 'val_img')
    test_img_path = os.path.join(split_path, 'test_img')
    if not os.path.exists(train_img_path):
        os.mkdir(train_img_path)
    if not os.path.exists(val_img_path):
        os.mkdir(val_img_path)
    if not os.path.exists(test_img_path):
        os.mkdir(test_img_path)
    train_vid_file = []
    val_vid_file = []
    test_vid_file = []

    for k, v in name_id.items():
        for i in range(1,4):
            with open(os.path.join(split_path, k+'_test_split'+str(i)+'.txt'), 'r') as file:
                for line in file:
                    file_name, cls = line.strip('\n').split(' ')[0:2]
                    print(cls)
                    if cls == '1':
                        img_path = train_img_path
                    elif cls == '2':
                        img_path = val_img_path
                    else:
                        img_path = test_img_path
                    vid = os.path.join(dataset_path, k, file_name)
                    num_frames, out_full_path = dump_frames(vid, img_path)
                    file_entry = (out_full_path, num_frames, v)
                    if cls == '1':
                        train_vid_file.append(file_entry)
                    elif cls == '2':
                        val_vid_file.append(file_entry)
                    else:
                        test_vid_file.append(file_entry)

    with open(os.path.join(dataset_path, 'train_full.txt'), 'w') as file:
        file.write('\n'.join('%s %s %s' % x for x in train_vid_file))
    with open(os.path.join(dataset_path, 'validate_full.txt'), 'w') as file:
        file.write('\n'.join('%s %s %s' % x for x in val_vid_file))
    with open(os.path.join(dataset_path, 'test_full.txt'), 'w') as file:
        file.write('\n'.join('%s %s %s' % x for x in test_vid_file))
