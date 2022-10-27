import cv2
import os
import sys

if __name__ == '__main__':
    dataset_path = '/home/neoncloud/hmdb_full'
    name_id = {
        'brush_hair': 0,
        'cartwheel': 1,
        'catch': 2,
        'chew': 3,
        'clap': 4,
        'climb': 5,
        'climb_stairs': 6,
        'dive': 7,
        'draw_sword': 8,
        'dribble': 9,
        'drink': 10,
        'eat': 11,
        'fall_floor': 12,
        'fencing': 13,
        'flic_flac': 14,
        'golf': 15,
        'handstand': 16,
        'hit': 17,
        'hug': 18,
        'jump': 19,
        'kick_ball': 20,
        'kick': 21,
        'kiss': 22,
        'laugh': 23,
        'pick': 24,
        'pour': 25,
        'pullup': 26,
        'punch': 27,
        'push': 28,
        'pushup': 29,
        'ride_bike': 30,
        'ride_horse': 31,
        'run': 32,
        'shake_hands': 33,
        'shoot_ball': 34,
        'shoot_bow': 35,
        'shoot_gun': 36,
        'sit': 37,
        'situp': 38,
        'smile': 39,
        'smoke': 40,
        'somersault': 41,
        'stand': 42,
        'swing_baseball': 43,
        'sword_exercise': 44,
        'sword': 45,
        'talk': 46,
        'throw': 47,
        'turn': 48,
        'walk': 49,
        'wave': 50
    }
    split_path = os.path.join(dataset_path, 'HMDB_split')
    train_vid_file = []
    val_vid_file = []

    for k, v in name_id.items():
        for i in range(1,4):
            with open(os.path.join(split_path, k+'_test_split'+str(i)+'.txt'), 'r') as file:
                for line in file:
                    file_name, cls = line.strip('\n').split(' ')[0:2]
                    print(cls)
                    vid = os.path.join(dataset_path, k, file_name)
                    file_entry = (vid, v)
                    if cls == '1' or cls == '0':
                        train_vid_file.append(file_entry)
                    elif cls == '2':
                        val_vid_file.append(file_entry)

    with open(os.path.join(dataset_path, 'train_full.txt'), 'w') as file:
        file.write('\n'.join('%s %s' % x for x in train_vid_file))
    with open(os.path.join(dataset_path, 'validate_full.txt'), 'w') as file:
        file.write('\n'.join('%s %s' % x for x in val_vid_file))
