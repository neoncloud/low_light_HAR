# Human Action Recognition in the Dark using CLIP
[Colab](https://colab.research.google.com/drive/14x5zyJFYTCSXnbHqa6Z7_ZAay7uLlTKi?usp=sharing)

My name is Shuai Chenhao.

This repo is for NTU EE6222 Assignment: Action Recognition in the Dark

Lecturer: Dr. Xu Yuecong

The design and coding of the network is heavily referenced from [ActionCLIP](https://github.com/sallymmx/ActionCLIP)
# Performance
On [ARID](https://xuyu0010.github.io/arid.html) dataset
* Top1: `82.1875`
* Top5: `99.375`

# Checkpoints
I will make it available after 09 Nov 2022 :)

# Requirement
```
pip3 install torch torchvison tqdm yaml dotmap einops av
```
# Dataset
## Pre-processing
Currently I have implemented two kind of datasets, `VideoDataset` to read the entire video, and `VideoFrameDataset` to read sequences of images that have been processed in advance. My recommendation is to use VideoFrameDataset to reduce memory usage and i/o overhead.

I have provided a small script `util/video2frames.py` to break the video into frames and record the path of the video. This script will also pre-process the video frames using the dark light enhancement algorithm proposed in this work, if you don't need it, please comment out the code about the enhancer: `frames = enhancer.forward(frames.cuda())`.

## Dataset structure
### VideoFrameDataset
For the `VideoFrameDataset`, once you have finished preparing the dataset, the dataset directory should look like this:
```
├── mapping_table.txt
├── validate_.txt
├── train_.txt
├── train_img
│   ├── Drink_10_1
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   ├── 000003.jpg
│   │   ├── 000004.jpg
        ...
├── val_img
│   ├── Drink_10_1
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   ├── 000003.jpg
│   │   ├── 000004.jpg
        ...
```

`mapping_table.txt` should be a csv file with lable-name pair. CLIP model will read `name` to construct text prompt. An example of `mapping_table.txt` is
```
id,name
0,Drink
1,Jump
2,Pick
3,Pour
4,Push
```

An example of `train_.txt` or `validate_.txt`:
```
train_img/Drink_10_1 129 0
train_img/Drink_10_10 61 0
train_img/Drink_10_11 62 0
train_img/Drink_10_12 60 0
train_img/Drink_10_13 60 0
train_img/Drink_10_14 69 0
train_img/Drink_10_15 100 0
train_img/Drink_10_2 77 0
train_img/Drink_10_3 119 0
train_img/Drink_10_4 117 0
```
The first column is path to image folder, and it should be a relative path. The second column is frame number count. The third column is label id. These files are automatically generated by pre-processing script.

### VideoDataset
For the `VideoDataset`, the dataset directory should look like this:
```
├── mapping_table.txt
├── validate_.txt
├── train_.txt
├── train_img
│   ├── Drink_10_1.mp4
│   ├── Drink_10_2.mp4
        ...
├── val_img
│   ├── Drink_11_1.mp4
│   ├── Drink_11_2.mp4
        ...
```
An example of `train_.txt` or `validate_.txt`:
```
brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0.avi 0
brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_1.avi 0
brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_2.avi 0
brush_hair/Aussie_Brunette_Brushing_Hair_II_brush_hair_u_nm_np1_ba_goo_4.avi 0
brush_hair/Aussie_Brunette_Brushing_Hair_II_brush_hair_u_nm_np1_ri_med_3.avi 0
brush_hair/Aussie_Brunette_Brushing_Hair_II_brush_hair_u_nm_np2_le_goo_0.avi 0
brush_hair/Aussie_Brunette_Brushing_Hair_II_brush_hair_u_nm_np2_le_goo_1.avi 0
brush_hair/Aussie_Brunette_Brushing_Hair_II_brush_hair_u_nm_np2_le_med_2.avi 0
brush_hair/Blonde_being_brushed_brush_hair_f_nm_np2_ri_med_0.avi 0
brush_hair/Blonde_being_brushed_brush_hair_u_cm_np2_ri_med_1.avi 0
```
The last column is label id.

# Config files

The configuration files are placed in the config folder. Modify the following entries to point to your own dataset and checkpoints.
```yaml
resume: ./checkpoints/arid/2022_11_03_06_05_00/model_best.pt
pretrain:
data:
  num_classes: 10
  image_tmpl: '{:06d}.jpg' #file name template
  train_list: /home/neoncloud/low_light_video/train_.txt
  val_list: /home/neoncloud/low_light_video/validate_.txt
  label_list: /home/neoncloud/low_light_video/mapping_table.txt
```

# RUN
## Run evaluation
```shell
python run.py --eval --config ./config/arid_test.yaml
```
You can also use `run_script.py` to evaluate with `*.mp4` files in a folder. It will output predicted labels:

```shell
python run_script.py --config ./config/arid_test.yaml --path /home/neoncloud/low_light_video/test
```

## Run training
```shell
python run.py --train --config ./config/arid_train.yaml
```

## Distributed training
Edit config file:
```yaml
optim:
  distributed: True
```
And launch with `torchrun`
```shell
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 run.py --train --config ./config/arid_train.yaml
```
