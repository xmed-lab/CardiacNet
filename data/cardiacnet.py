#jyangcu@connect.ust.hk
import os
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image, ImageFilter

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import torchvision

import transform as transform

import matplotlib.pyplot as plt
import nibabel as nib

np.set_printoptions(threshold=np.inf)
random.seed(8888)
np.random.seed(8888)


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_size, mask_ratio, train_mode):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        if not isinstance(mask_size, tuple):
            mask_size = (mask_size,) * 2

        self.height, self.width, _ = input_size
        self.length = 1
        self.mask_h_size, self.mask_w_size = mask_size
        self.num_patches = (self.height//self.mask_h_size) * (self.width//self.mask_w_size) * self.length
        self.empty_image = np.ones((self.num_patches, self.mask_h_size, self.mask_w_size))
        self.num_mask = int(mask_ratio * self.num_patches)
        self.train_mode = train_mode

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        if self.train_mode == 'train':
            select_num_mask = int(self.num_mask * (np.random.randint(20, 80) / 100))
        else:
            select_num_mask = int(self.num_mask * 0.75)

        mask = np.hstack([
            np.ones(self.num_patches - select_num_mask),
            np.zeros(select_num_mask),
        ])
        np.random.shuffle(mask)
        mask = np.expand_dims(mask, (1,2))
        mask = np.repeat(mask, self.mask_h_size, axis=(1))
        mask = np.repeat(mask, self.mask_w_size, axis=(2))
        mask = self.empty_image * mask
        return mask # [196]


class CardiacNet_Dataset(Dataset):
    def __init__(self, args, select_set=['ASD', 'Non-ASD'], is_video=True, is_train=True, is_vaild=False, is_test=False):
        self.args = args
        self.select_set = select_set
        self.is_video = is_video
        self.is_train = is_train

        data_list = list()
        data_dict, self.data_all = self.mapping_data()
        for data_type in select_set:
            data_list+=(list(data_dict[data_type].keys()))

        train_list = random.sample(data_list, int(len(data_list) * 0.1))
        test_vaild_list = list(set(data_list).difference(set(train_list)))
        vaild_list = random.sample(test_vaild_list, int(len(test_vaild_list) * 0.5))
        test_list = list(set(test_vaild_list).difference(set(vaild_list)))

        if self.is_train:
            self.data_select = train_list
        elif self.is_vaild:
            self.data_select = vaild_list
        elif self.is_test:
            self.data_select = test_list

        if is_train:
            self.transform = transforms.Compose([
                                transform.ToTensorVideo(),
                                #transform.NormalizeVideo(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
                                transform.ResizedVideo((144, 144)),
                                transform.RandomCropVideo((112, 112)),
                                transform.RandomHorizontalFlipVideo(),
                                ])
        else:
            self.transform = transforms.Compose([
                                transform.ToTensorVideo(),
                                #transform.NormalizeVideo(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
                                transform.ResizedVideo((144, 144)),
                                transform.CenterCropVideo((112, 112)),
                                ])

        self.masked_position_generator = RandomMaskingGenerator(input_size=args.image_size, mask_size=args.mask_size, mask_ratio=args.mask_ratio, train_mode=mode)

    def __getitem__(self, index):
        if self.is_train:
            index = index // 4
        def get_case_attr(index):
            case_dir = self.data_select[index]
            case_attr = self.data_all[case_dir]
            return case_dir, case_attr

        frame_list = list()
        if self.is_video:
            case_dir, case_attr = get_case_attr(index)

            nii_file = nib.load(case_dir)
            image_data = nii_file.get_fdata()
            print(image_data.shape)
            w, h, video_length = image_data.shape
            if video_length >= self.args.image_size[-1]:
                if self.is_train:
                    if self.args.max_sample_rate is not None:
                        max_sample_rate = self.args.max_sample_rate
                        if max_sample_rate > video_length // self.args.image_size[-1]:
                            max_sample_rate = video_length // self.args.image_size[-1]
                    else:
                        max_sample_rate = video_length // self.args.image_size[-1]

                    if self.args.min_sample_rate is not None:
                        if self.args.min_sample_rate < max_sample_rate:
                            min_sample_rate = self.args.min_sample_rate
                    else:
                        min_sample_rate = max_sample_rate // 2

                    if max_sample_rate >= 8:
                        sample_rate = random.randint(min_sample_rate, 8)
                    elif max_sample_rate > 4 and max_sample_rate <= 8:
                        sample_rate = random.randint(min_sample_rate, max_sample_rate)
                    elif max_sample_rate > 2 and max_sample_rate <= 4:
                        sample_rate = random.randint(min_sample_rate, max_sample_rate)
                    elif max_sample_rate <= 2 and max_sample_rate > 1:
                        sample_rate = random.randint(1, max_sample_rate)
                    else:
                        sample_rate = 1
                    
                    start_idx = random.randint(0, (video_length-sample_rate * self.args.image_size[-1]))

                elif self.is_train is False:
                    sample_rate = 1
                    start_idx = 0
                
                frame_list = image_data[:,:,start_idx : start_idx + sample_rate * self.args.image_size[-1] : sample_rate]
            
            # elif video_length < self.args.image_size[-1]:
            #     view_frame_list = image_data
            #     for _ in range(self.args.image_size[-1]-video_length):
            #         view_frame_list.append(np.zeros_like(view_frame_list[-1].shape))
            video = np.transpose(np.array(frame_list), (2, 0, 1))
            bboxs = self.mask_find_bboxs(np.where(video[0] != 0))
            video = video[:, bboxs[0]:bboxs[1], bboxs[2]+15:bboxs[3]]
            video = self.transform(np.expand_dims(video, axis=-1).astype(np.uint8))

            # torchvision.utils.save_image(video, os.path.join("/home/jyangcu/Pulmonary_Arterial_Hypertension/results", f"example_{video.shape[0]}.jpg"), nrow=video.shape[0])
            # print('saved')

            class_type = case_attr['class']

        return video, class_type, self.masked_position_generator()

    def __len__(self):
        if self.is_train:
            return len(self.data_select) * 4
        else:
            return len(self.data_select)

    def mapping_data(self):
        data_all = {}
        data_dict = {'Non-ASD':{}, 'ASD':{}, 'Non-PAH':{},'PAH':{}}
        # Mapping All the Data According to the Hospital Center
        for dir in os.listdir(self.args.dataset_path):
            dir_path = os.path.join(self.args.dataset_path, dir)
            if os.path.isdir(dir_path):
                data_dict[dir] = {}
                # Mapping All the Data for a Hospital Center According to the Device
                for sub_dir in os.listdir(dir_path):
                    sub_dir_path = os.path.join(dir_path, sub_dir)
                    for case_dir in os.listdir(sub_dir_path):
                        if 'label' in case_dir:
                            pass
                        else:
                            if 'Non-ASD' in sub_dir_path:
                                type_class = 0
                                data_dict['Non-ASD'][sub_dir_path+'/'+case_dir] = {'class':type_class}
                            if 'ASD' in sub_dir_path and 'Non-ASD' not in sub_dir_path:
                                type_class = 1
                                data_dict['ASD'][sub_dir_path+'/'+case_dir] = {'class':type_class}
                            if 'Non-PAH' in sub_dir_path:
                                type_class = 2
                                data_dict['Non-PAH'][sub_dir_path+'/'+case_dir] = {'class':type_class}
                            if 'PAH' in sub_dir_path and 'Non-PAH' not in sub_dir_path:
                                type_class = 3
                                data_dict['PAH'][sub_dir_path+'/'+case_dir] = {'class':type_class}
                            data_all[sub_dir_path+'/'+case_dir] = {'class':type_class}
        return data_dict, data_all

    def mask_find_bboxs(self, mask):
        bbox = np.min(mask[0]), np.max(mask[0]), np.min(mask[1]), np.max(mask[1])
        return bbox

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--image-size', type=int, default=(224,224,32), help='Image height and width (default: 256)')
    parser.add_argument('--crop_length', type=int, default=16, help='Video length (default: 256)')
    parser.add_argument('--min_sample_rate', type=int, default=1, help='The min sampling rate for the video')
    parser.add_argument('--max_sample_rate', type=int, default=1, help='The max sampling rate for the video')
    parser.add_argument('--blurring', type=bool, default=True, help='Whether blur the image')
    parser.add_argument('--mask-size', type=int, default=8, help='The size of mask patch (default: 16)')
    parser.add_argument('--mask-ratio', type=float, default=0.6, help='The ratio of masking area in an image (default: 0.75)')
    args = parser.parse_args()
    args.dataset_path = r'/home/jyangcu/Dataset/CardiacNet'

    data_dict = dict()

    from monai.data import DataLoader
    from torchvision import utils as vutils
    train_ds = CardiacNet_Dataset(args, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)
    from einops import rearrange
    count = 1
    for img, mask, class_type in tqdm(train_loader):
        count += 1
        if count > 1000:
            break
        else: 
            pass