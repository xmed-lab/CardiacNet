#jyangcu@connect.ust.hk
import os
import random
import numpy as np

import SimpleITK as sitk
from tqdm import tqdm
from shutil import copyfile
import cv2

import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
import transform as transform

import shutil

np.set_printoptions(threshold=np.inf)
random.seed(7777)
np.random.seed(7777)


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_size, mask_ratio, train_mode):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        if not isinstance(mask_size, tuple):
            mask_size = (mask_size,) * 2

        self.height, self.width, self.length = input_size
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
        if self.train_mode:
            select_num_mask = int(self.num_mask * (np.random.randint(50, 80) / 100))
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


# data_type=['normal','middle','slight','severe'], ['ASD','None_ASD']

class CardiacNet(Dataset):
    def __init__(self, args, infos, data_type=['normal','pah'], is_train=True, is_test=False, set_select=['rmyy','gy','shph','szfw'], view_num=['4'], is_video=True):
        self.root = args.dataset_path
        self.data_type = data_type
        self.set_select = set_select
        self.view_num = view_num
        self.is_train = is_train
        self.is_video = is_video
        self.is_test  = is_test
        self.crop_length = args.image_size[2]

        if is_train and not is_test:
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
    
        self.masked_position_generator = RandomMaskingGenerator(input_size=args.image_size, mask_size=args.mask_size, mask_ratio=args.mask_ratio, train_mode=is_train)

        self.all_image, self.all_video = self.get_dict(infos, data_type)
        if self.is_video:
            self.all_data = list(self.all_video['all'].keys())
        else:
            self.all_data = list(self.all_image['all'].keys())

        self.num_data = len(self.all_data)
        
        self.train_list = self.all_data[:int(self.num_data * 0.9)]
        self.valid_list = list(set(self.all_data).difference(set(self.train_list)))

        # if os.path.exists('./datasets/train_{}.npy'.format(data_type)):
        #     print("Loading the training and validation samples.")
        #     self.train_list = np.load('./datasets/train_{}.npy'.format(data_type)).tolist()
        #     self.valid_list = np.load('./datasets/valid_{}.npy'.format(data_type)).tolist()
        #     print("Finish Files Loading")
        # else:
        #     print("Cannot Find the File for Training Samples, Generate New Training and Validation Samples.")
        #     train_save = np.array(self.train_list)
        #     valid_save = np.array(self.valid_list)
        #     np.save('./datasets/train_{}.npy'.format(data_type), train_save)
        #     np.save('./datasets/valid_{}.npy'.format(data_type), valid_save)
        #     print("Finish Files save.")

        self.data_list = self.train_list if is_train else self.valid_list
        self.data_list.sort()

    def __getitem__(self, index):
        def get_frame_list(index):
            video_name = self.data_list[index]
            frame_list, mpap, pasp = self.all_video['all'][video_name]['images'], self.all_video['all'][video_name]['mPAP'], self.all_video['all'][video_name]['pasp']
            return frame_list, mpap, pasp

        index = index if self.is_train and not self.is_test else index
        if self.is_video:
            frame_list = list()
            while len(frame_list) < self.crop_length:
                frame_list, mpap, pasp = get_frame_list(index)
                if len(frame_list) < self.crop_length:
                    index = random.randint(0, len(self.data_list))
                    index = index if self.is_train else index
                new_index = index

            video_name = self.data_list[new_index]
            video = list()

            for frame in frame_list:
                video.append(np.array(Image.open(video_name+frame)))
            bboxs = self.mask_find_bboxs(np.where(video[0] != 0))
            video = np.stack(video, axis=0)
            video = video[:, bboxs[0]:bboxs[1], bboxs[2]+15:bboxs[3]]
            
            video_length = video.shape[0]
            if self.is_train:
                max_sample_rate = video_length // self.crop_length
                if max_sample_rate > 4:
                    sample_rate = random.randint(2, 4)
                elif max_sample_rate <= 4 and max_sample_rate > 2:
                    sample_rate = random.randint(2, max_sample_rate)
                else:
                    sample_rate = 2
            else:
                sample_rate = 2
            start_idx = random.randint(0, (video_length-sample_rate*self.crop_length))
            video = video[start_idx:start_idx+sample_rate*self.crop_length:sample_rate]
            # video = self.transform(np.repeat(np.expand_dims(video, axis=-1), 3, axis=-1))
            video = self.transform(np.expand_dims(video, axis=-1))

            return video, self.masked_position_generator(), float(mpap), float(pasp)

        else:
            img_name = self.data_list[index]
            img_name_split = img_name.split('/')
            img_name_print = img_name_split[-3][:-6]
            img_order = img_name_split[-1][:-4]
            img_name_size = self.set_select[0] + '_' + '_'.join(img_name_split[-4].split('_')[-2:])

            mpap, pasp = self.data[img_name]['mPAP'], self.data[img_name]['pasp']
            img = Image.open(img_name)
            img = np.array(img)
            bboxs = self.mask_find_bboxs(np.where(img != 0))
            img = img[bboxs[0]:bboxs[1], bboxs[2]+15:bboxs[3]]
            img = np.array(self.transform(Image.fromarray(img)))
            img = (img / 127.5 - 1.0).astype(np.float32)

            if self.is_train:
                return np.expand_dims(img, axis=0), self.masked_position_generator(), float(mpap), float(pasp)
            else:
                return np.expand_dims(img, axis=0), self.masked_position_generator(), float(mpap), float(pasp), img_name_size+'_'+img_name_print, img_name_size+'_'+img_name_print+'_'+img_order

    def __len__(self):
        return len(self.data_list) if self.is_train and not self.is_test else len(self.data_list)

    def mask_find_bboxs(self, mask):
        bbox = np.min(mask[0]), np.max(mask[0]), np.min(mask[1]), np.max(mask[1])
        return bbox

    def get_dict(self, infos, data_type):
        
        def is_number(s):
            if isinstance(s, str):
                return False

            if s == 'nan':
                return False
            
            if np.isnan(s):
                return False
            
            try:
                float(s)
                return True
            except ValueError:
                pass
        
            try:
                import unicodedata
                unicodedata.numeric(s)
                return True
            except (TypeError, ValueError):
                pass
        
            return False
        
        def detech_none_digit_value(value):
            if is_number(value):
                return value
            else:
                return 0

        selected_dict = dict()
        all_images = {'normal':{}, 'middle':{}, 'slight':{}, 'severe':{}, 'pah':{}, 'ASD':{}, 'None_ASD': {}, 'ASD-severe': {}}
        all_videos = {'normal':{}, 'middle':{}, 'slight':{}, 'severe':{}, 'pah':{}, 'ASD':{}, 'None_ASD': {}, 'ASD-severe': {}}
        a_cases = 0
        b_cases = 0
        c_cases = 0
        d_cases = 0
        data_num = 0
        for k, v in infos.items():
            if v['dataset_name'] in self.set_select:
                selected_dict[k] = {}
                selected_dict[k]['images'] = v['views_images']
                selected_dict[k]['masks']  = v['views_labels']
                selected_dict[k]['fold'] = v['fold']

                selected_dict[k]['Vmax'] = v['Vmax']
                selected_dict[k]['dataset_name'] = v['dataset_name']
                data_num += 1
                if v['dataset_name'] == 'rmyy':
                    # if (self.is_number(v['RHC-mPAP']) or self.is_number(v['RHC-pasp'])):
                    mPAP  = detech_none_digit_value(v['RHC-mPAP'])
                    pasp  = detech_none_digit_value(v['RHC-pasp'])
                    EmPAP = detech_none_digit_value(v['Echo-mPAP'])
                    Epasp = detech_none_digit_value(v['Echo-pasp'])
                
                elif v['dataset_name'] == 'gy' or v['dataset_name'] == 'shph':
                    # if (self.is_number(v['R-mpap']) or self.is_number(v['R-pasp'])):
                    mPAP  = detech_none_digit_value(v['R-mpap'])
                    pasp  = detech_none_digit_value(v['R-pasp'])
                    EmPAP = detech_none_digit_value(v['E-mpap'])
                    Epasp = detech_none_digit_value(v['E-pasp'])

                elif v['dataset_name'] == 'szfw':
                    # if (self.is_number(v['R-mpap']) or self.is_number(v['R-pasp'])):
                    mPAP  = detech_none_digit_value(v['R-mPAP'])
                    pasp  = detech_none_digit_value(v['R-pasp'])
                    EmPAP = detech_none_digit_value(v['E-mPAP'])
                    Epasp = detech_none_digit_value(v['E-pasp'])

                selected_dict[k]['mPAP']  = mPAP
                selected_dict[k]['pasp']  = pasp
                selected_dict[k]['EmPAP'] = EmPAP
                selected_dict[k]['Epasp'] = Epasp
                if 'ASD' in v:
                    selected_dict[k]['ASD'] = v['ASD']
                else:
                    selected_dict[k]['ASD'] = None
        
                for id in list(selected_dict.keys()):
                    view_path = selected_dict[id]['images']
                    mPAP = selected_dict[id]['mPAP']
                    pasp = selected_dict[id]['pasp']
                    EmPAP= selected_dict[id]['EmPAP']
                    Epasp= selected_dict[id]['Epasp']
                    ASD  = selected_dict[id]['ASD']

                if mPAP == 0 and pasp == 0:
                    if EmPAP != 0:
                        mPAP = EmPAP
                        if Epasp != 0:
                            pasp = Epasp
                        else:
                            mPAP = random.randint(25, 50)
                            if mPAP < 35:
                                pasp=mPAP+15
                            elif mPAP > 35:
                                pasp=50+(mPAP-45)*2
                    if pasp != 0:
                        pasp = Epasp
                        if EmPAP != 0:
                            mPAP = EmPAP
                        else:
                            if pasp < 50:
                                mPAP=pasp-15
                            elif pasp > 50:
                                mPAP=35+(pasp-50)/2
                    else:
                        continue
                
                if mPAP == 0:
                    if pasp != 0:
                        if pasp < 50:
                            mPAP=pasp-15
                        elif pasp > 50:
                            mPAP=35+(pasp-50)/2
                elif pasp == 0:
                    if mPAP != 0:
                        if mPAP < 35:
                            pasp=mPAP+15
                        elif mPAP > 35:
                            pasp=50+(mPAP-45)*2

                if ASD:
                    a_cases += 1
                else:
                    b_cases += 1
                
                # new_path = '/home/jyangcu/Dataset/CardiacNet/CardiacNet-ASD/Non-ASD/'
                # if ASD == 0:
                    # c_cases += 1
                    # if '4' in view_path.keys():
                    #     if view_path['4'] is not None:
                    #         path_split = view_path['4'].split('/')
                    #         path_split[4] = 'dataset_pa_iltrasound_nill_files_clean'
                    #         name = str(c_cases).zfill(3) + '_image.nii.gz'
                    #         target_path = new_path + name
                    #         maskname = path_split[-1].replace('image', 'label')
                    #         maskpath = '/'.join(path_split[:-1]) + '/' + maskname
                    #     original_path = '/'.join(path_split)
                    #     shutil.copy(original_path, target_path)
                    #     if os.path.exists(maskpath):
                    #         mask_target_path = new_path + str(c_cases).zfill(3) + '_label.nii.gz'
                    #         shutil.copy(maskpath, mask_target_path)

                    for k in self.view_num:
                        if k in view_path.keys():
                            if view_path[k] is None:
                                image_path = ''
                            else:
                                image_fold, image_name = view_path[k].split('/')[-2:]
                                image_name = image_name[:-7]
                                image_path = self.root + '/' + image_fold + '/' + image_name + '/image/'
                    
                    image_dict = dict()
                    video_dict = dict()
                    if os.path.exists(image_path):
                        for _, _, images in os.walk(image_path):
                            video_dict[image_path] = {'images':images, 'mPAP':mPAP, 'pasp':pasp, 'ASD': ASD}
                            for i in images:
                                image_dict[image_path+i]={'mPAP':mPAP, 'pasp':pasp}
                        if 0 < mPAP <= 25:
                            all_images['normal'].update(image_dict)
                            all_videos['normal'].update(video_dict)
                        elif 0 < pasp <= 40:
                            all_images['normal'].update(image_dict)
                            all_videos['normal'].update(video_dict)
                        elif 25 < mPAP <= 35:
                            all_images['slight'].update(image_dict)
                            all_videos['slight'].update(video_dict)
                        elif 40 < pasp <= 50:
                            all_images['slight'].update(image_dict)
                            all_videos['slight'].update(video_dict)
                        elif 35 < mPAP <= 45:
                            all_images['middle'].update(image_dict)
                            all_videos['middle'].update(video_dict)
                        elif 35 < pasp <= 70:
                            all_images['middle'].update(image_dict)
                            all_videos['middle'].update(video_dict)
                        else:
                            all_images['severe'].update(image_dict)
                            all_videos['severe'].update(video_dict)
                        
                        if mPAP > 25:
                            all_images['pah'].update(image_dict)
                            all_videos['pah'].update(video_dict)
                        
                        if ASD is not None and ASD == 1:
                            all_images['ASD'].update(image_dict)
                            all_videos['ASD'].update(video_dict)
                            if mPAP >= 45:
                                all_images['ASD-severe'].update(image_dict)
                                all_videos['ASD-severe'].update(video_dict)
                            elif pasp >= 70:
                                all_images['ASD-severe'].update(image_dict)
                                all_videos['ASD-severe'].update(video_dict)
                        elif ASD is not None and ASD == 0:
                            all_images['None_ASD'].update(image_dict)
                            all_videos['None_ASD'].update(video_dict)
        
        print("a_cases", a_cases)
        print("b_cases", b_cases)
        print("c_cases", c_cases)
        print("d_cases", d_cases)
        print("data_num", data_num)
        image_list = {'all':{}}
        video_list = {'all':{}}
        for selected_type in data_type:
            image_list['all'].update(all_images[selected_type])
            video_list['all'].update(all_videos[selected_type])
        print("frame_num", len(list(image_list['all'].keys())))
        return image_list, video_list

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--image-size', type=int, default=(112,112,16), help='Image height and width (default: 256)')
    parser.add_argument('--mask-size', type=int, default=8, help='The size of mask patch (default: 16)')
    parser.add_argument('--mask-ratio', type=float, default=0.6, help='The ratio of masking area in an image (default: 0.75)')
    args = parser.parse_args()
    args.dataset_path = r'/home/jyangcu/Dataset/dataset_pa_iltrasound_nill_files_clean_image'

    data_dict = dict()
    infos = np.load(f'/home/jyangcu/Dataset/dataset_pa_iltrasound_nii_files_3rdcenters/save_infos_reg_v3.npy', allow_pickle=True).item()
    
    from monai.data import DataLoader
    from torchvision import utils as vutils
    train_ds = CardiacNet(args, infos, set_select=['rmyy','gy','shph','szfw'], view_num=['4'])
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=1)
    from einops import rearrange
    for img, mask, mpap, pasp in tqdm(train_loader):
        print(img.shape)
        print(mask.shape)
        # masked_pos = rearrange(mask, 'b (l h w) p1 p2 -> b l (h p1) (w p2)', 
        #                               h=args.image_size[0]//args.mask_size, w=args.image_size[1]//args.mask_size, 
        #                               l=args.image_size[2], p1=args.mask_size, p2=args.mask_size).unsqueeze(1)
        # print(masked_pos.shape)
        # imgs = img.add(1.0).mul(0.5) * masked_pos.float()
        # vutils.save_image(imgs[:,:,4], "data_sample.jpg", nrow=4)