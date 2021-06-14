from os.path import splitext
from os import listdir
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import torchvision.transforms.functional as transF
from imgaug import augmenters as iaa
import scipy.io as io
from utils.offset_helper import DTOffsetHelper
mean_std_dict={'WHU_BUILDING':[0.3,[0.43782742, 0.44557303, 0.41160695],[0.19686149, 0.18481555, 0.19296625],'.tif'],\
                'Inriaall':[0.2,[0.31815762,0.32456695,0.29096074],[0.18410079,0.17732723,0.18069517],'.png'],
               'Mass':[0.9,[0.31815762,0.32456695,0.29096074],[0.18410079,0.17732723,0.18069517],'.png']
               }

class SegfixDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir,training=False,get_edge=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = 1
        self.training=training
        self.get_edge=get_edge
        self.res,self.mean,self.std,self.shuffix=mean_std_dict[imgs_dir.split('/')[-4]]
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        img = Image.open(self.imgs_dir +self.ids[0] + self.shuffix)
        self.transform = iaa.Sequential([
            iaa.Rot90([0,1,2,3]),
            iaa.VerticalFlip(p=0.5),
            iaa.HorizontalFlip(p=0.5),

        ])
    def __len__(self):
        return len(self.ids)
    def _load_mat(self, filename):
        return io.loadmat(filename)
    def _load_maps(self, filename,):
        dct = self._load_mat(filename)
        distance_map = dct['depth'].astype(np.int32)
        dir_deg = dct['dir_deg'].astype(np.float)  # in [0, 360 / deg_reduce]
        deg_reduce = dct['deg_reduce'][0][0]

        dir_deg = deg_reduce * dir_deg - 180  # in [-180, 180]
        return distance_map, dir_deg
    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = self.masks_dir +idx + self.shuffix
        img_file = self.imgs_dir +idx + self.shuffix
        mask = Image.open(mask_file)
        img = Image.open(img_file)
        distance_map, angle_map = self._load_maps(self.imgs_dir.replace('image','dt_offset') + idx )
        distance_map = np.array(distance_map)
        angle_map = np.array(angle_map)
        _, direction_map = DTOffsetHelper.align_angle(torch.tensor(angle_map), num_classes=8,
                                                 return_tensor=True)
        if self.training:
            img,mask=self.transform(image=img,segmentation_maps=np.stack((mask[np.newaxis,:,:],direction_map[np.newaxis,:,:],distance_map[np.newaxis,:,:]),axis=-1).astype(np.int32))
            mask,direction_map,distance_map=mask[0,:,:,0],mask[0,:,:,1],mask[0,:,:,2]
        img,mask=transF.to_tensor(img.copy()),(transF.to_tensor(mask.copy())>0).int()
        img=transF.normalize(img,self.mean,self.std)
        return {
            'image': img.float(),
            'mask': mask.float(),
            'direction_map':direction_map,
            'distance_map':distance_map,
            'name': self.imgs_dir + idx + self.shuffix
        }

