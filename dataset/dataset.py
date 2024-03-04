import pickle
from monai.transforms import (
    Compose,
    RandCropByPosNegLabeld,
    CropForegroundd,
    SpatialPadd,
    ScaleIntensityRanged,
    RandShiftIntensityd,
    RandFlipd,
    RandZoomd,
    RandRotated,
    RandRotate90d,
    RandGaussianNoised,
    RandGaussianSmoothd,
    NormalizeIntensityd,
    MapTransform,
    RandScaleIntensityd,
    RandSpatialCropd,
    CenterSpatialCropd,
    Spacingd,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    RandAdjustContrastd
)

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
import os
import pandas as pd 

    
class BaseDataset(Dataset):
    def __init__(
            self,
            args,
            image_paths,
            transforms = None
    ):
        super().__init__()
        self.img_dict = pd.read_csv(image_paths)
        self.img_dict['label']
        #self.img_dict.loc[self.img_dict['label']==5,'label']=4
        self.root = args.root
        #self._set_dataset_stat()
        self.transforms = transforms#self.get_transforms()

        class_counts = self.img_dict['label'].value_counts().sort_index().values
        class_weights = 1.0 / class_counts
        self.sampler_weight = class_weights[self.img_dict['label'].values]

    # def _set_dataset_stat(self):
    #     self.spacing = (1.0, 1.0, 1.0)
    #     self.spatial_index = [2, 1, 0]  # index used to convert to DHW
    #     self.target_class = 1

    def __len__(self):
        return len(self.img_dict)

    def cal_weight(self):
        class_counts = self.img_dict['label'].value_counts().sort_index().values
        return class_counts
        
    def read(self, path):
        itk_image = sitk.ReadImage(self.root + path)
        image = sitk.GetArrayFromImage(itk_image).transpose(1,2,0)
        return image

    def __getitem__(self, idx):
        path = self.img_dict.iloc[idx]
        t2w = self.read(path['t2w'])
        dwi = self.read(path['dwi'])
        adc = self.read(path['adc'])
        
        label = torch.tensor(path['label'], dtype=torch.long)
        # print(z_min, z_max)
        img = np.stack([t2w, dwi, adc], 0)
        if self.transforms is not None:    
            trans_dict = self.transforms({"image": img})
            if type(trans_dict) == list:
                trans_dict = trans_dict[0]
            img = trans_dict["image"]
        return img, label, torch.tensor(idx, dtype=torch.long) 


def get_train_transforms(args):
    train_transforms = [
        NormalizeIntensityd(
            keys="image", 
            nonzero=True, 
            channel_wise=True
        ),
        RandRotated(
            keys=["image"],
            prob=0.3,
            range_x=10 / 180 * np.pi,
            range_y=10 / 180 * np.pi,
            range_z=10 / 180 * np.pi,
            keep_size=False,
        ),
        RandZoomd(
            keys=["image"],
            prob=0.3,
            min_zoom=[0.9, 0.9, 0.9],
            max_zoom=[1.1, 1.1, 1.1],
            mode=["trilinear"],
        ),
        RandSpatialCropd(
            keys=["image"],
            roi_size = args.crop_spatial_size
        ),
        # RandAffined(
        #     keys=["image"],
        #     mode=("bilinear"),
        #     prob=0.15,
        #     spatial_size=args.crop_spatial_size,
        #     translate_range=(0, 10, 10),
        #     rotate_range=(np.pi / 12, 0, 0),
        #     scale_range=(0, 0.1, 0.1),
        #     shear_range=(0, 0.2, 0.2),
        #     padding_mode='zeros'
        # ),
        # Rand3DElasticd(
        #     keys=["image"],
        #     prob=0.15,  
        #     sigma_range=(5,7),
        #     magnitude_range=(10, 20),  
        #     rotate_range=(0, 0, 0),  
        #     scale_range=(0.1, 0.1, 0.1),  
        #     mode='bilinear',  
        #     padding_mode='border'  
        # ),
        # RandGaussianNoised(
        #     keys=["image"],
        #     prob=0.15,
        #     mean=0,
        #     std=0.07
        # ),
        RandFlipd(
            keys="image",
            prob=0.5, 
            spatial_axis=2
        ),
        SpatialPadd(
            keys="image", 
            spatial_size = args.crop_spatial_size
        )
    ]
    train_transforms = Compose(train_transforms)
    return train_transforms


def get_test_transforms(args):
    test_transforms = [ 
        NormalizeIntensityd(
            keys="image", 
            nonzero=True, 
            channel_wise=True
        ),
        SpatialPadd(
            keys="image", 
            spatial_size = args.crop_spatial_size
        )
    ]
    test_transforms = Compose(test_transforms)
    return test_transforms


def build_loader(args):
    train_csv = f'split_group/split_group/train_{args.split}.csv'
    test_csv = f'split_group/split_group/test_{args.split}.csv'

    train_set = BaseDataset(args, train_csv, get_train_transforms(args))
    
    args.cls_account = train_set.cal_weight() / len(train_set)


    train_set, val_set = torch.utils.data.random_split(train_set, [0.9, 0.1])
    args.memeory_size = len(train_set)
    val_set.transforms = get_test_transforms(args)
    test_set = BaseDataset(args, test_csv, get_test_transforms(args)) 
    print(f'build dataset')
    print(f'train size: {len(train_set)},  val size: {len(val_set)},  test size: {len(test_set)}')

    #if args.loss == 'FocalLoss':
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=False, pin_memory=True)
    # else:
    #     sampler = WeightedRandomSampler(weights=train_set.cal_weight(), num_samples=len(train_set), replacement=True)
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
    #                             num_workers=args.num_workers, drop_last=False, pin_memory=True)

    if args.weightsampler:
        sampler_weight = [train_set.dataset.sampler_weight[i] for i in train_set.indices]
        sampler = WeightedRandomSampler(weights=sampler_weight, num_samples=len(train_set), replacement=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                                 num_workers=args.num_workers, drop_last=False, pin_memory=True)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=False, pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=args.batch_size, 
                              num_workers=args.num_workers, drop_last=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, 
                              num_workers=args.num_workers, drop_last=False, pin_memory=True)
    return train_loader, val_loader, test_loader
