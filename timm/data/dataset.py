""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2019, Ross Wightman
"""
import io
import logging
from typing import Optional

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage
import sys
import os
from threading import Thread

from .readers import create_reader
from aug import denormalize_img

_logger = logging.getLogger(__name__)

import cv2
from random import randint, random
import random


_ERROR_RETRY = 50


class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            load_bytes=False,
            input_img_mode='RGB',
            transform=None,
            target_transform=None,
            cover_prob=0.0,
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or '',
                root=root,
                split=split,
                class_map=class_map
            )
        self.reader = reader
        self.load_bytes = load_bytes
        self.input_img_mode = input_img_mode
        self.transform = transform
        self.cover_prob = cover_prob
        self.target_transform = target_transform
        self._consecutive_errors = 0
    
    def covering(self, org_img):

        org_img = np.stack((org_img[0], org_img[1], org_img[2]), axis=2)

        width, height = org_img.shape[0], org_img.shape[1]
        q_width1 = int(width / 4)
        q_height1 = int(height / 4)
        q_width2 = int(width * 3 / 4)
        q_height2 = int(height * 3 / 4)
        limit = min(width, height)

        blue_color = (255, 0, 0)
        green_color = (0, 255, 0)
        red_color = (0, 0, 255)
        white_color = (255, 255, 255)
        cover_color = (randint(0, 255), randint(0, 255), randint(0, 255))

        blank_img = np.zeros((width, height, 3), np.uint8)
        center_area = np.array([[q_width1, q_height1], [q_width1, q_height2], [q_width2, q_height2], [q_width2, q_height1]], np.int32)
        only_center = cv2.fillConvexPoly(blank_img, center_area, white_color)

        while True:
            blank_img = np.zeros((width, height, 3), np.uint8)
            cover1 = np.array([[randint(0, limit), randint(0, limit)], [randint(0, limit), randint(0, limit)], [randint(0, limit), randint(0, limit)]], np.int32)
            plus_cover1 = cv2.fillConvexPoly(blank_img, cover1, white_color)
            cover2 = np.array([[randint(0, limit), randint(0, limit)], [randint(0, limit), randint(0, limit)], [randint(0, limit), randint(0, limit)]], np.int32)
            plus_cover2 = cv2.fillConvexPoly(plus_cover1, cover2, white_color)

            cover_pixel_num = np.count_nonzero(plus_cover2) / 3
            cover_ratio = cover_pixel_num / (width * height)
            if cover_ratio < 0.03:      # prevent too small cover
                continue

            center_cover1 = np.logical_and(plus_cover2, only_center).astype(dtype=np.uint8) * 255
            covered_pixel_num = np.count_nonzero(center_cover1) / 3
            covered_ratio = covered_pixel_num / (width * height / 4)
            if covered_ratio < 0.05 or covered_ratio > 0.25:    # prevent useless cover & too large cover
                continue
            threshold = abs(covered_ratio - 0.15) / 0.1
            rand_num = random.random()
            if rand_num > threshold:        # control probability distribution
                continue
            break

        img = cv2.fillConvexPoly(org_img, cover1, cover_color)
        img = cv2.fillConvexPoly(img, cover2, cover_color)

        return torch.Tensor(np.stack((img[:,:,0], org_img[:,:,1], org_img[:,:,2]), axis=0).astype(np.float32))

    def __getitem__(self, index):
        img, target = self.reader[index]

        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.reader))
            else:
                raise e
        self._consecutive_errors = 0

        if self.input_img_mode and not self.load_bytes:
            img = img.convert(self.input_img_mode)
        if self.transform is not None:
            img = self.transform(image=np.array(img))["image"]   # img = self.transform(img)
        if self.cover_prob > 0.0:
            rand_num = random.random()
            if rand_num < self.cover_prob:
                img = self.covering(org_img=np.array(img))
        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class CustomImageDataset(data.Dataset):

    def load_img(self, img_name_lst, data_lst, label_lst, class_dir, class_idx, class_name):
        total_num = len(img_name_lst)
        percent = 0.0
        for img_idx, img_name in enumerate(img_name_lst):
            img_path = os.path.join(class_dir, img_name)
            image = Image.open(img_path).convert(self.input_img_mode).resize((224, 224))
            data_lst.append(np.array(image, dtype=np.uint8))
            label_lst.append(class_idx)
            if percent <= img_idx / total_num:
                print(class_name, 'loading is completed', percent * 100, '%!')
                percent += 0.05

    def __init__(
            self,
            root,
            split='train',
            input_img_mode='RGB',
            transform=None
            ):
        self.process_num = 10
        self.data = [list() for _ in range(self.process_num)]
        self.labels = [list() for _ in range(self.process_num)]
        self.root_dir = root
        self.transform = transform
        self.input_img_mode = input_img_mode
        self.total_data_size = 0
        
        
        for class_idx, class_name in enumerate(os.listdir(self.root_dir)):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                img_name_lst = os.listdir(class_dir)
                total_num = len(img_name_lst)
                processes = [Thread(target=self.load_img, args=(img_name_lst[total_num // self.process_num * i:total_num // self.process_num * (i + 1)], self.data[i], self.labels[i], class_dir, class_idx, class_name)) for i in range(self.process_num)]
                for process in processes:
                    process.start()

                for process in processes:
                    process.join()
        
        for a_data in self.data:
            self.total_data_size += len(a_data)
        print('total dataset size is', self.total_data_size, '!!!')
    
    def __len__(self):
        return self.total_data_size
    
    def __getitem__(self, idx):
        for i in range(self.process_num):
            if idx < self.total_data_size // self.process_num * (i+1):
                image = self.data[i][idx - self.total_data_size // self.process_num * i]
                label = self.labels[i][idx - self.total_data_size // self.process_num * i]
                break
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            is_training=False,
            batch_size=1,
            num_samples=None,
            seed=42,
            repeats=0,
            download=False,
            input_img_mode='RGB',
            input_key=None,
            target_key=None,
            transform=None,
            target_transform=None,
            max_steps=None,
    ):
        assert reader is not None
        if isinstance(reader, str):
            self.reader = create_reader(
                reader,
                root=root,
                split=split,
                class_map=class_map,
                is_training=is_training,
                batch_size=batch_size,
                num_samples=num_samples,
                seed=seed,
                repeats=repeats,
                download=download,
                input_img_mode=input_img_mode,
                input_key=input_key,
                target_key=target_key,
                max_steps=max_steps,
            )
        else:
            self.reader = reader
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.reader:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def __len__(self):
        if hasattr(self.reader, '__len__'):
            return len(self.reader)
        else:
            return 0

    def set_epoch(self, count):
        # TFDS and WDS need external epoch count for deterministic cross process shuffle
        if hasattr(self.reader, 'set_epoch'):
            self.reader.set_epoch(count)

    def set_loader_cfg(
            self,
            num_workers: Optional[int] = None,
    ):
        # TFDS and WDS readers need # workers for correct # samples estimate before loader processes created
        if hasattr(self.reader, 'set_loader_cfg'):
            self.reader.set_loader_cfg(num_workers=num_workers)

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)
