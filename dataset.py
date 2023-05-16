# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import queue
import threading

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import imgproc

__all__ = [
    "TrainValidImageDataset", "TestImageDataset"
]


class TrainValidImageDataset(Dataset):
    """Define training/valid dataset loading methods.

    Args:
        gt_image_dir (str): Train/Valid dataset address.
        gt_image_size (int): Ground-truth resolution image size.
        upscale_factor (int): Image up scale factor.
        mode (str): Data set loading method, the training data set is for data enhancement, and the
            verification dataset is not for data enhancement.
    """

    def __init__(self, gt_image_dir: str, gt_image_size: int, upscale_factor: int, mode: str) -> None:
        super(TrainValidImageDataset, self).__init__()
        self.image_file_names = [os.path.join(gt_image_dir, image_file_name) for image_file_name in
                                 absoluteFilePaths(gt_image_dir)]
        
        self.gt_image_size = gt_image_size
        self.upscale_factor = upscale_factor
        self.mode = mode

    def __getitem__(self, batch_index: int) -> dict[str, Tensor]:
        # Read a batch of image data
        im_path = self.image_file_names[batch_index]
        gt_image = cv2.imread(im_path).astype(np.float32) / 255.
        
        # Get the original image. i.e. go from /work3/s204163/data/ImageNet/train/0150_0043.png -> im_59.bmp
        org_im_path = im_path.replace("/train/", "/cropped/")
        org_im_name = "/".join(org_im_path.split("_")[:-1])
        
        org_im_name = f"{org_im_name}.png"
        org_image = cv2.imread(org_im_name).astype(np.float32) / 255.
        
        # Image processing operations
        if self.mode == "Train":
            gt_crop_image = imgproc.random_crop(gt_image, self.gt_image_size)
        elif self.mode == "Valid":
            gt_crop_image = imgproc.center_crop(gt_image, self.gt_image_size)
        else:
            raise ValueError("Unsupported data processing model, please use `Train` or `Valid`.")

        lr_crop_image = imgproc.image_resize(gt_crop_image, 1 / self.upscale_factor)

        # BGR convert RGB
        gt_crop_image = cv2.cvtColor(gt_crop_image, cv2.COLOR_BGR2RGB)
        lr_crop_image = cv2.cvtColor(lr_crop_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        gt_crop_tensor = imgproc.image_to_tensor(gt_crop_image, False, False)
        lr_crop_tensor = imgproc.image_to_tensor(lr_crop_image, False, False)
        org_image_tensor = imgproc.image_to_tensor(org_image, False, False)

        # return {"gt": gt_crop_tensor, "lr": lr_crop_tensor, "original": org_image_tensor}
        return [gt_crop_tensor, lr_crop_tensor, org_image_tensor]#, "original": org_image}

    def __len__(self) -> int:
        return len(self.image_file_names)

def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

class TestImageDataset(Dataset):
    """Define Test dataset loading methods.

    Args:
        test_gt_images_dir (str): ground truth image in test image
        test_lr_images_dir (str): low-resolution image in test image
    """

    def __init__(self, test_gt_images_dir: str, test_lr_images_dir: str) -> None:
        super(TestImageDataset, self).__init__()
        # Get all image file names in folder
        self.gt_image_file_names = [os.path.join(test_gt_images_dir, x) for x in absoluteFilePaths(test_gt_images_dir) if not x.startswith('.')]
        self.lr_image_file_names = [os.path.join(test_lr_images_dir, x) for x in absoluteFilePaths(test_lr_images_dir) if not x.startswith('.')]

    def __getitem__(self, batch_index: int) -> dict[str, torch.Tensor]:
        # Read a batch of image data
        gt_image = cv2.imread(self.gt_image_file_names[batch_index]).astype(np.float32) / 255.
        lr_image = cv2.imread(self.lr_image_file_names[batch_index]).astype(np.float32) / 255.

        # BGR convert RGB
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        gt_tensor = imgproc.image_to_tensor(gt_image, False, False)
        lr_tensor = imgproc.image_to_tensor(lr_image, False, False)

        return {"gt": gt_tensor, "lr": lr_tensor}

    def __len__(self) -> int:
        return len(self.gt_image_file_names)

