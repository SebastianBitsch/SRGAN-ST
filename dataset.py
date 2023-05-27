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
import cv2
import imgproc

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.io import read_image


class TrainImageDataset(Dataset):
    """
    A wrapper class around the torch Dataset class.
    Used for getting GT and LR images for training
    """

    def __init__(self, gt_image_dir: str, upscale_factor: int) -> None:
        super(TrainImageDataset, self).__init__()
        self.image_file_names = [
            os.path.join(gt_image_dir, image_file_name) for image_file_name in absoluteFilePaths(gt_image_dir)
        ]
        self.upscale_factor = upscale_factor

    def __getitem__(self, batch_index: int) -> tuple[Tensor, Tensor]:
        # Read a batch of image data
        im_path = self.image_file_names[batch_index]

        gt_tensor = read_image(im_path).float().unsqueeze(0) / 255.0
        lr_tensor = F.interpolate(gt_tensor, scale_factor = 1.0 / self.upscale_factor, mode='bicubic', antialias=True)
        gt_tensor = gt_tensor.squeeze()
        lr_tensor = lr_tensor.squeeze()

        # Read GT image and generate LR
        # gt_image = cv2.imread(im_path).astype(np.float32) / 255.
        # lr_image = imgproc.image_resize(gt_image, 1 / self.upscale_factor)

        # # BGR convert RGB
        # gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        # lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # # Convert image data into Tensor stream format (PyTorch).
        # # Note: The range of input and output is between [0, 1]
        # gt_tensor = imgproc.image_to_tensor(gt_image, False, False)
        # lr_tensor = imgproc.image_to_tensor(lr_image, False, False)
        # print("gt, shaep", gt_tensor.shape, lr_tensor.shape)
        return gt_tensor, lr_tensor

    def __len__(self) -> int:
        return len(self.image_file_names)



class TestImageDataset(Dataset):
    """
    A wrapper class around the torch Dataset class.
    Used for getting GT and LR images for training
    """

    def __init__(self, test_gt_images_dir: str, test_lr_images_dir: str) -> None:
        super(TestImageDataset, self).__init__()
        # Get all image file names in folder
        self.gt_image_file_names = [os.path.join(test_gt_images_dir, x) for x in absoluteFilePaths(test_gt_images_dir) if not x.startswith('.')]
        self.lr_image_file_names = [os.path.join(test_lr_images_dir, x) for x in absoluteFilePaths(test_lr_images_dir) if not x.startswith('.')]

    def __getitem__(self, batch_index: int) -> tuple[Tensor, Tensor]:
        # Read a batch of image data
        # gt_image = cv2.imread(self.gt_image_file_names[batch_index]).astype(np.float32) / 255.
        # lr_image = cv2.imread(self.lr_image_file_names[batch_index]).astype(np.float32) / 255.

        # # BGR convert RGB
        # gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        # lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # # Convert image data into Tensor stream format (PyTorch).
        # # Note: The range of input and output is between [0, 1]
        # gt_tensor = imgproc.image_to_tensor(gt_image, False, False)
        # lr_tensor = imgproc.image_to_tensor(lr_image, False, False)

        gt_tensor = read_image(self.gt_image_file_names[batch_index]).float() / 255.0
        lr_tensor = read_image(self.lr_image_file_names[batch_index]).float() / 255.0

        return gt_tensor, lr_tensor

    def __len__(self) -> int:
        return len(self.gt_image_file_names)


def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))
