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
import argparse
import multiprocessing
import os
import shutil
import random

import cv2
import numpy as np
from tqdm import tqdm


def main(args) -> None:
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Get all image paths and select n random ones to use
    all_image_file_names = os.listdir(args.images_dir)
    random.shuffle(all_image_file_names)
    image_file_names = all_image_file_names[:args.num_images]

    # Get the smallest image and crop the other ones to match
    min_w, min_h = get_min_image_dims(all_image_file_names)

    # Write the cropped images to a folder
    os.makedirs(args.cropped_dir, exist_ok=True)
    for im_name in tqdm(all_image_file_names, desc="Cropping images"):
        im = cv2.imread(f"{args.images_dir}/{im_name}", cv2.IMREAD_UNCHANGED)
        cropped_im = random_crop(im, min_w, min_h)
        cv2.imwrite(f"{args.cropped_dir}/{im_name}", cropped_im)


    # Splitting images with multiple threads
    progress_bar = tqdm(total=len(image_file_names), unit="image", desc="Prepare split image")
    workers_pool = multiprocessing.Pool(args.num_workers)
    for image_file_name in image_file_names:
        workers_pool.apply_async(worker, args=(image_file_name, args), callback=lambda arg: progress_bar.update(1))
    workers_pool.close()
    workers_pool.join()
    progress_bar.close()


def get_min_image_dims(all_image_file_names: list[str]) -> tuple:
    """ Return the size of the smallest original image """
    min_w = np.inf
    min_h = np.inf
    for im_name in tqdm(all_image_file_names, desc="Finding smallest image"):
        im = cv2.imread(f"{args.images_dir}/{im_name}", cv2.IMREAD_UNCHANGED)
        if im.shape[0] < min_h:
            min_h = im.shape[0]
        if im.shape[1] < min_w:
            min_w = im.shape[1]
    print(f"Smallest image found: ({min_w},{min_h})")
    return min_w, min_h


def worker(image_file_name, args) -> None:
    """ Parallel worker that grabs a slice of an image and saves it to folder"""
    image = cv2.imread(f"{args.cropped_dir}/{image_file_name}", cv2.IMREAD_UNCHANGED)

    image_height, image_width = image.shape[0:2]

    index = 1
    if image_height >= args.image_size and image_width >= args.image_size:
        for pos_y in range(0, image_height - args.image_size + 1, args.step):
            for pos_x in range(0, image_width - args.image_size + 1, args.step):
                # Crop
                crop_image = image[pos_y: pos_y + args.image_size, pos_x:pos_x + args.image_size, ...]
                crop_image = np.ascontiguousarray(crop_image)
                # Save image
                cv2.imwrite(f"{args.output_dir}/{image_file_name.split('.')[-2]}_{index:04d}.{image_file_name.split('.')[-1]}", crop_image)

                index += 1


def random_crop(image: np.ndarray, w:int, h:int) -> np.ndarray:
    """Crop small image patches from one image.

    Args:
        image (np.ndarray): The input image for `OpenCV.imread`.
        w (int): The size of the captured image width.
        h (int): The size of the captured image height.

    Returns:
        patch_image (np.ndarray): Small patch image

    """
    image_height, image_width = image.shape[:2]

    # Just need to find the top and left coordinates of the image
    top = random.randint(0, image_height - h)
    left = random.randint(0, image_width - w)

    # Crop image patch
    patch_image = image[top:top + h, left:left + w, ...]

    return patch_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare database scripts.")
    parser.add_argument("--images_dir", type=str, help="Path to input image directory.")
    parser.add_argument("--output_dir", type=str, help="Path to generate training image directory.")
    parser.add_argument("--cropped_dir", type=str, help="Path to save cropped images to.")
    parser.add_argument("--image_size", type=int, help="Low-resolution image size from raw image.")
    parser.add_argument("--step", type=int, help="Crop image similar to sliding window.")
    parser.add_argument("--num_workers", type=int, help="How many threads to open at the same time.")
    parser.add_argument("--num_images", type=int, help="How many of the original to slice up.")
    args = parser.parse_args()

    main(args)