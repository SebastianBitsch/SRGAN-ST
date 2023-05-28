import argparse
import multiprocessing
import os

import cv2
import numpy as np
from tqdm import tqdm

def main(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    # Get all image paths
    image_file_names = os.listdir(args.input_dir)

    # Split images on N threads
    progress_bar = tqdm(total=len(image_file_names), unit="image", desc="Prepare split images")
    workers_pool = multiprocessing.Pool(args.num_workers)

    for image_file_name in image_file_names:
        workers_pool.apply_async(worker, args=(image_file_name, args), callback = lambda _: progress_bar.update(1))

    workers_pool.close()
    workers_pool.join()
    progress_bar.close()


def worker(image_file_name:str, args:dict) -> None:
    """ Worker responsible for processing a single image into """

    image = cv2.imread(f"{args.input_dir}/{image_file_name}", cv2.IMREAD_UNCHANGED)
    im_h, im_w, _ = image.shape
    index = 1
    
    if args.output_size <= im_h and args.output_size <= im_w:
        for pos_y in range(0, im_h - args.output_size + 1, args.step_size):
            for pos_x in range(0, im_w - args.output_size + 1, args.step_size):
                # Crop
                crop_image = image[pos_y: pos_y + args.output_size, pos_x:pos_x + args.output_size, ...]
                crop_image = np.ascontiguousarray(crop_image)

                # TODO: Check if image should be saved, can be used in situtions where there is lots of
                # whitespace, as in the bone-images. For now save all images
                
                # Save image
                cv2.imwrite(f"{args.output_dir}/{image_file_name.split('.')[-2]}_{index:04d}.{image_file_name.split('.')[-1]}", crop_image)

                index += 1


if __name__ == "__main__":
    # config = Config()

    parser = argparse.ArgumentParser(
        description="""
            Slice a directory of images to sub-images of a given size. We use this to generate our HR images.
            This is used for slicing up the original images in i.e. DIV2K, to smaller images to help
            with not storing many large 2k images in memory.

            By default we slice the original images into HR images of size 192x192, which we then
            at runtime downscale by 4 to get 48x48 LR images.
        """
    )
    parser.add_argument("--input_dir", type=str, default="/work3/s204163/data/ImageNet/original")
    parser.add_argument("--output_dir", type=str, default="/work3/s204163/data/ImageNet/train")
    parser.add_argument("--output_size", type=int, default=96)
    parser.add_argument("--step_size", type=int, default=96)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    main(args)