import cv2
import os
import argparse

import numpy as np
import torch
from tqdm import tqdm

from config import Config
from torch.utils.data import DataLoader
from dataset import TestImageDataset
from model import Generator
from utils import load_state_dict, bgr2ycbcr, tensor2img, PSNR, SSIM
from bicubic import Bicubic, NearestNeighbourUpscale

from statistics import NormalDist

def confidence_interval(data, confidence=0.95):
  """ 
  Calculates the confidence interval
  Stolen from: https://stackoverflow.com/a/55270860/19877091
  """
  dist = NormalDist.from_samples(data)
  z = NormalDist().inv_cdf((1 + confidence) / 2.)
  h = dist.stdev * z / ((len(data) - 1) ** .5)
  return h # dist.mean - h, dist.mean + h

def test(config: Config, save_images: bool = True, g_path: str = None, concat_w_gt: bool = False):
    """
    Test a generator, if no path to generator is given the generator at current exp.name is used.
    """
    if not g_path:
        g_path = f"results/{config.EXP.NAME}/g_best.pth"
    
    test_datasets = TestImageDataset(config.DATA.TEST_GT_IMAGES_DIR, config.DATA.TEST_LR_IMAGES_DIR)

    test_dataloader = DataLoader(
        dataset = test_datasets,
        batch_size = 1,
        shuffle = False,
        num_workers = 1,
        pin_memory = True,
        drop_last = False,
        persistent_workers = True,
    )

    # Initialize generator and load weights
    if config.EXP.NAME == "bicubic":
        generator = Bicubic(device=config.DEVICE).to(config.DEVICE)
    elif config.EXP.NAME == "nearest":
        generator = NearestNeighbourUpscale(config.DATA.UPSCALE_FACTOR).to(config.DEVICE)
    else:
        generator = Generator(config).to(config.DEVICE)
        generator = load_state_dict(generator, torch.load(g_path, map_location=config.DEVICE))
        generator.eval()

    # Test
    _psnr, _ssim = _validate(generator=generator, val_loader=test_dataloader, config=config, save_images=save_images, concat_with_gt=concat_w_gt, save_metrics=True)


def _validate(generator, val_loader: DataLoader, config: Config, save_images:bool = False, concat_with_gt:bool = False, save_metrics:bool = False) -> tuple[float, float]:
    """ Run testing on a generator(or bicubic etc.) with a given dataset """

    if save_metrics:
        path = os.path.join(config.DATA.TEST_SR_IMAGES_DIR, config.EXP.NAME)
        os.makedirs(path, exist_ok=True)
        file = open(file = os.path.join(config.DATA.TEST_SR_IMAGES_DIR, config.EXP.NAME, "_metrics.txt"), mode='w')

    with torch.no_grad():
        all_psnr = []
        all_ssim = []

        for idx, (hr_img, lr_img) in enumerate(tqdm(val_loader)):
            lr_img = lr_img.to(config.DEVICE)
            hr_img = hr_img.to(config.DEVICE)

            output = generator(lr_img)

            output = tensor2img(output)
            gt = tensor2img(hr_img)

            # Save SR images
            if save_images:
                path = os.path.join(config.DATA.TEST_SR_IMAGES_DIR, config.EXP.NAME)
                os.makedirs(path, exist_ok=True)
                if concat_with_gt:
                    cv2.imwrite(f"{path}/{idx}.png", np.concatenate([output, gt], axis=1))
                else:
                    cv2.imwrite(f"{path}/{idx}.png", output)

            output = output.astype(np.float32) / 255.0
            gt = gt.astype(np.float32) / 255.0

            output = bgr2ycbcr(output, only_y=True)
            gt = bgr2ycbcr(gt, only_y=True)
            psnr = PSNR(output * 255, gt * 255)
            ssim = SSIM(output * 255, gt * 255)
            all_psnr.append(psnr)
            all_ssim.append(ssim)

            if save_metrics:
                file.write(f"{idx}.png | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}\n")

        avg_psnr = sum(all_psnr) / len(all_psnr)
        avg_ssim = sum(all_ssim) / len(all_ssim)

    output = f"[Test] | PSNR: {avg_psnr:.2f} ± {confidence_interval(all_psnr):.2f} | SSIM: {avg_ssim:.4f} ± {confidence_interval(all_ssim):.4f} | \n"
    print(output)

    if save_metrics:
        file.write("\n" + output + "\n")

    return avg_psnr, avg_ssim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
            Run evaluation on a model. Note if config.EXP.NAME is 'bicubic' or 'nearest' a bicubic 
            or nearest neighbour model will be run instead of a generator
        """
    )
    parser.add_argument("--save_images", type=bool, default=True, help="Should the SR images be saved")
    parser.add_argument("--concat_w_gt", type=bool, default=False, help="Should the GT images be saved alongside the SR images")
    parser.add_argument("--gpath", type=str, default=None, help="If the model being evaluated is not from a experiment i.e. the weights are not in the /results/ folder, the absolute path to the weights can be given here")
    args = parser.parse_args()
    
    config = Config()

    # Set the model to test - model should be in /results/ folder, else use gpath parameter for test func
    config.EXP.NAME = "patchwise-st-double-content-gpua-queue"

    # Set the dataset to test on
    config.DATA.TEST_SET = "Urban100"
    config.DATA.TEST_GT_IMAGES_DIR = F"/work3/{config.EXP.USER}/data/{config.DATA.TEST_SET}/GTmod12"
    config.DATA.TEST_LR_IMAGES_DIR = f"/work3/{config.EXP.USER}/data/{config.DATA.TEST_SET}/LRbicx4"

    test(config = config, save_images = args.save_images, concat_w_gt = args.concat_w_gt, g_path=args.gpath)