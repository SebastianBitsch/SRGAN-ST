import cv2
import os
import math

import numpy as np
from torchvision.utils import make_grid, save_image
import torch

from config import Config
from torch.utils.data import DataLoader
from dataset import TestImageDataset
from model import Generator

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    # 4D: grid (B, C, H, W), 3D: (C, H, W), 2D: (H, W)
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])

    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), padding=0, normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()



def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def test(config: Config, save_images: bool = True, g_path: str = None):
    """
    Test a generator, if not path to generator is given the generator at current exp-name is used.
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

    # Initialize the super-resolution bsrgan_model
    generator = Generator(config).to(config.DEVICE)
    generator.load_state_dict(torch.load(g_path))

    # Test
    psnr, ssim = _validate(generator=generator, val_loader=test_dataloader, config=config, save_images=save_images)
    print(f"[Test] [PSNR: {psnr}] [SSIM: {ssim}]")


def _validate(generator: Generator, val_loader: DataLoader, config: Config, save_images:bool = False) -> tuple[float, float]:
    """ Run testing on a generator with a given dataset """
    with torch.no_grad():
        psnr_l = []
        ssim_l = []

        for idx, (hr_img, lr_img) in enumerate(val_loader):
            lr_img = lr_img.to(config.DEVICE)
            hr_img = hr_img.to(config.DEVICE)

            output = generator(lr_img)

            output = tensor2img(output)
            gt = tensor2img(hr_img)

            # Save SR images
            if save_images:
                path = os.path.join(config.DATA.TEST_SR_IMAGES_DIR, config.EXP.NAME)
                os.makedirs(path, exist_ok=True)
                cv2.imwrite(f"{path}/{idx}.png", np.concatenate([output, gt], axis=1))

            output = output.astype(np.float32) / 255.0
            gt = gt.astype(np.float32) / 255.0

            output = bgr2ycbcr(output, only_y=True)
            gt = bgr2ycbcr(gt, only_y=True)
            psnr = calculate_psnr(output * 255, gt * 255)
            ssim = calculate_ssim(output * 255, gt * 255)
            psnr_l.append(psnr)
            ssim_l.append(ssim)

        avg_psnr = sum(psnr_l) / len(psnr_l)
        avg_ssim = sum(ssim_l) / len(ssim_l)

    return avg_psnr, avg_ssim


if __name__ == "__main__":
    config = Config()
    config.EXP.NAME = "stock-srgan"
    # gpath = ""

    test(config = config, save_images = True)