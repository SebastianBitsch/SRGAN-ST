import cv2
import os
import sys
import inspect

from tqdm import tqdm
import numpy as np
import torch
from torchvision.io import read_image

# Weird hacky way of importing modules from parent directory. See: https://stackoverflow.com/a/11158224/19877091
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from config import Config
from model import Generator
from utils import tensor2img, load_state_dict
from bicubic import Bicubic, NearestNeighbourUpscale

def save_image_patch(config: Config, generator_names: list[str], image_name:str, xmin:int, xmax:int, ymin:int, ymax:int) -> None:
    """
    Save a given ground truth image with a red rectangle around a set of coordinates, along with
    the output of some generator(s) for that red rectangle.

    This is useful when making figures such as those seen in the GramGAN and BestBuddyGAN paper,
    where we want to compare how well different generators upsclaed a certain area of an image.

    Coordinate system is from top left corner.
    """

    # Make a directory in the test/_patch dir to store the images from every generator
    base_path = os.path.join(config.DATA.TEST_SR_IMAGES_DIR, "_patch", f"{image_name}.png")
    os.makedirs(base_path, exist_ok=True)

    # Read LR and GT images
    gt_path = f"{config.DATA.TEST_GT_IMAGES_DIR}/{image_name}.png"
    lr_path = f"{config.DATA.TEST_LR_IMAGES_DIR}/{image_name}.png"
    gt = tensor2img(read_image(gt_path).float() / 255.0)
    lr = read_image(lr_path).float().to(config.DEVICE).unsqueeze(0) / 255.0

    # Save the gt patch
    gt_patch = gt[xmin:xmax, ymin:ymax, :]
    cv2.imwrite(f"{base_path}/{image_name}_gt.png", gt_patch)

    # Draw red rect on gt image
    gt[xmin,ymin:ymax,:] = np.array([0,0,255])
    gt[xmax,ymin:ymax,:] = np.array([0,0,255])
    gt[xmin:xmax,ymin,:] = np.array([0,0,255])
    gt[xmin:xmax,ymax,:] = np.array([0,0,255])

    # Save the entire gt image with red rect
    cv2.imwrite(f"{base_path}/gt.png", gt)


    with torch.no_grad():
        for generator_name in tqdm(generator_names):

            # Initialize generator and load weights
            if generator_name == "bicubic":
                generator = Bicubic(device=config.DEVICE).to(config.DEVICE)
            elif generator_name == "nearest":
                generator = NearestNeighbourUpscale(config.DATA.UPSCALE_FACTOR).to(config.DEVICE)
            else:
                generator = Generator(config).to(config.DEVICE)
                weights = torch.load(f"results/{generator_name}/g_best.pth", map_location=config.DEVICE)
                generator = load_state_dict(generator, weights)

            sr = generator(lr)
            output = tensor2img(sr)

            # Save only the slize
            patch = output[xmin:xmax, ymin:ymax, :]
            cv2.imwrite(f"{base_path}/{image_name}_{generator_name}.png", patch)


if __name__ == "__main__":
    config = Config()
    config.DATA.TEST_SET = "Set14"
    config.DATA.TEST_GT_IMAGES_DIR = F"/work3/{config.EXP.USER}/data/{config.DATA.TEST_SET}/GTmod12"
    config.DATA.TEST_LR_IMAGES_DIR = f"/work3/{config.EXP.USER}/data/{config.DATA.TEST_SET}/LRbicx4"

    image_name = "baboon"
    models = ["resnet50", "bicubic", "nearest"]

    save_image_patch(config=config, generator_names=models, image_name=image_name, xmin=20, xmax = 100, ymin = 0, ymax = 200)
