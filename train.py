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
import time

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse

import model

from dataset import TrainImageDataset, TestImageDataset
from image_quality_assessment import PSNR, SSIM
from utils import load_state_dict, save_checkpoint, AverageMeter, ProgressMeter

import srgan_config

def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    best_ssim = 0.0

    train_dataloader, test_dataloader = build_dataloaders()
    print("Load all datasets successfully.")

    d_model, g_model = build_model()
    print(f"Successfully built generator.")

    # Initialize the loss functions by moving them to the cuda device
    define_loss()
    print("Define all loss functions successfully.")

    d_optimizer, g_optimizer = define_optimizer(d_model, g_model)
    print("Define all optimizer functions successfully.")

    d_scheduler, g_scheduler = define_scheduler(d_optimizer, g_optimizer)
    print("Define all optimizer scheduler functions successfully.")

    print("Check whether to load pretrained d model weights...")
    if srgan_config.pretrained_d_model_weights_path:
        d_model = load_state_dict(d_model, srgan_config.pretrained_d_model_weights_path)
        print(f"Loaded `{srgan_config.pretrained_d_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained d model weights not found.")

    print("Check whether to load pretrained g model weights...")
    if srgan_config.pretrained_g_model_weights_path:
        g_model = load_state_dict(g_model, srgan_config.pretrained_g_model_weights_path)
        print(f"Loaded `{srgan_config.pretrained_g_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained g model weights not found.")

    print("Check whether the pretrained d model is restored...")
    if srgan_config.resume_d_model_weights_path:
        d_model, _, start_epoch, best_psnr, best_ssim, optimizer, scheduler = load_state_dict(
            d_model,
            srgan_config.resume_d_model_weights_path,
            optimizer=d_optimizer,
            scheduler=d_scheduler,
            load_mode="resume")
        print("Loaded pretrained model weights.")
    else:
        print("Resume training d model not found. Start training from scratch.")

    print("Check whether the pretrained g model is restored...")
    if srgan_config.resume_g_model_weights_path:
        g_model, _, start_epoch, best_psnr, best_ssim, optimizer, scheduler = load_state_dict(
            g_model,
            srgan_config.resume_g_model_weights_path,
            optimizer=g_optimizer,
            scheduler=g_scheduler,
            load_mode="resume")
        print("Loaded pretrained model weights.")
    else:
        print("Resume training g model not found. Start training from scratch.")

    # Create a experiment results
    samples_dir = os.path.join("samples", srgan_config.exp_name)
    results_dir = os.path.join("results", srgan_config.exp_name)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", srgan_config.exp_name))

    # Create an IQA evaluation model
    psnr_model = PSNR(srgan_config.upscale_factor, srgan_config.only_test_y_channel)
    ssim_model = SSIM(srgan_config.upscale_factor, srgan_config.only_test_y_channel)

    # Transfer the IQA model to the specified device
    psnr_model = psnr_model.to(device=srgan_config.device)
    ssim_model = ssim_model.to(device=srgan_config.device)

    for epoch in range(start_epoch, srgan_config.epochs):
        train(d_model, g_model, train_dataloader, srgan_config.g_losses, d_optimizer, g_optimizer, epoch, writer)
        
        psnr, ssim = validate(g_model, test_dataloader, epoch,writer,psnr_model,ssim_model,"Test")

        # Update LR
        d_scheduler.step()
        g_scheduler.step()

        # Automatically save the model with the highest index
        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = (epoch + 1) == srgan_config.epochs
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        if not srgan_config.save_checkpoints:
            continue
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "best_psnr": best_psnr,
                "best_ssim": best_ssim,
                "state_dict": d_model.state_dict(),
                "optimizer": d_optimizer.state_dict(),
                "scheduler": d_scheduler.state_dict()
            },
            f"d_epoch_{epoch + 1}.pth.tar",
            samples_dir,
            results_dir,
            "d_best.pth.tar",
            "d_last.pth.tar",
            is_best,
            is_last
        )
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "best_psnr": best_psnr,
                "best_ssim": best_ssim,
                "state_dict": g_model.state_dict(),
                "optimizer": g_optimizer.state_dict(),
                "scheduler": g_scheduler.state_dict()
            },
            f"g_epoch_{epoch + 1}.pth.tar",
            samples_dir,
            results_dir,
            "g_best.pth.tar",
            "g_last.pth.tar",
            is_best,
            is_last
        )


def build_dataloaders() -> tuple[DataLoader, DataLoader]:
    # Load train, test and valid datasets
    train_datasets = TrainImageDataset(
        srgan_config.train_gt_images_dir,
        srgan_config.upscale_factor,
    )
    test_datasets = TestImageDataset(
        srgan_config.test_gt_images_dir, 
        srgan_config.test_lr_images_dir
    )

    # Generator all dataloader
    train_dataloader = DataLoader(
        train_datasets,
        batch_size=srgan_config.batch_size,
        shuffle=True,
        num_workers=srgan_config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        test_datasets,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    return train_dataloader, test_dataloader


def build_model() -> tuple[nn.Module, nn.Module, nn.Module]:
    d_model = model.Discriminator()
    g_model = model.Generator(
        in_channels=srgan_config.in_channels,
        out_channels=srgan_config.out_channels,
        channels=srgan_config.channels,
        num_rcb=srgan_config.num_rcb,
        upscale_factor=srgan_config.upscale_factor
    )

    d_model = d_model.to(device=srgan_config.device)
    g_model = g_model.to(device=srgan_config.device)

    return d_model, g_model


def define_loss() -> None:
    """ Initialize the loss functions by moving them to the cuda device """
    for name, loss in srgan_config.g_losses.items():
        srgan_config.g_losses[name] = loss.to(device=srgan_config.device)


def define_optimizer(d_model, g_model) -> tuple[optim.Adam, optim.Adam]:
    d_optimizer = optim.Adam(d_model.parameters(),
                             srgan_config.model_lr,
                             srgan_config.model_betas,
                             srgan_config.model_eps,
                             srgan_config.model_weight_decay)
    g_optimizer = optim.Adam(g_model.parameters(),
                             srgan_config.model_lr,
                             srgan_config.model_betas,
                             srgan_config.model_eps,
                             srgan_config.model_weight_decay)

    return d_optimizer, g_optimizer


def define_scheduler(d_optimizer: optim.Adam, g_optimizer: optim.Adam) -> tuple[lr_scheduler.StepLR, lr_scheduler.StepLR]:
    d_scheduler = lr_scheduler.StepLR(
        d_optimizer,
        srgan_config.lr_scheduler_step_size,
        srgan_config.lr_scheduler_gamma
    )
    g_scheduler = lr_scheduler.StepLR(
        g_optimizer,
        srgan_config.lr_scheduler_step_size,
        srgan_config.lr_scheduler_gamma
    )
    return d_scheduler, g_scheduler


def train(
    d_model: nn.Module,
    g_model: nn.Module,
    train_dataloader: DataLoader,
    loss_fns,
    d_optimizer: optim.Adam,
    g_optimizer: optim.Adam,
    epoch: int,
    writer: SummaryWriter
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_dataloader)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    
    loss_meters = [AverageMeter(name, ":6.6f") for name, _ in loss_fns.items()]
    d_gt_probabilities = AverageMeter("D(GT)", ":6.3f")
    d_sr_probabilities = AverageMeter("D(SR)", ":6.3f")
    
    progress = ProgressMeter(
        num_batches = batches,
        meters = [batch_time, data_time] + loss_meters + [d_gt_probabilities, d_sr_probabilities],
        prefix = f"Epoch: [{epoch + 1}]"
    )

    # Put the adversarial network model in training mode
    d_model.train()
    g_model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Get the initialization training time
    end = time.time()
    for gt, lr in train_dataloader:
        
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)
        
        # Transfer in-memory data to CUDA devices to speed up training
        gt = gt.to(device=srgan_config.device, non_blocking=True)
        lr = lr.to(device=srgan_config.device, non_blocking=True)

        # Set the real sample label to 1, and the false sample label to 0
        batch_size, _, height, width = gt.shape
        real_label = torch.full([batch_size, 1], 1.0, dtype=gt.dtype, device=srgan_config.device)
        fake_label = torch.full([batch_size, 1], 0.0, dtype=gt.dtype, device=srgan_config.device)

        # Start training the discriminator model
        # During discriminator model training, enable discriminator model backpropagation
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = True

        # Initialize the discriminator model gradients
        d_model.zero_grad(set_to_none=True)

        # Calculate the classification score of the discriminator model for real samples
        gt_output = d_model(gt)
        d_loss_gt = loss_fns['AdversarialLoss'](gt_output, real_label)
        
        
        # Call the gradient scaling function in the mixed precision API to
        # back-propagate the gradient information of the fake samples
        d_loss_gt.backward(retain_graph=True)

        # Calculate the classification score of the discriminator model for fake samples
        # Use the generator model to generate fake samples
        sr = g_model(lr)
        sr_output = d_model(sr.detach().clone())
        
        d_loss_sr = loss_fns['AdversarialLoss'](sr_output, fake_label)
        
        # Call the gradient scaling function in the mixed precision API to
        # back-propagate the gradient information of the fake samples
        d_loss_sr.backward()

        # Calculate the total discriminator loss value
        d_loss = d_loss_gt + d_loss_sr

        # Improve the discriminator model's ability to classify real and fake samples
        d_optimizer.step()
        # Finish training the discriminator model

        # Start training the generator model
        # During generator training, turn off discriminator backpropagation
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = False

        # Initialize generator model gradients
        g_model.zero_grad(set_to_none=True)
        
        loss_vals = []
        g_loss = srgan_config.loss_weights['AdversarialLoss'] * loss_fns['AdversarialLoss'](d_model(sr), real_label)
        loss_vals.append(g_loss.detach().cpu().numpy())
        
        for name, loss_fn in loss_fns.items():
            if name == "AdversarialLoss":
                continue
            elif name == "BBLoss":
                val = srgan_config.loss_weights[name] * loss_fn(sr, gt)
            else:
                val = srgan_config.loss_weights[name] * loss_fn(sr, gt)
            
            g_loss += val
            loss_vals.append(val.detach().cpu().numpy())
        
        g_loss.backward()

        # Encourage the generator to generate higher quality fake samples, making it easier to fool the discriminator
        g_optimizer.step()
        # Finish training the generator model

        # Calculate the score of the discriminator on real samples and fake samples,
        # the score of real samples is close to 1, and the score of fake samples is close to 0
        d_gt_probability = torch.sigmoid_(torch.mean(gt_output.detach()))
        d_sr_probability = torch.sigmoid_(torch.mean(sr_output.detach()))

        # Statistical accuracy and loss value for terminal data output
        for meter, loss_val in zip(loss_meters, loss_vals):
            meter.update(loss_val.item(), lr.size(0))

        d_gt_probabilities.update(d_gt_probability.item(), lr.size(0))
        d_sr_probabilities.update(d_sr_probability.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % srgan_config.train_print_frequency == 0:
            iters = batch_index + epoch * batches + 1
            writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            
            for (name, _), loss_val in zip(loss_fns.items(), loss_vals):
                writer.add_scalar(f"Train/{name}", loss_val.item(), iters)
            writer.add_scalar("Train/D(GT)_Probability", d_gt_probability.item(), iters)
            writer.add_scalar("Train/D(SR)_Probability", d_sr_probability.item(), iters)
            progress.display(batch_index + 1)


        # After training a batch of data, add 1 to the number of data batches to ensure that the
        # terminal print data normally
        batch_index += 1


def validate(
        g_model: nn.Module,
        test_dataloader: DataLoader,
        epoch: int,
        writer: SummaryWriter,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        mode: str
) -> tuple[float, float]:
    # Calculate how many batches of data are in each Epoch
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(test_dataloader), [batch_time, psnres, ssimes], prefix=f"{mode}: ")

    # Put the adversarial network model in validation mode
    g_model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        for gt, lr in test_dataloader:
            # Transfer the in-memory data to the CUDA device to speed up the test
            gt = gt.to(device=srgan_config.device, non_blocking=True)
            lr = lr.to(device=srgan_config.device, non_blocking=True)

            # Use the generator model to generate a fake sample
            sr = g_model(lr)

            # Statistical loss value for terminal data output
            psnr = psnr_model(sr, gt)
            ssim = ssim_model(sr, gt)
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % srgan_config.valid_print_frequency == 0:
                progress.display(batch_index + 1)

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
        writer.add_scalar(f"{mode}/SSIM", ssimes.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return psnres.avg, ssimes.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'SRGAN-ST',
        description = 'Does super resolution',
        epilog = 'Bottom text'
    )
    parser.add_argument('-exp_name', type=str, help='The name of the experiment', default="SRGAN_x4-UNNAMED-EXP")
    parser.add_argument('-model_name', type=str, help="The loss functions to use")
    parser.add_argument('-epochs', type=int, help="The number of epochs to run")

    args = parser.parse_args()

    srgan_config.mode = "train"
    srgan_config.exp_name = args.exp_name
    srgan_config.epochs = args.epochs
    srgan_config.lr_scheduler_step_size = args.epochs // 2

    if args.model_name == "srgan":
        srgan_config.g_losses = srgan_config.srgan_losses
    elif args.model_name == "bbgan":
        srgan_config.g_losses = srgan_config.bbgan_losses
    elif args.model_name == "gramgan":
        srgan_config.g_losses = srgan_config.gramgan_losses
    elif args.model_name == "stgan":
        srgan_config.g_losses = srgan_config.stgan_losses
    else:
        raise NotImplementedError(f"The model '{args.model_name}' is not implemented")

    main()