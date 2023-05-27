import os

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import model

from dataset import TrainImageDataset, TestImageDataset
from image_quality_assessment import PSNR, SSIM

# from torchmetrics.functional import structural_similarity_index_measure
# from torchmetrics import PeakSignalNoiseRatio

# import srgan_config
from loss import ContentLoss

from config import Config
from utils import init_random_seed


def train(config: Config = None):
    
    # Set seed
    init_random_seed(config.DATA.SEED)

    # House keeping variables
    batches_done = 0
    best_psnr = best_ssim = 0.0
    loss_values = dict()

    # Define models
    # discriminator = model.Discriminator(config.MODEL.D_IN_CHANNEL, config.MODEL.D_N_CHANNEL).to(config.MODEL.DEVICE)
    # generator = model.Generator(3,3,64,23,32).to(config.MODEL.DEVICE)

    discriminator = model.Discriminator().to(config.MODEL.DEVICE)
    generator = model.Generator(config).to(config.MODEL.DEVICE)

    # Define losses
    adversarial_criterion = nn.BCEWithLogitsLoss()

    # Optimizers
    d_optimizer = optim.Adam(
        params = discriminator.parameters(),
        lr     = config.SOLVER.D_BASE_LR,
        betas  = (config.SOLVER.D_BETA1, config.SOLVER.D_BETA2),
        eps    = config.SOLVER.D_EPS,
        weight_decay = config.SOLVER.D_WEIGHT_DECAY
    )
    g_optimizer = optim.Adam(
        params = generator.parameters(),
        lr     = config.SOLVER.G_BASE_LR,
        betas  = (config.SOLVER.G_BETA1, config.SOLVER.G_BETA2),
        eps    = config.SOLVER.G_EPS,
        weight_decay = config.SOLVER.G_WEIGHT_DECAY
    )

    # Schedulers
    d_scheduler = lr_scheduler.StepLR(
        optimizer = d_optimizer,
        step_size = config.SCHEDULER.STEP_SIZE,
        gamma = config.SCHEDULER.GAMMA
    )
    g_scheduler = lr_scheduler.StepLR(
        optimizer = g_optimizer,
        step_size = config.SCHEDULER.STEP_SIZE,
        gamma = config.SCHEDULER.GAMMA
    )

    # Dataloaders
    # Load train, test and valid datasets
    train_datasets = TrainImageDataset(config.DATA.TRAIN_GT_IMAGES_DIR, config.DATA.UPSCALE_FACTOR)
    test_datasets = TestImageDataset(config.DATA.TEST_GT_IMAGES_DIR, config.DATA.TEST_LR_IMAGES_DIR)

    # Generator all dataloader
    train_dataloader = DataLoader(
        dataset = train_datasets,
        batch_size = config.DATA.BATCH_SIZE,
        shuffle = True,
        num_workers = 1,
        pin_memory = True,
        drop_last = True,
        persistent_workers = True,
    )
    test_dataloader = DataLoader(
        dataset = test_datasets,
        batch_size = 1,
        shuffle = False,
        num_workers = 1,
        pin_memory = True,
        drop_last = False,
        persistent_workers = True,
    )

    # Create an IQA evaluation model
    ssim_model = SSIM(config.DATA.UPSCALE_FACTOR, True).to(device=config.MODEL.DEVICE)
    psnr_model = PSNR(config.DATA.UPSCALE_FACTOR, True).to(device=config.MODEL.DEVICE)

    # Init Tensorboard writer to store train and test info
    # also save the config used in this run to Tensorboard
    writer = SummaryWriter(f"samples/logs/{config.EXP.NAME}")
    writer.add_text("Config/Params", config.get_all_params())

    for epoch in range(config.EXP.START_EPOCH, config.EXP.N_EPOCHS):
        print(f"Beginning train epoch: {epoch+1}")

        # ----------------
        #  Train
        # ----------------
            
        for batch_num, (gt, lr) in enumerate(train_dataloader):
            batches_done += 1

            discriminator.train()
            generator.train()

            # Transfer in-memory data to CUDA devices to speed up training
            gt = gt.to(device=config.MODEL.DEVICE, non_blocking=True)
            lr = lr.to(device=config.MODEL.DEVICE, non_blocking=True)

            # Set the real sample label to 1, and the false sample label to 0
            real_label = torch.full([config.DATA.BATCH_SIZE, 1], 1.0 - config.EXP.LABEL_SMOOTHING, dtype=gt.dtype, device=config.MODEL.DEVICE)
            fake_label = torch.full([config.DATA.BATCH_SIZE, 1], 0.0, dtype=gt.dtype, device=config.MODEL.DEVICE)

            # ----------------
            #  Update Generator
            # ----------------
            g_optimizer.zero_grad()
            generator.zero_grad()

            for p in discriminator.parameters():
                p.requires_grad = False

            sr = generator(lr)
            
            # Warmup generator
            if batches_done < config.EXP.N_WARMUP_BATCHES:

                # Calculate loss for the warmup criterions
                warmup_loss = torch.tensor(0.0, device=config.MODEL.DEVICE)
                for name in config.MODEL.G_LOSS.WARMUP_CRITERIONS:
                    fn = config.MODEL.G_LOSS.CRITERIONS[name]
                    weight = config.MODEL.G_LOSS.CRITERION_WEIGHTS[name]
                    warmup_loss += fn(sr, gt) * weight

                warmup_loss.backward()
                g_optimizer.step()
                g_scheduler.step()

                writer.add_scalar("Train/G_Loss", warmup_loss.item(), batches_done)
                if batch_num % config.LOG_TRAIN_PERIOD == 0:
                    print(f"[Epoch {epoch+1}/{config.EXP.N_EPOCHS}] [Batch {batch_num}/{len(train_dataloader)}] [Warmup loss: {warmup_loss.item()}]")
                continue


            # Calculate Generator loss
            g_loss = torch.tensor(0.0, device=config.MODEL.DEVICE)
            for name, criterion in config.MODEL.G_LOSS.CRITERIONS.items():
                weight = config.MODEL.G_LOSS.CRITERION_WEIGHTS[name]
                if name == 'Adversarial':
                    # Extract validity predictions from discriminator
                    # pred_sr = discriminator(sr).detach() # [0,1]
                    # loss = weight * torch.log(real_label - pred_sr)
                    # pred_sr = discriminator(sr)
                    # pred_gt = discriminator(gt)#.detach()
                    # print("preds sr",pred_sr, pred_gt)
                    
                    # print("FGH",pred_sr - pred_gt)
                    # real_loss = 0.5 * criterion(pred_gt - torch.mean(pred_sr), fake_label)
                    # fake_loss = 0.5 * criterion(pred_sr - torch.mean(pred_gt), real_label)
                    # loss = weight * (real_loss + fake_loss)
                    # loss = weight * torch.sum(ones - pred_sr)#(torch.sum(ones) - torch.sum(pred_sr)) / torch.sum(ones)
                    # print("Generator adv loss", loss)
                    loss = criterion(discriminator(sr),real_label)
                else:
                    loss = criterion(sr, gt)
                g_loss += loss * weight
                loss_values[name] = loss.item() # Used for logging to Tensorboard


            g_loss.backward()
            g_optimizer.step()
            g_scheduler.step()

            # --------------------
            #  Update Discriminator
            # --------------------
            d_optimizer.zero_grad()
            discriminator.zero_grad()
            for p in discriminator.parameters():
                p.requires_grad = True

            # pred_sr = discriminator(sr).detach()    # 16 tal [0-1]
            # pred_gt = discriminator(gt)             # 16 tal [0-1]
            # real_loss = adversarial_criterion(pred_gt - torch.mean(pred_sr), real_label)
            # real_loss.backward()

            # pred_sr = discriminator(sr.detach())    # 16 tal [0-1]
            # fake_loss = adversarial_criterion(pred_sr - torch.mean(pred_gt.detach()), fake_label)
            # fake_loss.backward()
            # real_loss = 0.5 * adversarial_criterion(pred_gt, real_label)
            # fake_loss = 0.5 * adversarial_criterion(pred_sr, fake_label)
            # d_loss = torch.sum(pred_sr) + torch.sum(real_label - pred_gt)
            gt_output = discriminator(gt)
            d_loss_gt = adversarial_criterion(gt_output, real_label)

            # backpropagate discriminator's loss on real samples
            d_loss_gt.backward()

            # Calculate the classification score of the generated samples by the discriminator model
            sr_output = discriminator(sr.detach().clone())
            d_loss_sr = adversarial_criterion(sr_output, fake_label)
            d_loss_sr.backward()

            d_loss = d_loss_gt + d_loss_sr
            # d_loss = (real_loss + fake_loss) / 2
            # print("Discriminator adv loss", d_loss)
            # real_loss = (torch.sum(ones) - torch.sum(pred_gt)) / torch.sum(ones)
            # fake_loss = torch.sum(pred_sr) / torch.sum(ones)

            # d_loss = torch.sum(torch.log(pred_sr))
            # d_loss = 0.5 * real_loss + 0.5 * fake_loss
            
            # d_loss.backward()
            d_optimizer.step()
            d_scheduler.step()

            # -------------
            #  Log Progress
            # -------------
            if batch_num % config.LOG_TRAIN_PERIOD != 0:
                continue
        
            # Log to TensorBoard
            writer.add_scalar("Train/D_Loss", d_loss.item(), batches_done)
            writer.add_scalar("Train/G_Loss", g_loss.item(), batches_done)
            for name, loss in loss_values.items():
                writer.add_scalar(f"Train/{name}", loss, batches_done)
            writer.add_scalar("Train/D(GT)_Probability", torch.sigmoid_(torch.mean(gt_output.detach())).item(), batches_done)
            writer.add_scalar("Train/D(SR)_Probability", torch.sigmoid_(torch.mean(sr_output.detach())).item(), batches_done)

            # Print to terminal / log
            print(f"[Epoch {epoch+1}/{config.EXP.N_EPOCHS}] [Batch {batch_num}/{len(train_dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [G losses: {loss_values}]")

        # Dont do validating if we are warming up
        if batches_done < config.EXP.N_WARMUP_BATCHES:
            continue
        # ----------------
        #  Validate
        # ----------------
        generator.eval()

        avg_psnr = 0.0
        avg_ssim = 0.0
        with torch.no_grad():
            for batch_num, (gt, lr) in enumerate(test_dataloader):
                
                gt = gt.to(device=config.MODEL.DEVICE, non_blocking=True)
                lr = lr.to(device=config.MODEL.DEVICE, non_blocking=True)

                sr = generator(lr)

                print("lr", lr.min(), lr.max(), lr.shape)
                print("sr", sr.min(), sr.max(), sr.shape)
                print("gt", gt.min(), gt.max(), gt.shape)
                psnr = psnr_model(sr, gt)
                ssim = ssim_model(sr, gt)

                avg_psnr += psnr
                avg_ssim += ssim

                # Print training log information
                if batch_num % config.LOG_VALIDATION_PERIOD == 0:
                    print(f"[Test: {batch_num+1}/{len(test_dataloader)}] [PSNR: {psnr.item()}] [SSIM: {ssim.item()}]")
                    # print(f"[Test: {batch_num+1}/{len(test_dataloader)}] [PSNR1: {psnr1.item()}] [SSIM1: {ssim1.item()}]")

        # Write avg PSNR and SSIM to Tensorflow and logs
        avg_psnr = (avg_psnr / len(test_dataloader)).item()
        avg_ssim = (avg_ssim / len(test_dataloader)).item()
        writer.add_scalar(f"Test/PSNR", avg_psnr, epoch + 1)
        writer.add_scalar(f"Test/SSIM", avg_ssim, epoch + 1)
        
        print(f"-----\n[AVG PSNR: {avg_psnr}] [AVG SSIM: {avg_ssim}]\n-----\n")
        
        # Update learning rate
        # d_scheduler.step()
        # g_scheduler.step()

        # ----------------
        #  Save best model
        # ----------------
        is_best = best_psnr < psnr and best_ssim < ssim
        is_last = config.EXP.START_EPOCH + epoch == config.EXP.N_EPOCHS - 1

        results_dir = f"results/{config.EXP.NAME}"
        os.makedirs(results_dir, exist_ok=True)
        if is_last:
            torch.save(generator.state_dict(), results_dir  + "/g_last.pth")
            torch.save(discriminator.state_dict(), results_dir  + "/d_last.pth")
        if is_best:
            torch.save(generator.state_dict(), results_dir  + "/g_best.pth")
            torch.save(discriminator.state_dict(), results_dir  + "/d_best.pth")
            best_psnr = psnr
            best_ssim = ssim

        #TODO: Maybe run test.py here and save to a grid of output images to see progress.