# Copyright 2023 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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
import os
import random
import time
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.backends import cudnn
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import model
from dataset import CUDAPrefetcher, BaseImageDataset, PairedImageDataset
from imgproc import random_crop_torch, random_rotate_torch, random_vertically_flip_torch, random_horizontally_flip_torch
from test import test
from utils import build_iqa_model, load_resume_state_dict, load_pretrained_state_dict, make_directory, save_checkpoint, \
    Summary, AverageMeter, ProgressMeter


def main():
    # Read parameters from configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="./configs/train/SRGAN_x4-SRGAN_ImageNet-Set5.yaml",
                        help="Path to train config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    # Fixed random number seed
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed_all(config["SEED"])

    # Because the size of the input image is fixed, the fixed CUDNN convolution method can greatly increase the running speed
    cudnn.benchmark = True

    # Mixed precision training
    scaler = amp.GradScaler()

    # Default to start training from scratch
    start_epoch = 0

    # Image clarity evaluation metrics
    best_psnr = 0.0
    best_ssim = 0.0

    # Running device
    device = torch.device("cuda", config["DEVICE_ID"])

    # Define the basic functions needed to start training
    train_data_prefetcher, paired_test_data_prefetcher = load_dataset(config, device)
    teacher_model, student_model, ema_student_model, d_model = build_models_for_kd(config, device)
    pixel_criterion, content_criterion, adversarial_criterion, distillation_criterion = define_loss_with_kd(config, device)
    student_optimizer, d_optimizer = define_optimizer(student_model, d_model, config)
    student_scheduler, d_scheduler = define_scheduler(student_optimizer, d_optimizer, config)

    # Load pretrained teacher model (if available)
    if config["TRAIN"]["CHECKPOINT"]["PRETRAINED_TEACHER_MODEL"]:
        teacher_model = load_pretrained_state_dict(teacher_model,
                                                    False,
                                                    config["TRAIN"]["CHECKPOINT"]["PRETRAINED_TEACHER_MODEL"])
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['PRETRAINED_TEACHER_MODEL']}` teacher model weights successfully.")
    else:
        print("Teacher model weights are required for knowledge distillation.")

    # Load pretrained student and discriminator models
    if config["TRAIN"]["CHECKPOINT"]["PRETRAINED_STUDENT_MODEL"]:
        student_model = load_pretrained_state_dict(student_model, False, config["TRAIN"]["CHECKPOINT"]["PRETRAINED_STUDENT_MODEL"])
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['PRETRAINED_STUDENT_MODEL']}` student model weights successfully.")
    if config["TRAIN"]["CHECKPOINT"]["PRETRAINED_D_MODEL"]:
        d_model = load_pretrained_state_dict(d_model,
                                             config["MODEL"]["D"]["COMPILED"],
                                             config["TRAIN"]["CHECKPOINT"]["PRETRAINED_D_MODEL"])
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['PRETRAINED_D_MODEL']}` pretrained model weights successfully.")
    else:
        print("Pretrained dd model weights not found.")

    # Initialize the image clarity evaluation method
    psnr_model, ssim_model = build_iqa_model(
        config["SCALE"],
        config["TEST"]["ONLY_TEST_Y_CHANNEL"],
        device,
    )

    # Create the folder where the model weights are saved
    samples_dir = os.path.join("samples", config["EXP_NAME"])
    results_dir = os.path.join("results", config["EXP_NAME"])
    make_directory(samples_dir)
    make_directory(results_dir)

    # create model training log
    writer = SummaryWriter(os.path.join("samples", "logs", config["EXP_NAME"]))

    # Training loop
    for epoch in range(start_epoch, config["TRAIN"]["HYP"]["EPOCHS"]):
        train_with_kd(
            student_model,
            teacher_model,
            ema_student_model,
            d_model,
            train_data_prefetcher,
            pixel_criterion,
            content_criterion,
            adversarial_criterion,
            distillation_criterion,
            student_optimizer,
            d_optimizer,
            epoch,
            scaler,
            writer,
            device,
            config)

        # Update LR
        student_scheduler.step()
        d_scheduler.step()

        psnr, ssim = test(student_model,
                          paired_test_data_prefetcher,
                          psnr_model,
                          ssim_model,
                          device,
                          config)
        print("\n")

        # Write the evaluation indicators of each round of Epoch to the log
        writer.add_scalar(f"Test/PSNR", psnr, epoch + 1)
        writer.add_scalar(f"Test/SSIM", ssim, epoch + 1)

        # Automatically save model weights
        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = (epoch + 1) == config["TRAIN"]["HYP"]["EPOCHS"]
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        save_checkpoint({"epoch": epoch + 1,
                         "psnr": psnr,
                         "ssim": ssim,
                         "state_dict": student_model.state_dict(),
                         "ema_state_dict": ema_student_model.state_dict() if ema_student_model is not None else None,
                         "optimizer": student_optimizer.state_dict()},
                        f"epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "student_best.pth.tar",
                        "student_last.pth.tar",
                        is_best,
                        is_last)
        save_checkpoint({"epoch": epoch + 1,
                         "psnr": psnr,
                         "ssim": ssim,
                         "state_dict": d_model.state_dict(),
                         "optimizer": d_optimizer.state_dict()},
                        f"epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "d_best.pth.tar",
                        "d_last.pth.tar",
                        is_best,
                        is_last)


def load_dataset(
        config: Any,
        device: torch.device,
) -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load the train dataset
    degenerated_train_datasets = BaseImageDataset(
        config["TRAIN"]["DATASET"]["TRAIN_GT_IMAGES_DIR"],
        config["TRAIN"]["DATASET"]["TRAIN_LR_IMAGES_DIR"],
        config["SCALE"],
    )

    # Load the registration test dataset
    paired_test_datasets = PairedImageDataset(config["TEST"]["DATASET"]["PAIRED_TEST_GT_IMAGES_DIR"],
                                              config["TEST"]["DATASET"]["PAIRED_TEST_LR_IMAGES_DIR"])

    # generate dataset iterator
    degenerated_train_dataloader = DataLoader(degenerated_train_datasets,
                                              batch_size=config["TRAIN"]["HYP"]["IMGS_PER_BATCH"],
                                              shuffle=config["TRAIN"]["HYP"]["SHUFFLE"],
                                              num_workers=config["TRAIN"]["HYP"]["NUM_WORKERS"],
                                              pin_memory=config["TRAIN"]["HYP"]["PIN_MEMORY"],
                                              drop_last=True,
                                              persistent_workers=config["TRAIN"]["HYP"]["PERSISTENT_WORKERS"])
    paired_test_dataloader = DataLoader(paired_test_datasets,
                                        batch_size=config["TEST"]["HYP"]["IMGS_PER_BATCH"],
                                        shuffle=config["TEST"]["HYP"]["SHUFFLE"],
                                        num_workers=config["TEST"]["HYP"]["NUM_WORKERS"],
                                        pin_memory=config["TEST"]["HYP"]["PIN_MEMORY"],
                                        drop_last=False,
                                        persistent_workers=config["TEST"]["HYP"]["PERSISTENT_WORKERS"])

    # Replace the data set iterator with CUDA to speed up
    train_data_prefetcher = CUDAPrefetcher(degenerated_train_dataloader, device)
    paired_test_data_prefetcher = CUDAPrefetcher(paired_test_dataloader, device)

    return train_data_prefetcher, paired_test_data_prefetcher


def build_model(
        config: Any,
        device: torch.device,
) -> [nn.Module, nn.Module or Any, nn.Module]:
    # Teacher Model
    teacher_model = model.SRResNet(in_channels=3, out_channels=3, channels=64, num_rcb=16, upscale=4).to(device)
    teacher_model.eval()  # Freeze teacher model

    # Student Model
    student_model = model.StudentSRResNet(in_channels=3, out_channels=3, channels=32, num_rcb=8, upscale=4).to(device)

    # Discriminator
    d_model = model.__dict__[config["MODEL"]["D"]["NAME"]](
        in_channels=config["MODEL"]["D"]["IN_CHANNELS"],
        out_channels=config["MODEL"]["D"]["OUT_CHANNELS"],
        channels=config["MODEL"]["D"]["CHANNELS"],
    ).to(device)

    # Exponential Moving Average (EMA) for student
    ema_student_model = AveragedModel(student_model, device=device, avg_fn=None) if config["MODEL"]["EMA"]["ENABLE"] else None

    return teacher_model, student_model, ema_student_model, d_model

def build_models_for_kd(
        config: Any,
        device: torch.device,
) -> [nn.Module, nn.Module or Any, nn.Module]:
    teacher_model = model.__dict__[config["MODEL"]["TEACHER"]["NAME"]](in_channels=config["MODEL"]["TEACHER"]["IN_CHANNELS"],
                                                           out_channels=config["MODEL"]["TEACHER"]["OUT_CHANNELS"],
                                                           channels=config["MODEL"]["TEACHER"]["CHANNELS"],
                                                           num_rcb=config["MODEL"]["TEACHER"]["NUM_RCB"])
    student_model = model.__dict__[config["MODEL"]["G"]["NAME"]](in_channels=config["MODEL"]["G"]["IN_CHANNELS"],
                                                           out_channels=config["MODEL"]["G"]["OUT_CHANNELS"],
                                                           channels=config["MODEL"]["G"]["CHANNELS"],
                                                           num_rcb=config["MODEL"]["G"]["NUM_RCB"])
    d_model = model.__dict__[config["MODEL"]["D"]["NAME"]](in_channels=config["MODEL"]["D"]["IN_CHANNELS"],
                                                           out_channels=config["MODEL"]["D"]["OUT_CHANNELS"],
                                                           channels=config["MODEL"]["D"]["CHANNELS"])

    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    d_model = d_model.to(device)

    if config["MODEL"]["EMA"]["ENABLE"]:
        # Generate an exponential average model based on a generator to stabilize model training
        ema_decay = config["MODEL"]["EMA"]["DECAY"]
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
            (1 - ema_decay) * averaged_model_parameter + ema_decay * model_parameter
        ema_student_model = AveragedModel(student_model, device=device, avg_fn=ema_avg_fn)
    else:
        ema_student_model = None

    # compile model
    if config["MODEL"]["TEACHER"]["COMPILED"]:
        teacher_model = torch.compile(teacher_model)
    if config["MODEL"]["G"]["COMPILED"]:
        student_model = torch.compile(student_model)
    if config["MODEL"]["D"]["COMPILED"]:
        d_model = torch.compile(d_model)
    if config["MODEL"]["EMA"]["COMPILED"] and ema_student_model is not None:
        ema_student_model = torch.compile(ema_student_model)

    return teacher_model, student_model, ema_student_model, d_model

def define_loss_with_kd(config: Any, device: torch.device) -> [nn.MSELoss, model.ENetContentLoss, nn.BCEWithLogitsLoss, nn.L1Loss]:
    if config["TRAIN"]["LOSSES"]["PIXEL_LOSS"]["NAME"] == "MSELoss":
        pixel_criterion = nn.MSELoss()
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['PIXEL_LOSS']['NAME']} is not implemented.")

    if config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["NAME"] == "ENetContentLoss":
        content_criterion = model.ENetContentLoss(
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["NET_CFG_NAME"],
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["BATCH_NORM"],
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["NUM_CLASSES"],
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["MODEL_WEIGHTS_PATH"],
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["FEATURE_NODES"],
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["FEATURE_NORMALIZE_MEAN"],
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["FEATURE_NORMALIZE_STD"],
        )
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['CONTENT_LOSS']['NAME']} is not implemented.")

    if config["TRAIN"]["LOSSES"]["ADVERSARIAL_LOSS"]["NAME"] == "vanilla":
        adversarial_criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['ADVERSARIAL_LOSS']['NAME']} is not implemented.")

    if config["TRAIN"]["LOSSES"]["DISTILLATION_LOSS"]["NAME"] == "L1Loss":
        distillation_criterion = nn.L1Loss()
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['DISTILLATION_LOSS']['NAME']} is not implemented.")

    pixel_criterion = pixel_criterion.to(device)
    content_criterion = content_criterion.to(device)
    adversarial_criterion = adversarial_criterion.to(device)
    distillation_criterion = distillation_criterion.to(device)

    return pixel_criterion, content_criterion, adversarial_criterion, distillation_criterion


def define_optimizer(student_model: nn.Module, d_model: nn.Module, config: Any) -> [optim.Adam, optim.Adam]:
    if config["TRAIN"]["OPTIM"]["NAME"] == "Adam":
        student_optimizer = optim.Adam(student_model.parameters(),
                                 config["TRAIN"]["OPTIM"]["LR"],
                                 config["TRAIN"]["OPTIM"]["BETAS"],
                                 config["TRAIN"]["OPTIM"]["EPS"],
                                 config["TRAIN"]["OPTIM"]["WEIGHT_DECAY"])
        d_optimizer = optim.Adam(d_model.parameters(),
                                 config["TRAIN"]["OPTIM"]["LR"],
                                 config["TRAIN"]["OPTIM"]["BETAS"],
                                 config["TRAIN"]["OPTIM"]["EPS"],
                                 config["TRAIN"]["OPTIM"]["WEIGHT_DECAY"])

    else:
        raise NotImplementedError(f"Optimizer {config['TRAIN']['OPTIM']['NAME']} is not implemented.")

    return student_optimizer, d_optimizer


def define_scheduler(student_optimizer: optim.Adam, d_optimizer: optim.Adam, config: Any) -> [lr_scheduler.MultiStepLR, lr_scheduler.MultiStepLR]:
    if config["TRAIN"]["LR_SCHEDULER"]["NAME"] == "MultiStepLR":
        student_scheduler = lr_scheduler.MultiStepLR(student_optimizer,
                                               config["TRAIN"]["LR_SCHEDULER"]["MILESTONES"],
                                               config["TRAIN"]["LR_SCHEDULER"]["GAMMA"])
        d_scheduler = lr_scheduler.MultiStepLR(d_optimizer,
                                               config["TRAIN"]["LR_SCHEDULER"]["MILESTONES"],
                                               config["TRAIN"]["LR_SCHEDULER"]["GAMMA"])

    else:
        raise NotImplementedError(f"LR Scheduler {config['TRAIN']['LR_SCHEDULER']['NAME']} is not implemented.")

    return student_scheduler, d_scheduler


def train_with_kd(
        student_model: nn.Module,
        teacher_model: nn.Module,
        ema_student_model: nn.Module,
        d_model: nn.Module,
        train_data_prefetcher: CUDAPrefetcher,
        pixel_criterion: nn.L1Loss,
        content_criterion: model.ENetContentLoss,
        adversarial_criterion: nn.BCEWithLogitsLoss,
        distillation_criterion: nn.L1Loss,
        student_optimizer: optim.Adam,
        d_optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device,
        config: Any,
) -> None:
    # Calculate how many batches of data there are under a dataset iterator
    batches = len(train_data_prefetcher)

    # The information printed by the progress bar
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
    g_losses = AverageMeter("G Loss", ":6.6f", Summary.NONE)
    d_losses = AverageMeter("D Loss", ":6.6f", Summary.NONE)
    progress = ProgressMeter(batches,
                             [batch_time, data_time, g_losses, d_losses],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Set the model to training mode
    student_model.train()
    teacher_model.eval()
    d_model.train()

    # Define loss function weights
    pixel_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["PIXEL_LOSS"]["WEIGHT"]).to(device)
    feature_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["WEIGHT"]).to(device)
    adversarial_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["ADVERSARIAL_LOSS"]["WEIGHT"]).to(device)
    distilaltion_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["DISTILLATION_LOSS"]["WEIGHT"]).to(device)

    # Initialize data batches
    batch_index = 0
    # Set the dataset iterator pointer to 0
    train_data_prefetcher.reset()
    # Record the start time of training a batch
    end = time.time()
    # load the first batch of data
    batch_data = train_data_prefetcher.next()

    # Used for discriminator binary classification output, the input sample comes from the data set (real sample) is marked as 1, and the input sample comes from the generator (generated sample) is marked as 0
    batch_size = batch_data["gt"].shape[0]
    if config["MODEL"]["D"]["NAME"] == "discriminator_for_PatchGAN":
        real_label = torch.full([batch_size, 1], 1.0, dtype=torch.float, device=device)
        fake_label = torch.full([batch_size, 1], 0.0, dtype=torch.float, device=device)
    elif config["MODEL"]["D"]["NAME"] == "discriminator_for_unet":
        image_height = config["TRAIN"]["DATASET"]["GT_IMAGE_SIZE"]
        image_width = config["TRAIN"]["DATASET"]["GT_IMAGE_SIZE"]
        real_label = torch.full([batch_size, 1, image_height, image_width], 1.0, dtype=torch.float, device=device)
        fake_label = torch.full([batch_size, 1, image_height, image_width], 0.0, dtype=torch.float, device=device)
    else:
        raise ValueError(f"The `{config['MODEL']['D']['NAME']}` is not supported.")

    while batch_data is not None:
        # Load batches of data
        gt = batch_data["gt"].to(device, non_blocking=True)
        lr = batch_data["lr"].to(device, non_blocking=True)

        # image data augmentation
        gt, lr = random_crop_torch(gt,
                                   lr,
                                   config["TRAIN"]["DATASET"]["GT_IMAGE_SIZE"],
                                   config["SCALE"])
        gt, lr = random_rotate_torch(gt, lr, config["SCALE"], [0, 90, 180, 270])
        gt, lr = random_vertically_flip_torch(gt, lr)
        gt, lr = random_horizontally_flip_torch(gt, lr)

        # Record the time to load a batch of data
        data_time.update(time.time() - end)

        # start training the generator model
        # Disable discriminator backpropagation during generator training
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = False

        # Initialize the generator model gradient
        student_model.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_output = teacher_model(lr)

        # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and confrontation loss
        with amp.autocast():
            sr = student_model(lr)
            pixel_loss = pixel_criterion(sr, gt)
            feature_loss = content_criterion(sr, gt)
            adversarial_loss = adversarial_criterion(d_model(sr), real_label)
            distillation_loss = distillation_criterion(sr, teacher_output)
            pixel_loss = torch.sum(torch.mul(pixel_weight, pixel_loss))
            feature_loss = torch.sum(torch.mul(feature_weight, feature_loss))
            adversarial_loss = torch.sum(torch.mul(adversarial_weight, adversarial_loss))
            distillation_loss = torch.sum(torch.mul(distilaltion_weight, distillation_loss))
            # Compute generator total loss
            g_loss = pixel_loss + feature_loss + adversarial_loss + distillation_loss
        # Backpropagation generator loss on generated samples
        scaler.scale(g_loss).backward()
        # update generator model weights
        scaler.step(student_optimizer)
        scaler.update()
        # end training generator model

        # start training the discriminator model
        # During discriminator model training, enable discriminator model backpropagation
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = True

        # Initialize the discriminator model gradient
        d_model.zero_grad(set_to_none=True)

        # Calculate the classification score of the discriminator model on real samples
        with amp.autocast():
            gt_output = d_model(gt)
            d_loss_gt = adversarial_criterion(gt_output, real_label)

        # backpropagate discriminator's loss on real samples
        scaler.scale(d_loss_gt).backward()

        # Calculate the classification score of the generated samples by the discriminator model
        with amp.autocast():
            sr_output = d_model(sr.detach().clone())
            d_loss_sr = adversarial_criterion(sr_output, fake_label)
        # backpropagate discriminator loss on generated samples
        scaler.scale(d_loss_sr).backward()

        # Compute the discriminator total loss value
        d_loss = d_loss_gt + d_loss_sr
        # Update discriminator model weights
        scaler.step(d_optimizer)
        scaler.update()
        # end training discriminator model

        if config["MODEL"]["EMA"]["ENABLE"]:
            # update exponentially averaged model weights
            ema_student_model.update_parameters(student_model)

        # record the loss value
        d_losses.update(d_loss.item(), batch_size)
        g_losses.update(g_loss.item(), batch_size)

        # Record the total time of training a batch
        batch_time.update(time.time() - end)
        end = time.time()

        # Output training log information once
        if batch_index % config["TRAIN"]["PRINT_FREQ"] == 0:
            # write training log
            iters = batch_index + epoch * batches
            writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
            writer.add_scalar("Train/D(GT)_Loss", d_loss_gt.item(), iters)
            writer.add_scalar("Train/D(SR)_Loss", d_loss_sr.item(), iters)
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
            writer.add_scalar("Train/Feature_Loss", feature_loss.item(), iters)
            writer.add_scalar("Train/Adversarial_Loss", adversarial_loss.item(), iters)
            writer.add_scalar("Train/Distillation_Loss", distillation_loss.item(), iters)
            writer.add_scalar("Train/D(GT)_Probability", torch.sigmoid_(torch.mean(gt_output.detach())).item(), iters)
            writer.add_scalar("Train/D(SR)_Probability", torch.sigmoid_(torch.mean(sr_output.detach())).item(), iters)
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = train_data_prefetcher.next()

        # After training a batch of data, add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1


if __name__ == "__main__":
    main()
