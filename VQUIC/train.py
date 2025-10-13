# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from compressai.zoo import load_state_dict
import argparse
import math
import random
import sys
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from ELICUtilis.datasets.utils import ImageFolder

from tensorboardX import SummaryWriter
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from ELICUtilis.utilis.utilis import DelfileList
from Network import TestModel
from lpips import LPIPS


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.bce = nn.BCELoss()
        self.lpips = LPIPS().cuda()
        self.l1 = nn.L1Loss()

    def forward(self, output, target, mask, disc_factor=0.):
        N, _, H, W = target.size()
        out = {}

        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["bce_loss"] = self.bce(output["mask"], mask)
        out["lpips_loss"] = self.lpips(output["x_hat"], target).mean()
        out["quant_loss"] = output["vq_loss"]
        out["f_loss"] = output["f_loss"]
        out["con_loss"] = torch.clamp(1.0 + 10 * output["good_vq_loss"] - 10 * output["bad_vq_loss"], min=0.0)
        out["gan_loss"] = disc_factor * output["gan_loss"]

        out["train_loss"] = (out["mse_loss"] + out["quant_loss"] + 0.001 * out["con_loss"] +
                             disc_factor * output["gen_loss"] + out["lpips_loss"])
        out["test_loss"] = out["mse_loss"] + out["quant_loss"] + out["lpips_loss"]
        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    # Freeze specific modules
    net.good_quantize.embedding.weight.requires_grad = False
    net.bad_quantize.embedding.weight.requires_grad = False
    
    for param in net.decoder.parameters():
        param.requires_grad = False

    for param in net.suimnet.parameters():
        param.requires_grad = False

    for param in net.criterionVGG.parameters():
        param.requires_grad = False

    # Separate parameters for different optimizers
    parameters = {
        n for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad and not n.startswith("discriminator")
    }
    
    disc_parameters = {
        n for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad and n.startswith("discriminator")
    }

    aux_parameters = {
        n for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate, betas=(0.9, 0.999)
    )
    
    optimizer_disc = optim.Adam(
        (params_dict[n] for n in sorted(disc_parameters)),
        lr=args.learning_rate, eps=1e-08, betas=(0.9, 0.999)
    )

    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate, betas=(0.9, 0.999)
    )
    
    return optimizer, optimizer_disc, aux_optimizer


def adopt_weight(disc_factor, i, threshold, value=0.):
    if i < threshold:
        disc_factor = value
    return disc_factor


def train_one_epoch(
        model, criterion, train_dataloader, optimizer, optimizer_disc, aux_optimizer, 
        epoch, clip_max_norm, noisequant=True
):
    model.train()
    device = next(model.parameters()).device
    
    # Initialize metrics
    train_loss = AverageMeter()
    train_mse_loss = AverageMeter()
    train_bce_loss = AverageMeter()
    train_f_loss = AverageMeter()
    train_con_loss = AverageMeter()
    train_vq_loss = AverageMeter()
    
    start = time.time()
    
    for i, [d, mask_d] in enumerate(train_dataloader):
        d = d.to(device)
        mask_d = mask_d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        optimizer_disc.zero_grad()
        
        out_net = model(d, noisequant)
        out_criterion = criterion(out_net, d, mask_d.detach(), disc_factor=0.0)

        # Update metrics
        train_loss.update(out_criterion["train_loss"].item())
        train_mse_loss.update(out_criterion["mse_loss"].item())
        train_bce_loss.update(out_criterion["bce_loss"].item())
        train_f_loss.update(out_criterion["f_loss"].item())
        train_con_loss.update(out_criterion["con_loss"].item())
        train_vq_loss.update(out_criterion["quant_loss"].item())
        
        # Backward passes
        out_criterion["train_loss"].backward(retain_graph=True)
        out_criterion["gan_loss"].backward()
        
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            
        optimizer.step()
        optimizer_disc.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10000 == 0:
            print(
                f"Train epoch {epoch}: [{i * len(d)}/{len(train_dataloader.dataset)} "
                f"({100. * i / len(train_dataloader):.0f}%)] "
                f'Loss: {out_criterion["train_loss"].item():.3f} | '
                f'MSE: {out_criterion["mse_loss"].item():.3f} | '
                f'BCE: {out_criterion["bce_loss"].item():.3f} | '
                f'Feature: {out_criterion["f_loss"].item():.3f} | '
                f'Contrastive: {out_criterion["con_loss"].item():.3f} | '
                f'VQ: {out_criterion["quant_loss"].item():.3f} | '
                f"Aux: {aux_loss.item():.2f}"
            )
            
    print(
        f"Train epoch {epoch}: Average losses: "
        f"Loss: {train_loss.avg:.3f} | "
        f"MSE: {train_mse_loss.avg:.3f} | "
        f"BCE: {train_bce_loss.avg:.3f} | "
        f"Feature: {train_f_loss.avg:.3f} | "
        f"Contrastive: {train_con_loss.avg:.3f} | "
        f"VQ: {train_vq_loss.avg:.3f} | "
        f"Time (s): {time.time() - start:.4f}"
    )

    return train_loss.avg, train_mse_loss.avg


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    # Initialize metrics
    loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    bce_loss = AverageMeter()
    f_loss = AverageMeter()
    con_loss = AverageMeter()

    with torch.no_grad():
        for [d, mask_d] in test_dataloader:
            d = d.to(device)
            mask_d = mask_d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d, mask_d.detach())

            aux_loss.update(model.aux_loss().item())
            loss.update(out_criterion["test_loss"].item())
            mse_loss.update(out_criterion["mse_loss"].item())
            bce_loss.update(out_criterion["bce_loss"].item())
            f_loss.update(out_criterion["f_loss"].item())
            con_loss.update(out_criterion["con_loss"].item())
            
    print(
        f"Test epoch {epoch}: Average losses: "
        f"Loss: {loss.avg:.3f} | "
        f"MSE: {mse_loss.avg:.3f} | "
        f"BCE: {bce_loss.avg:.3f} | "
        f"Feature: {f_loss.avg:.3f} | "
        f"Contrastive: {con_loss.avg:.3f} | "
        f"Aux: {aux_loss.avg:.4f}\n"
    )

    return loss.avg, mse_loss.avg


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "--N",
        default=192,
        type=int,
        help="Number of channels of main codec",
    )
    parser.add_argument(
        "--M",
        default=320,
        type=int,
        help="Number of channels of latent",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=4000,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=15e-3,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", default=1926, type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="use the pretrain model to refine the models",
    )
    parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--savepath', default='./checkpoint', type=str, help='Path to save the checkpoint')
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")

    parser.add_argument("--goodcodebookcheckpoint", default='./goodcodebook16384_32.pth.tar', type=str, help="Path to a good codebook checkpoint")
    parser.add_argument("--badcodebookcheckpoint", default='./badcodebook16384_32.pth.tar', type=str, help="Path to a bad codebook checkpoint")
    parser.add_argument("--pretrainedcodebookcheckpoint", default='./last16384_f16.ckpt', type=str, help="Path to a pretrained codebook checkpoint")

    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size)]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size)]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )


    # Load checkpoints
    checkpoint = torch.load(args.checkpoint)
    pretrained_codebook = torch.load(args.pretrainedcodebookcheckpoint, weights_only=False)
    good_codebook = torch.load(args.goodcodebookcheckpoint)
    bad_codebook = torch.load(args.badcodebookcheckpoint)
    
    checkpoint['quantize.embedding.weight'] = pretrained_codebook['state_dict']['quantize.embedding.weight']
    checkpoint['good_quantize.embedding.weight'] = good_codebook['state_dict']['quantize.embedding.weight']
    checkpoint['bad_quantize.embedding.weight'] = bad_codebook['state_dict']['quantize.embedding.weight']

    net = TestModel(N=args.N, M=args.M).from_state_dict(load_state_dict(checkpoint))

    net = net.to(device)


    if not os.path.exists(args.savepath):
        try:
            os.mkdir(args.savepath)
        except:
            os.makedirs(args.savepath)
    writer = SummaryWriter(args.savepath)
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    # Configure optimizers and schedulers
    optimizer, optimizer_disc, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.1)
    lr_disc_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_disc, milestones=[1000], gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    stemode = False
    
    if args.checkpoint and args.pretrained:
        optimizer.param_groups[0]['lr'] = args.learning_rate
        aux_optimizer.param_groups[0]['lr'] = args.aux_learning_rate
        del lr_scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=100)
        last_epoch = 0
        stemode = True

    # Training loop
    noisequant = True
    best_loss = float("inf")
    
    for epoch in range(last_epoch, args.epochs):
        if epoch > 1800 or stemode:
            noisequant = False
            
        print(f"Epoch {epoch} - noisequant: {noisequant}, stemode: {stemode}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        
        train_loss, train_mse = train_one_epoch(
            net, criterion, train_dataloader, optimizer, optimizer_disc, 
            aux_optimizer, epoch, args.clip_max_norm, noisequant
        )
        
        writer.add_scalar('Train/loss', train_loss, epoch)
        writer.add_scalar('Train/mse', train_mse, epoch)
        
        loss, mse = test_epoch(epoch, test_dataloader, net, criterion)
        writer.add_scalar('Test/loss', loss, epoch)
        writer.add_scalar('Test/mse', mse, epoch)
        
        lr_scheduler.step()
        lr_disc_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            DelfileList(args.savepath, "checkpoint_last")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                filename=os.path.join(args.savepath, f"checkpoint_last_{epoch}.pth.tar")
            )
            
            if is_best:
                DelfileList(args.savepath, "checkpoint_best")
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    filename=os.path.join(args.savepath, f"checkpoint_best_loss_{epoch}.pth.tar")
                )


if __name__ == "__main__":
    main(sys.argv[1:])
