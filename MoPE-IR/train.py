

import subprocess
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import Callback
from torchvision.models import vgg16
from torchvision import transforms
import torch.nn.functional as F
from utils.val_utils import compute_psnr_ssim
from net.moe import log_moe_statistics
from net.moe import log_moe_statistics

import lpips
from pytorch_msssim import ssim
import sys
sys.path.append(r"/opt/data/private/bfs_files/PromptIR-main/stable_diffusion")
sys.path.append("/opt/data/private/bfs_files/PromptIR-main/stable_diffusion/taming")
class PrintEpochLossCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return  
        train_loss = trainer.callback_metrics.get("train_loss", None)
        l1_loss = trainer.callback_metrics.get("l1_loss", None)
        ssim_loss = trainer.callback_metrics.get("ssim_loss", None)
        lpips_loss = trainer.callback_metrics.get("lpips_loss", None)
        aux_loss = trainer.callback_metrics.get("aux_loss", None)
        psnr = trainer.callback_metrics.get("train_psnr", None)
        ssim_val = trainer.callback_metrics.get("train_ssim", None)
        if all(v is not None for v in [train_loss, l1_loss, ssim_loss, lpips_loss, psnr, ssim_val]):
            print(
                f"Epoch {trainer.current_epoch} completed, "
                f"train_loss: {train_loss:.4f}, "
                f"ssim_loss: {ssim_loss:.4f}, "
                f"lpips_loss: {lpips_loss:.4f}, "
                f"l1_loss: {l1_loss:.4f}, "
                f"aux_loss: {aux_loss:.4f}, "
                f"PSNR: {psnr:.2f}, SSIM: {ssim_val:.4f}"
            )

class PromptIRModel(pl.LightningModule):
    def __init__(self, use_diffusion=False, diffusion_ckpt=None, diffusion_steps=50):
        super().__init__()
        self.net = PromptIR(
            decoder=True,
            use_diffusion=use_diffusion,
            diffusion_ckpt=diffusion_ckpt,
            diffusion_steps=diffusion_steps
        )

        self.loss_fn = nn.L1Loss()

        self.lpips_model = lpips.LPIPS(net='vgg')

        self.ssim_fn = ssim

        self.w_ssim = 0.1
        self.w_lpips = 0.5
        self.w_aux = 0.1

        self._eps = 1e-8

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch

        restored, aux_loss = self.net(degrad_patch)

        if restored is None:
            print(f"[Batch {batch_idx}] 模型输出为 None，跳过该 batch")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        if clean_patch is None:
            print(f"[Batch {batch_idx}] clean_patch 为 None，跳过该 batch")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        l1 = self.loss_fn(restored, clean_patch)

        try:
            ssim_val = self.ssim_fn(restored, clean_patch, data_range=1.0, size_average=True)
            ssim_loss = 1.0 - ssim_val
            ssim_loss = torch.clamp(ssim_loss, min=0.0)
        except Exception as e:
            print(f"SSIM compute error: {e}")
            ssim_loss = torch.tensor(0.0, device=self.device)

        try:
            if next(self.lpips_model.parameters()).device != self.device:
                self.lpips_model.to(self.device)

            restored_lpips = restored * 2.0 - 1.0
            clean_lpips = clean_patch * 2.0 - 1.0
            lpips_out = self.lpips_model(restored_lpips, clean_lpips)
            lpips_loss = lpips_out.mean()
            if torch.isnan(lpips_loss) or lpips_loss < 0:
                lpips_loss = torch.clamp(lpips_loss, min=0.0)
        except Exception as e:
            print(f"LPIPS compute error: {e}")
            lpips_loss = torch.tensor(0.0, device=self.device)

        if aux_loss is None:
            aux_loss = torch.tensor(0.0, device=self.device)
        else:
            if torch.isnan(aux_loss) or torch.isinf(aux_loss):
                aux_loss = torch.tensor(0.0, device=self.device)

        total_loss = l1 + self.w_ssim * ssim_loss + self.w_lpips * lpips_loss + self.w_aux * aux_loss

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"[Batch {batch_idx}] total_loss 无效 (NaN/Inf)，替换为 L1")
            total_loss = l1.detach()

        try:
            psnr, ssim_metric, _ = compute_psnr_ssim(restored, clean_patch)
        except Exception:
            psnr, ssim_metric = torch.tensor(0.0), torch.tensor(0.0)

        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("l1_loss", l1, on_step=False, on_epoch=True)
        self.log("ssim_loss", ssim_loss, on_step=False, on_epoch=True)
        self.log("lpips_loss", lpips_loss, on_step=False, on_epoch=True)
        self.log("aux_loss", aux_loss, on_step=False, on_epoch=True)
        self.log("train_psnr", psnr, on_step=False, on_epoch=True)
        self.log("train_ssim", ssim_metric, on_step=False, on_epoch=True)

        if batch_idx % 6000 == 0:
            print(f"[Batch {batch_idx}] l1={l1.item():.6f}, ssim_loss={ssim_loss.item():.6f}, "
                  f"lpips={lpips_loss.item():.6f}, aux={aux_loss.item():.6f}, total={total_loss.item():.6f}")

        return total_loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-6)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=150)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def main():
    print("Options")
    print(opt)
    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath=opt.ckpt_dir, every_n_epochs=1, save_top_k=-1)
    print_loss_callback = PrintEpochLossCallback()
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    model = PromptIRModel(
        use_diffusion=opt.use_diffusion,
        diffusion_ckpt=opt.diffusion_ckpt,
        diffusion_steps=opt.diffusion_steps
    )
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        callbacks=[checkpoint_callback, print_loss_callback],
        gradient_clip_val=1.0,  
        precision=32,
        benchmark=True
    )
    trainer.fit(model=model, train_dataloaders=trainloader, ckpt_path=opt.ckpt_path)
if __name__ == '__main__':
    main()


