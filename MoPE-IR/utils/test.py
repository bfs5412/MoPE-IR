





import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset, DeblurLowlightDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import PromptIR

import lightning.pytorch as pl
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

class PromptIRModel(pl.LightningModule):
    def __init__(self, use_diff_ckpt=False,use_diffusion=True, diffusion_ckpt=None, diffusion_steps=50,num_experts=5):
        super().__init__()
        self.net = PromptIR(
            decoder=True,
            use_diffusion=use_diffusion,
            diffusion_ckpt=diffusion_ckpt,
            diffusion_steps=diffusion_steps,
            num_experts=num_experts  
        )
        self.loss_fn = nn.L1Loss()

    def forward(self,x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        self.log("train_loss", loss)
        return loss

    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]

def pad_to_multiple(x, multiple=64):
    c, h, w = x.shape  
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x, pad_h, pad_w

def custom_collate_fn(batch):
    names = [item[0][0] for item in batch]
    degrad_patches = [item[1] for item in batch]
    clean_patches = [item[2] for item in batch]
    return names, degrad_patches, clean_patches

def test_Denoise(net, dataset, sigma=15):
    output_path = testopt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_sigma(sigma)
    testloader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True,
        collate_fn=custom_collate_fn
    )

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for (clean_names, degrad_patches, clean_patches) in tqdm(testloader):
            degrad_patches = [x.cuda() for x in degrad_patches]
            clean_patches = [x.cuda() for x in clean_patches]

            max_h = max([x.shape[-2] for x in degrad_patches])
            max_w = max([x.shape[-1] for x in degrad_patches])

            target_h = (max_h + 63) // 64 * 64
            target_w = (max_w + 63) // 64 * 64

            degrad_batch = []
            pad_hw_list = []  
            for x in degrad_patches:
                c, h, w = x.shape
                pad_h = target_h - h
                pad_w = target_w - w
                x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
                degrad_batch.append(x_padded)
                pad_hw_list.append((pad_h, pad_w))

            degrad_batch = torch.stack(degrad_batch, dim=0)

            restored_batch, _ = net(degrad_batch)


            for i in range(len(restored_batch)):
                pad_h, pad_w = pad_hw_list[i]
                h, w = clean_patches[i].shape[-2:]
                restored = restored_batch[i][..., :h, :w]
                print("Restored shape:", restored.shape)
                clean = clean_patches[i][..., :h, :w]

                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean)
                psnr.update(temp_psnr, N)
                ssim.update(temp_ssim, N)
                print(f"{clean_names[i]} - PSNR: {temp_psnr:.2f}, SSIM: {temp_ssim:.4f}")  
                save_image_tensor(restored, output_path + clean_names[i] + '.png')

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))

def test_Derain_Dehaze(net, dataset, task="derain"):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True,
        collate_fn=custom_collate_fn
    )
    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for (clean_names, degrad_patches, clean_patches) in tqdm(testloader):
            degrad_patches = [x.cuda() for x in degrad_patches]
            clean_patches = [x.cuda() for x in clean_patches]

            max_h = max([x.shape[-2] for x in degrad_patches])
            max_w = max([x.shape[-1] for x in degrad_patches])

            target_h = (max_h + 63) // 64 * 64
            target_w = (max_w + 63) // 64 * 64

            degrad_batch = []
            pad_hw_list = []  
            for x in degrad_patches:
                c, h, w = x.shape
                pad_h = target_h - h
                pad_w = target_w - w
                x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
                degrad_batch.append(x_padded)
                pad_hw_list.append((pad_h, pad_w))

            degrad_batch = torch.stack(degrad_batch, dim=0)

            restored_batch, _ = net(degrad_batch)

            for i in range(len(restored_batch)):
                pad_h, pad_w = pad_hw_list[i]
                h, w = clean_patches[i].shape[-2:]
                restored = restored_batch[i][..., :h, :w]
                clean = clean_patches[i][..., :h, :w]

                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean)
                psnr.update(temp_psnr, N)
                ssim.update(temp_ssim, N)
                print(f"{clean_names[i]} - PSNR: {temp_psnr:.2f}, SSIM: {temp_ssim:.4f}")  
                save_image_tensor(restored, output_path + clean_names[i] + '.png')

        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))

if __name__ == '__main__':

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')

    parser.add_argument('--denoise_path', type=str, default="data/test/denoise/", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="data/test/derain/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="data/test/dehaze/", help='save path of test hazy images')
    parser.add_argument('--deblur_path', type=str, default="data/test/deblur/", help='save path of test blur images')
    parser.add_argument('--lowlight_path', type=str, default="data/test/lowlight/", help='save path of test lowlight images')

    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="epoch=113-step=574674.ckpt", help='checkpoint save path')
    parser.add_argument('--use_diffusion', type=bool, default=True, help='使用潜在扩散模型进行图像恢复')
    parser.add_argument('--diffusion_ckpt', type=str, default=r"/opt/data/private/bfs_files/PromptIR-main/ckpt/v1-5-pruned-emaonly.safetensors", help='潜在扩散模型权重路径')
    parser.add_argument('--diffusion_steps', type=int, default=30, help='扩散模型采样步数')
    parser.add_argument('--use_diff_ckpt', type=bool, default=False, help='使用潜在扩散模型进行图像恢复')
    testopt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    ckpt_path = "train_MoeDiff260413_Haze_1D_ckpt/" + testopt.ckpt_name

    denoise_splits = ["BSD68/"]
    derain_splits = ["Rain100L/"]

    denoise_tests = []
    derain_tests = []

    base_path = testopt.denoise_path
    for i in denoise_splits:
        testopt.denoise_path = os.path.join(base_path,i)
        denoise_testset = DenoiseTestDataset(testopt)
        denoise_tests.append(denoise_testset)

    print("CKPT name : {}".format(ckpt_path))
    net = PromptIRModel.load_from_checkpoint(
        ckpt_path,
        strict=False,
        use_diffusion=testopt.use_diffusion,
        diffusion_ckpt=testopt.diffusion_ckpt,
        diffusion_steps=testopt.diffusion_steps,
        num_experts=1  
    ).cuda()
    net.eval()
    net = torch.nn.DataParallel(net).cuda()
    if testopt.mode == 0:
        for testset,name in zip(denoise_tests,denoise_splits) :
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15)

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25)

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50)
    elif testopt.mode == 1:
        print('Start testing rain streak removal...')
        derain_base_path = testopt.derain_path
        for name in derain_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
            test_Derain_Dehaze(net, derain_set, task="derain")
    elif testopt.mode == 2:
        print('Start testing SOTS...')
        dehaze_set = DerainDehazeDataset(testopt, task="dehaze", addnoise=False, sigma=15)
        test_Derain_Dehaze(net, dehaze_set, task="dehaze")

    elif testopt.mode == 4:
        print('Start testing GoPro...')
        deblur_set = DeblurLowlightDataset(testopt, task="deblur", addnoise=False, sigma=15)
        test_Derain_Dehaze(net, deblur_set, task="deblur")
    elif testopt.mode == 5:
        print('Start testing LOL...')
        lowlight_set = DeblurLowlightDataset(testopt, task="lowlight", addnoise=False, sigma=15)
        test_Derain_Dehaze(net, lowlight_set, task="lowlight")

    elif testopt.mode == 3:
        for testset,name in zip(denoise_tests,denoise_splits) :
            print('Start {} testing Sigma=15...'.format(name))

            test_Denoise(net, testset, sigma=15)

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25)

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50)

        derain_base_path = testopt.derain_path
        print(derain_splits)
        for name in derain_splits:

            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path,name)
            derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
            test_Derain_Dehaze(net, derain_set, task="derain")

        print('Start testing SOTS...')
        test_Derain_Dehaze(net, derain_set, task="dehaze")

