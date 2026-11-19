

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=4,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['dehaze'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--deblur_dir', type=str, default='data/Train/Deblur/',
                    help='where training images of deblurring saves.')
parser.add_argument('--lowlight_dir', type=str, default='data/Train/LowLight/',
                    help='where training images of deblurring saves.')

parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default="promptir",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="train_MoeDiff260413_Haze_1D_ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default= 2,help = "Number of GPUs to use for training")
parser.add_argument('--use_diffusion', type=bool, default=True, help='使用潜在扩散模型进行图像恢复')

parser.add_argument('--diffusion_ckpt', type=str, default=r"/opt/data/private/bfs_files/PromptIR-main/ckpt/v1-5-pruned-emaonly.safetensors", help='潜在扩散模型权重路径')
parser.add_argument('--diffusion_steps', type=int, default=50, help='扩散模型采样步数')
options = parser.parse_args()
