



## PromptIR: Prompting for All-in-One Blind Image Restoration
## Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2306.13090

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
from einops.layers.torch import Rearrange
import time
from net.moe import MoE
import os
import sys
import torchvision.utils as vutils
sys.path.append(r"/opt/data/private/bfs_files/PromptIR-main/stable_diffusion")
sys.path.append("/opt/data/private/bfs_files/PromptIR-main/stable_diffusion/taming")


from stable_diffusion.ldm.models.diffusion.ddpm_edit import LatentDiffusion
from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler

sys.path.append("/opt/data/private/bfs_files/PromptIR-main/sd3.5-main")  
from sd3_impls import BaseModel, SDVAE, SD3LatentFormat
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors
from omegaconf import OmegaConf

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out
class resblock(nn.Module):
    def __init__(self, dim):
        super(resblock, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))
    def forward(self, x):
        res = self.body((x))
        res += x
        return res


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))
    def forward(self, x):
        return self.body(x)
class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
    def forward(self, x):
        return self.body(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
    def forward(self, x):
        x = self.proj(x)
        return x

class DetailBoost(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.3))  

    def forward(self, x):
        blur = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - blur
        return x + self.gain * high


class ColorCorrection(nn.Module):
    def __init__(self, in_channels=3, mid=32):
        super().__init__()

        self.global_gain = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Tanh()
        )

        self.local = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 3, padding=1)
        )

    def forward(self, x):
        g = self.global_gain(x)  
        local = self.local(x)  
        return x * (1 + g) + local


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid_channels = max(in_channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared(self.avg_pool(x))
        max_out = self.shared(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        inter_channels = max(in_channels // 8, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, inter_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, in_channels, x_channels=3):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels)
        self.spatial_attn = SpatialAttention(in_channels)
        self.spatial_refine = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.x_proj = nn.Conv2d(x_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, y, x_ref):
        y = self.channel_attn(y)
        y = self.spatial_attn(y)
        y_cbam = y + self.spatial_refine(y)

        x_down = F.interpolate(x_ref, size=y_cbam.shape[2:], mode='bilinear', align_corners=False)
        x_proj = self.x_proj(x_down)

        diff = y_cbam - x_proj
        mask = torch.sigmoid(-torch.abs(diff))

        y_clean = y_cbam + y_cbam * mask
        return y_clean


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class FeaturePurifier(nn.Module):
    """
    双分支净化：
    - 高频去雨/去噪
    - 低频去雾/去模糊/低光/色偏
    """

    def __init__(self, in_channels):
        super().__init__()

        self.gate_high = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

        self.gate_low = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1)
        )

    def forward(self, y):
        blur1 = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
        high = y - blur1

        blur2 = F.avg_pool2d(blur1, kernel_size=5, stride=1, padding=2)
        low = blur1 - blur2

        y_clean = y \
                  - self.gate_high(y) * high \
                  - self.gate_low(y) * low

        return self.refine(y_clean)


class CrossFuseConv(nn.Module):
    def __init__(self, x_channels, y_channels, out_channels):
        super().__init__()
        in_ch = x_channels + y_channels + x_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.highpass = nn.Conv2d(x_channels, x_channels,
                                  3, padding=1, groups=x_channels, bias=False)

    def forward(self, x, y):
        if y.size(2) != x.size(2):
            y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)

        x_down = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_up = F.interpolate(x_down, size=x.shape[2:], mode='bilinear', align_corners=False)
        diff = x - x_up
        diff_hp = self.highpass(diff)

        cat = torch.cat([x, y, diff_hp], dim=1)
        return self.fuse(cat)


class AdaptiveAlpha(nn.Module):
    def __init__(self, in_channels_x, in_channels_y):
        super().__init__()
        in_ch = in_channels_x + in_channels_y
        mid = max(in_ch // 4, 1)

        self.local = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
        )

        self.global_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x, y):
        if y.size(2) != x.size(2):
            y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)

        inp = torch.cat([x, y], dim=1)
        local_map = self.local(inp)
        global_map = self.global_fc(inp)
        global_map_up = F.interpolate(global_map, size=local_map.shape[2:], mode='nearest')
        return self.sig(local_map + global_map_up)


class DetailRefinementModule(nn.Module):
    def __init__(self, in_channels=3, mid_channels=64, y_channels=48, num_blocks=6):
        super().__init__()
        self.feature_purifier = FeaturePurifier(y_channels)
        self.cbam_on_y = CBAM(y_channels, x_channels=in_channels)
        self.cross_fuse = CrossFuseConv(in_channels, y_channels, mid_channels)
        self.blocks = nn.Sequential(*[ResidualBlock(mid_channels) for _ in range(num_blocks)])
        self.final = nn.Conv2d(mid_channels, in_channels, 3, padding=1)
        self.detail_boost = DetailBoost(in_channels)
        self.color_correct = ColorCorrection(in_channels)
        self.adaptive_alpha = AdaptiveAlpha(in_channels, y_channels)

    def forward(self, x, y):
        """
        x: SD 恢复图像 (B,3,H,W)
        y: 原图 transformer 一级特征 (B,C,h,w)
        """
        y_purified = self.feature_purifier(y)
        y_cbam = self.cbam_on_y(y_purified, x)
        fused = self.cross_fuse(x, y_cbam)
        refined = self.blocks(fused)
        refined = self.final(refined)
        refined = self.detail_boost(refined)
        refined = self.color_correct(refined)
        y_up = F.interpolate(y_cbam, size=x.shape[2:], mode='bilinear', align_corners=False)
        alpha = self.adaptive_alpha(x, y_up)

        return x + alpha * refined



class PromptPoolingFusion(nn.Module):
    def __init__(self, dim=768, num_heads=8, pooled_tokens=4):  
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, pooled_tokens, dim))  
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):  
        B = x.size(0)
        query = self.query.expand(B, -1, -1)  
        attended, _ = self.attn(query, x, x)  
        return self.norm(attended)  


class PromptIR(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  
                 decoder=False,
                 use_diffusion=False,  
                 diffusion_ckpt=None,  
                 diffusion_steps=50,  
                 num_experts=3  
                 ):

        super(PromptIR, self).__init__()

        self.prompt3_linear = nn.Linear(384, 768)
        self.prompt1_linear = nn.Linear(48, 768)
        self.prompt2_linear = nn.Linear(96, 768)
        self.prompt4_linear = nn.Linear(192, 768)
        self.prompt1_pool = PromptPoolingFusion(dim=768, num_heads=8, pooled_tokens=32)
        self.prompt2_pool = PromptPoolingFusion(dim=768, num_heads=8, pooled_tokens=16)
        self.prompt4_pool = PromptPoolingFusion(dim=768, num_heads=8, pooled_tokens=16)
        self.prompt3_pool = PromptPoolingFusion(dim=768, num_heads=8, pooled_tokens=8)
        self.final_fusion = PromptPoolingFusion(dim=768, num_heads=8, pooled_tokens=4)

        self.prompt_fusion = PromptPoolingFusion(dim=768, num_heads=8)
        self.reduce_conv = nn.Conv2d(8, 4, kernel_size=1)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder = decoder
        self.use_diffusion = use_diffusion
        self.diffusion_steps = diffusion_steps

        self.detail_refiner = DetailRefinementModule(in_channels=3, mid_channels=64, y_channels=48)
        if self.use_diffusion:
            try:
                self.load_diffusion_model(diffusion_ckpt)
            except Exception as e:
                print(f"加载扩散模型失败: {e}")
                self.use_diffusion = False
        if self.decoder:
            self.prompt1 = MoE(input_size=48, head_size=64, num_experts=num_experts,mlp_hidden_dim=128, k=1, activation=nn.GELU(),acc_aux_loss=True)
            self.prompt2 = MoE(input_size=96, head_size=64, num_experts=num_experts,mlp_hidden_dim=128, k=1, activation=nn.GELU(),acc_aux_loss=True)
            self.prompt4 = MoE(input_size=192, head_size=128, num_experts=num_experts,mlp_hidden_dim=128, k=1, activation=nn.GELU(),acc_aux_loss=True)
            self.prompt3 = MoE(input_size=384, head_size=320, num_experts=num_experts,mlp_hidden_dim=128, k=1, activation=nn.GELU(),acc_aux_loss=True)


        self.chnl_reduce1 = nn.Conv2d(64, 64, kernel_size=1, bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128, 128, kernel_size=1, bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320, 256, kernel_size=1, bias=bias)
        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64, dim, kernel_size=1, bias=bias)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)  
        self.reduce_noise_channel_2 = nn.Conv2d(int(dim * 2 ** 1) + 128, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim * 2 ** 1))  
        self.reduce_noise_channel_3 = nn.Conv2d(int(dim * 2 ** 2) + 256, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim * 2 ** 2))  
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

    def load_diffusion_model(self, ckpt_path=None, config_path=None):
        if config_path is None:
            config_path = r"/opt/data/private/bfs_files/PromptIR-main/stable_diffusion/configs/stable-diffusion/v1-inference.yaml"
        config = OmegaConf.load(config_path)
        if ckpt_path is None:
            ckpt_path = r"/opt/data/private/bfs_files/PromptIR-main/ckpt/v1-5-pruned-emaonly.safetensors"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"扩散模型权重文件不存在: {ckpt_path}")
        self.ldm = LatentDiffusion(
            **config.model.params
        )
        if ckpt_path.lower().endswith(".safetensors"):
            state_dict = load_safetensors(ckpt_path)
        else:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        self.ldm.load_state_dict(state_dict, strict=False)
        for param in self.ldm.model.parameters():  
            param.requires_grad = False
        for param in self.ldm.first_stage_model.encoder.parameters():
            param.requires_grad = True
        for param in self.ldm.first_stage_model.decoder.parameters():
            param.requires_grad = True
        self.ldm.eval()  
    def forward_diffusion_train(self, inp_img, cond=None):
        encoder = self.ldm.first_stage_model.encoder
        latent = encoder(inp_img)
        latent = self.reduce_conv(latent)
        if hasattr(latent, "sample"):
            latent = latent.sample()
        b = latent.shape[0]
        t = torch.randint(0, self.ldm.num_timesteps, (b,), device=latent.device).long()
        noise = torch.randn_like(latent)
        latent_noisy = self.ldm.q_sample(latent, t, noise)

        cond_for_unet = self.detach_cond(cond)
        pred = self.ldm.model(latent_noisy, t, cond_for_unet)
        decoder = self.ldm.first_stage_model.decoder
        restored = decoder(pred)
        restored = torch.clamp(restored, 0.0, 1.0)
        return restored
    def detach_cond(self,cond):
        if cond is None:
            return None
        elif isinstance(cond, torch.Tensor):
            return cond.detach()
        elif isinstance(cond, list):
            return [self.detach_cond(c) for c in cond]
        elif isinstance(cond, dict):
            return {k: self.detach_cond(v) if k != "c_crossattn" else v for k, v in cond.items()}
        else:
            raise TypeError(f"Unsupported cond type: {type(cond)}")

    def forward(self, inp_img, noise_emb=None, return_prompt_only=False):
        aux_loss_total = 0.0  
        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        prompt1_out, loss1 = self.prompt1(out_enc_level1)
        prompt1_out = self.prompt1_linear(prompt1_out)
        prompt1_pooled = self.prompt1_pool(prompt1_out)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        prompt2_out, loss2 = self.prompt2(out_enc_level2)
        prompt2_out = self.prompt2_linear(prompt2_out)
        prompt2_pooled = self.prompt2_pool(prompt2_out)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        prompt4_out, loss4 = self.prompt4(out_enc_level3)
        prompt4_out = self.prompt4_linear(prompt4_out)
        prompt4_pooled = self.prompt4_pool(prompt4_out)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        if self.decoder:
            B, C, H, W = latent.shape
            latent_flat = latent.view(B, C, -1).permute(0, 2, 1)  
            prompt3_out, loss3 = self.prompt3(latent_flat)  
            prompt3_out = self.prompt3_linear(prompt3_out)  
            prompt3_pooled = self.prompt3_pool(prompt3_out)

            prompt_all = torch.cat([prompt1_pooled, prompt2_pooled, prompt3_pooled, prompt4_pooled], dim=1)

            

            restored = self.forward_diffusion_train(inp_img, cond={"c_crossattn": [prompt_all]})


            restored = self.detail_refiner(restored,out_enc_level1)


            aux_loss_total = loss1 + loss2 + loss3 + loss4

        return restored,aux_loss_total
