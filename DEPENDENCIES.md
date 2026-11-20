# Dependency Notes

This list describes imports in the current code; it is not a tested lockfile.
Do not blindly combine the legacy upstream environment with a modern PyTorch
environment. Recover exact versions from the original training environment
when preparing a reproducible release.

Main restoration entry points and utilities use:

```text
torch
torchvision
lightning
numpy
Pillow
opencv-python
scipy
scikit-image
einops
omegaconf
safetensors
tqdm
wandb
tensorboard
lpips
pytorch-msssim
```

Expert-usage analysis additionally uses `matplotlib`. Optional vendored
diffusion modules also reference `pytorch-lightning`, `transformers`, `kornia`,
`albumentations`, and OpenAI CLIP, among other upstream dependencies. Their
original environment/configuration files are retained as references, not as
guarantees that all entry points install together unchanged.

`sd3_impls` is an unresolved local dependency in the current model imports;
it is not included and is not satisfied by this dependency list.
