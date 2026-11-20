





import argparse
import csv
import json
import re
from datetime import datetime
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
    def __init__(self, use_diff_ckpt=False,use_diffusion=True, diffusion_ckpt=None, diffusion_steps=50,num_experts=5,
                 validate_moe_weights=False):
        super().__init__()
        self.validate_moe_weights = validate_moe_weights
        self.net = PromptIR(
            decoder=True,
            use_diffusion=use_diffusion,
            diffusion_ckpt=diffusion_ckpt,
            diffusion_steps=diffusion_steps,
            num_experts=num_experts  
        )
        self.loss_fn = nn.L1Loss()

    def on_load_checkpoint(self, checkpoint):
        if not self.validate_moe_weights:
            return
        state = checkpoint["state_dict"]
        prefixes = tuple("net.{}.".format(name) for name in
                         ("prompt1", "prompt2", "prompt4", "prompt3"))
        expected = {name: param for name, param in self.named_parameters()
                    if name.startswith(prefixes)}
        missing = [name for name in expected if name not in state]
        mismatched = [name for name, param in expected.items()
                      if name in state and state[name].shape != param.shape]
        unexpected = [name for name in state if name.startswith(prefixes)
                      and name not in expected
                      and not name.endswith(".expert_call_count")]
        if missing or mismatched or unexpected:
            raise RuntimeError(
                "MoPE checkpoint mismatch. Use the same --num_experts as training. "
                "Missing: {}; shape mismatch: {}; unexpected: {}".format(
                    missing[:5], mismatched[:5], unexpected[:5]))

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

class ExpertUsageRecorder:
    BRANCHES = (("scale1", "prompt1", 1), ("scale2", "prompt2", 2),
                ("scale3", "prompt4", 4), ("scale4", "prompt3", 8))

    def __init__(self, net, root, task, dataset_path, checkpoint_path, sigma=None, audit_images=3):
        if isinstance(net, nn.DataParallel):
            raise ValueError("Expert usage collection requires single-GPU testing.")
        self.model = net.net
        self.task = task
        self.sigma = sigma
        self.num_experts = self.model.prompt1.num_experts
        if audit_images < 0:
            raise ValueError("audit_images must be nonnegative")
        self.audit_images = audit_images
        self._audited_images = 0
        for _, branch, _ in self.BRANCHES:
            module = getattr(self.model, branch)
            if module.k != 1 or module.num_experts != self.num_experts:
                raise ValueError("Expected Top-1 routing with equal expert counts at all scales.")

        dataset_name = os.path.basename(os.path.normpath(dataset_path)) or task
        self.dataset_name = dataset_name
        safe_name = re.sub(r"[^\w.-]+", "_", dataset_name)
        parts = [root, task, safe_name]
        if sigma is not None:
            parts.append("sigma_{}".format(sigma))
        parts.append(datetime.utcnow().strftime("run_%Y%m%dT%H%M%S_%fZ"))
        self.save_dir = os.path.join(*parts)
        os.makedirs(self.save_dir, exist_ok=False)
        self.per_image_path = os.path.join(self.save_dir, "per_image.csv")
        self.summary_path = os.path.join(self.save_dir, "summary.csv")
        self.metadata_path = os.path.join(self.save_dir, "metadata.json")
        self.audit_dir = os.path.join(self.save_dir, "routing_audit")
        self.audit_log_path = os.path.join(self.save_dir, "routing_audit.jsonl")
        if self.audit_images:
            os.makedirs(self.audit_dir)
        self.ratios = {scale: [] for scale, _, _ in self.BRANCHES}
        self.counts = {scale: [] for scale, _, _ in self.BRANCHES}
        self._feature_shapes = {}
        self._routing = {}
        self._handles = []
        self._original_gating = {}
        self._reported = set()
        self.fields = ["image_name", "degradation", "dataset", "sigma", "scale", "branch",
                       "input_height", "input_width", "padded_height", "padded_width",
                       "feature_height", "feature_width",
                       "valid_token_count", "feature_token_count", "excluded_padding_tokens"]
        self.fields += ["expert{}_count".format(e + 1) for e in range(self.num_experts)]
        self.fields += ["expert{}_ratio".format(e + 1) for e in range(self.num_experts)]
        with open(self.per_image_path, "w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.fields).writeheader()
        self.metadata = {
            "status": "running", "degradation": task, "dataset": dataset_name,
            "dataset_path": os.path.abspath(dataset_path), "sigma": sigma,
            "checkpoint_path": os.path.abspath(checkpoint_path),
            "num_experts": self.num_experts, "top_k": 1,
            "expert_labels": "expert1 corresponds to zero-based routing index 0",
            "scales": {scale: {"branch": branch, "nominal_stride": stride}
                       for scale, branch, stride in self.BRANCHES},
            "routing_observer_version": 3,
            "routing_audit_images": self.audit_images,
            "routing_audit_temperature": 2.0,
            "routing_audit": "Same-forward gate logits, returned Top-1 indices, dispatch sizes and token order; no second gate call",
            "counting": "Actual Top-1 indices captured from top_k_gating during forward; padded positions excluded",
            "valid_region": "top-left floor(input_height*map_height/padded_height) x floor(input_width*map_width/padded_width)",
            "aggregation": "arithmetic mean of per-image expert ratios, not pooled token counts",
            "seed": 0,
        }
        self._save_metadata()
        self._attach_routing_observers()
        print("[Expert usage] Saving to {}".format(self.save_dir))

    def _attach_routing_observers(self):
        stages = {"prompt1": "encoder_level1", "prompt2": "encoder_level2",
                  "prompt4": "encoder_level3", "prompt3": "latent"}
        for _, branch, _ in self.BRANCHES:
            stage = getattr(self.model, stages[branch], None)
            if stage is not None:
                def capture_feature(module, inputs, output, branch=branch):
                    if isinstance(output, torch.Tensor) and output.dim() == 4:
                        self._feature_shapes[branch] = tuple(output.shape)
                self._handles.append(stage.register_forward_hook(capture_feature))

            moe = getattr(self.model, branch)
            def capture_input(module, inputs, branch=branch):
                feature = inputs[0]
                if feature.dim() == 4:
                    self._feature_shapes[branch] = tuple(feature.shape)
                self._routing[branch] = {"input_shape": tuple(feature.shape)}
            self._handles.append(moe.register_forward_pre_hook(capture_input))

            def capture_logits(module, inputs, output, branch=branch):
                if self._audited_images < self.audit_images:
                    logits = output[0] if isinstance(output, (tuple, list)) else output
                    self._routing[branch]["audit_logits"] = logits.detach().clone()
            self._handles.append(moe.f_gate.register_forward_hook(capture_logits))
            for expert_stage in ("experts", "output_experts"):
                def capture_dispatch(module, inputs, branch=branch, expert_stage=expert_stage):
                    if self._audited_images < self.audit_images:
                        self._routing[branch][expert_stage + "_sizes"] = inputs[1].detach().clone()
                self._handles.append(getattr(moe, expert_stage).register_forward_pre_hook(capture_dispatch))

            original = moe.top_k_gating
            self._original_gating[branch] = original
            def capture_gating(*args, _original=original, _branch=branch, _moe=moe, **kwargs):
                result = _original(*args, **kwargs)
                if (not isinstance(result, (tuple, list)) or len(result) < 2
                        or not isinstance(result[1], torch.Tensor)):
                    raise RuntimeError(
                        "{}: top_k_gating must return (loss, top_k_indices); got {}. "
                        "Check the server's net/moe.py implementation.".format(_branch, type(result)))
                indices = result[1].detach().clone()
                if indices.dim() == 2 and indices.shape[1] == 1:
                    indices = indices[:, 0]
                if indices.dim() != 1 or indices.is_floating_point():
                    raise RuntimeError("{}: unexpected Top-1 index shape/dtype: {} / {}".format(
                        _branch, tuple(indices.shape), indices.dtype))
                self._routing[_branch]["indices"] = indices
                if self._audited_images < self.audit_images:
                    observation = self._routing[_branch]
                    observation["expert_size"] = _moe.expert_size.detach().clone()
                    observation["batch_index"] = _moe.batch_index.detach().clone()
                    skip_mask = kwargs.get("skip_mask", args[1] if len(args) > 1 else None)
                    observation["skip_mask"] = None if skip_mask is None else skip_mask.detach().clone()
                return result
            moe.top_k_gating = capture_gating

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for branch, original in self._original_gating.items():
            getattr(self.model, branch).top_k_gating = original
        self._original_gating.clear()

    def begin_batch(self):
        self._feature_shapes.clear()
        self._routing.clear()

    def _save_metadata(self):
        with open(self.metadata_path, "w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, indent=2, ensure_ascii=True)

    def _audit_batch(self, branch, indices):
        observation = self._routing[branch]
        logits = observation.get("audit_logits")
        if logits is None or tuple(logits.shape) != (indices.numel(), self.num_experts):
            raise RuntimeError("{}: missing or unexpected same-forward gate logits".format(branch))
        if not torch.isfinite(logits).all().item():
            raise RuntimeError("{}: gate logits contain NaN/Inf; Top-1 counts are not reliable".format(branch))
        probabilities = torch.softmax(logits / 2.0, dim=1)
        if observation["skip_mask"] is not None:
            probabilities = probabilities.masked_fill(observation["skip_mask"], 0)
        maxima = probabilities.max(dim=1).values
        chosen = probabilities.gather(1, indices[:, None]).squeeze(1)
        if (maxima <= 0).any().item() or not torch.equal(chosen, maxima):
            raise RuntimeError("{}: returned Top-1 indices do not select maximal gate probabilities".format(branch))

        full_counts = torch.bincount(indices, minlength=self.num_experts)
        sizes = observation["expert_size"].to(dtype=torch.long)
        if not torch.equal(full_counts, sizes):
            raise RuntimeError("{}: raw Top-1 histogram disagrees with actual dispatch sizes".format(branch))
        for stage in ("experts", "output_experts"):
            actual = observation.get(stage + "_sizes")
            if actual is None or not torch.equal(actual.to(dtype=torch.long), sizes):
                raise RuntimeError("{}: {} did not receive the expected dispatch sizes".format(branch, stage))
        token_order = observation["batch_index"].to(dtype=torch.long)
        if (token_order.numel() != indices.numel() or (token_order < 0).any().item()
                or (token_order >= indices.numel()).any().item()
                or not torch.all(torch.bincount(token_order, minlength=indices.numel()) == 1).item()):
            raise RuntimeError("{}: Top-1 dispatch does not visit every token exactly once".format(branch))
        dispatched = torch.empty_like(indices)
        dispatched[token_order] = torch.repeat_interleave(
            torch.arange(self.num_experts, device=indices.device), sizes)
        if not torch.equal(dispatched, indices):
            raise RuntimeError("{}: reconstructed expert dispatch disagrees with returned Top-1 indices".format(branch))
        return logits, probabilities, dispatched

    def _save_image_audit(self, name, scale, branch, image_index, map_h, map_w,
                          valid_h, valid_w, assignments, audit, counts):
        logits, probabilities, dispatched = audit
        raw = assignments[image_index]
        dispatch_map = dispatched.reshape(-1, map_h, map_w)[image_index]
        logits_map = logits.reshape(-1, map_h, map_w, self.num_experts)[image_index]
        probs_map = probabilities.reshape(-1, map_h, map_w, self.num_experts)[image_index]
        valid_probs = probs_map[:valid_h, :valid_w].reshape(-1, self.num_experts)
        raw_counts = torch.bincount(raw.reshape(-1), minlength=self.num_experts).cpu().numpy()
        mask = torch.zeros_like(raw, dtype=torch.bool)
        mask[:valid_h, :valid_w] = True
        mask_counts = torch.bincount(raw[mask], minlength=self.num_experts).cpu().numpy()
        if not np.array_equal(mask_counts, counts):
            raise RuntimeError("{}: audit mask counts disagree with CSV counts".format(branch))
        peak = valid_probs.max(dim=1).values
        tied = ((valid_probs == peak[:, None]).sum(dim=1) > 1).sum().item()
        entropy_probs = valid_probs.float()
        entropy = -(entropy_probs * entropy_probs.clamp_min(1e-12).log()).sum(dim=1).mean().item()
        margin = None
        if self.num_experts > 1:
            top_two = valid_probs.topk(2, dim=1).values
            margin = (top_two[:, 0] - top_two[:, 1]).mean().item()
        filename = "image_{:06d}_{}.npz".format(self._audited_images + image_index, branch)
        np.savez_compressed(os.path.join(self.audit_dir, filename),
                            top1_indices=raw.cpu().numpy(), dispatch_indices=dispatch_map.cpu().numpy(),
                            valid_mask=mask.cpu().numpy(), logits=logits_map.float().cpu().numpy(),
                            probabilities=probs_map.float().cpu().numpy())
        record = {
            "image_name": name, "scale": scale, "branch": branch,
            "snapshot": os.path.join("routing_audit", filename), "indices_are_zero_based": True,
            "feature_shape": [map_h, map_w], "valid_shape": [valid_h, valid_w],
            "raw_top1_counts_including_padding": raw_counts.tolist(),
            "valid_top1_counts": counts.tolist(), "padding_top1_counts": (raw_counts - counts).tolist(),
            "dispatch_matches_top1": True, "csv_matches_mask_counts": True,
            "selected_probabilities_are_maximal": True,
            "valid_mean_soft_probabilities": valid_probs.mean(dim=0).cpu().tolist(),
            "valid_mean_top1_probability": peak.mean().item(), "valid_mean_top1_top2_margin": margin,
            "valid_mean_soft_entropy": entropy, "valid_tied_max_tokens": int(tied),
        }
        with open(self.audit_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
        print("[Routing audit] {} {}: raw={}, valid={}, mean_soft_probs={}, margin={}".format(
            name, branch, raw_counts.tolist(), counts.tolist(),
            [round(value, 6) for value in record["valid_mean_soft_probabilities"]], margin))

    def record(self, names, original_sizes, padded_size):
        if self.model.training:
            raise RuntimeError("Call eval() before collecting expert usage.")
        if len(names) != len(original_sizes):
            raise ValueError("Image names and sizes must match the current batch.")
        padded_h, padded_w = padded_size
        rows = []
        for scale, branch, stride in self.BRANCHES:
            observation = self._routing.get(branch, {})
            indices = observation.get("indices")
            feature_shape = self._feature_shapes.get(branch)
            if indices is None or feature_shape is None:
                raise RuntimeError(
                    "{}: routing was not captured. MoPE input={}, encoder feature={}, "
                    "Top-1 indices present={}. Verify that model.forward calls this MoPE branch.".format(
                        branch, observation.get("input_shape"), feature_shape, indices is not None))
            map_h, map_w = feature_shape[-2:]
            expected_tokens = len(names) * map_h * map_w
            if feature_shape[0] != len(names) or indices.numel() != expected_tokens:
                raise RuntimeError(
                    "{}: token-wise routing shape mismatch. MoPE input={}, encoder feature={}, "
                    "actual routed tokens={}, expected tokens={}, padded image={}. "
                    "Check for pooling/resizing inside the server's MoPE implementation.".format(
                        branch, observation.get("input_shape"), feature_shape,
                        indices.numel(), expected_tokens, padded_size))
            if indices.numel() and (indices.min().item() < 0
                                   or indices.max().item() >= self.num_experts):
                raise RuntimeError("{}: expert indices are outside [0, {}).".format(branch, self.num_experts))
            assignments = indices.reshape(len(names), map_h, map_w)
            audit = self._audit_batch(branch, indices) if self._audited_images < self.audit_images else None
            if branch not in self._reported:
                print("[Expert usage] {}: input={}, feature_map={}x{}, routed_tokens={}".format(
                    branch, observation.get("input_shape"), map_h, map_w, indices.numel()))
                self._reported.add(branch)
            for index, (name, size) in enumerate(zip(names, original_sizes)):
                height, width = size
                valid_h, valid_w = height * map_h // padded_h, width * map_w // padded_w
                if not (0 < valid_h <= map_h and 0 < valid_w <= map_w):
                    raise ValueError("Invalid original image size for {}.".format(branch))
                selected = assignments[index, :valid_h, :valid_w].reshape(-1)
                counts = torch.bincount(selected, minlength=self.num_experts).cpu().numpy()
                ratios = counts.astype(np.float64) / selected.numel()
                if audit is not None and self._audited_images + index < self.audit_images:
                    self._save_image_audit(name, scale, branch, index, map_h, map_w,
                                           valid_h, valid_w, assignments, audit, counts)
                row = {
                    "image_name": name, "degradation": self.task, "dataset": self.dataset_name,
                    "sigma": self.sigma if self.sigma is not None else "",
                    "scale": scale, "branch": branch,
                    "input_height": height, "input_width": width,
                    "padded_height": padded_h, "padded_width": padded_w,
                    "feature_height": map_h, "feature_width": map_w,
                    "valid_token_count": selected.numel(), "feature_token_count": map_h * map_w,
                    "excluded_padding_tokens": map_h * map_w - selected.numel(),
                }
                for expert in range(self.num_experts):
                    row["expert{}_count".format(expert + 1)] = int(counts[expert])
                    row["expert{}_ratio".format(expert + 1)] = float(ratios[expert])
                rows.append(row)
                self.counts[scale].append(counts)
                self.ratios[scale].append(ratios)
        with open(self.per_image_path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.fields).writerows(rows)
        self._audited_images = min(self.audit_images, self._audited_images + len(names))

    def finalize(self):
        fields = ["degradation", "dataset", "sigma", "scale", "branch", "expert",
                  "num_images", "total_tokens", "expert_token_count", "mean_ratio",
                  "std_ratio", "images_using_expert", "image_usage_fraction"]
        with open(self.summary_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for scale, branch, _ in self.BRANCHES:
                if not self.ratios[scale]:
                    continue
                ratios = np.stack(self.ratios[scale])
                counts = np.stack(self.counts[scale])
                mean = ratios.mean(axis=0)
                std = ratios.std(axis=0, ddof=1 if len(ratios) > 1 else 0)
                for expert in range(self.num_experts):
                    active = int((counts[:, expert] > 0).sum())
                    writer.writerow(dict(zip(fields, [
                        self.task, self.dataset_name,
                        self.sigma if self.sigma is not None else "", scale, branch,
                        expert + 1, len(ratios), int(counts.sum()),
                        int(counts[:, expert].sum()), float(mean[expert]), float(std[expert]),
                        active, float(active / len(ratios))])))
        self.metadata.update(status="complete", num_images=len(self.ratios["scale1"]))
        self._save_metadata()
        self.close()
        print("[Expert usage] Summary saved to {}".format(self.summary_path))


def create_expert_usage_recorder(net, dataset, task, sigma=None):
    if not testopt.save_expert_usage:
        return None
    root = testopt.expert_usage_dir or os.path.join(testopt.output_path, "expert_usage")
    dataset_path = getattr(dataset.args, task + "_path")
    return ExpertUsageRecorder(net, root, task, dataset_path, testopt.ckpt_path, sigma,
                               audit_images=testopt.expert_audit_images)


def test_Denoise(net, dataset, sigma=15):
    output_path = os.path.join(testopt.output_path, 'denoise', str(sigma))
    os.makedirs(output_path, exist_ok=True)

    dataset.set_sigma(sigma)
    testloader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True,
        collate_fn=custom_collate_fn
    )

    psnr = AverageMeter()
    ssim = AverageMeter()

    expert_usage = create_expert_usage_recorder(net, dataset, "denoise", sigma)

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

            if expert_usage is not None:
                expert_usage.begin_batch()
            restored_batch, _ = net(degrad_batch)

            if expert_usage is not None:
                expert_usage.record(clean_names, [tuple(x.shape[-2:]) for x in degrad_patches],
                                    tuple(degrad_batch.shape[-2:]))

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
                save_image_tensor(restored, os.path.join(output_path, clean_names[i] + '.png'))

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))
    if expert_usage is not None:
        expert_usage.finalize()

def test_Derain_Dehaze(net, dataset, task="derain"):
    output_path = os.path.join(testopt.output_path, task)
    os.makedirs(output_path, exist_ok=True)

    dataset.set_dataset(task)
    testloader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True,
        collate_fn=custom_collate_fn
    )
    psnr = AverageMeter()
    ssim = AverageMeter()

    expert_usage = create_expert_usage_recorder(net, dataset, task)

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

            if expert_usage is not None:
                expert_usage.begin_batch()
            restored_batch, _ = net(degrad_batch)

            if expert_usage is not None:
                expert_usage.record(clean_names, [tuple(x.shape[-2:]) for x in degrad_patches],
                                    tuple(degrad_batch.shape[-2:]))
            for i in range(len(restored_batch)):
                pad_h, pad_w = pad_hw_list[i]
                h, w = clean_patches[i].shape[-2:]
                restored = restored_batch[i][..., :h, :w]
                clean = clean_patches[i][..., :h, :w]

                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean)
                psnr.update(temp_psnr, N)
                ssim.update(temp_ssim, N)
                print(f"{clean_names[i]} - PSNR: {temp_psnr:.2f}, SSIM: {temp_ssim:.4f}")  
                save_image_tensor(restored, os.path.join(output_path, clean_names[i] + '.png'))

        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
    if expert_usage is not None:
        expert_usage.finalize()

if __name__ == '__main__':

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0,
                        help='0 denoise, 1 derain, 2 dehaze, 3 all five tasks, 4 deblur, 5 lowlight')

    parser.add_argument('--denoise_path', type=str, default="data/test/denoise/", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="data/test/derain/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="data/test/dehaze/", help='save path of test hazy images')
    parser.add_argument('--deblur_path', type=str, default="data/test/deblur/", help='save path of test blur images')
    parser.add_argument('--lowlight_path', type=str, default="data/test/lowlight/", help='save path of test lowlight images')

    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="epoch=163-step=2755680.ckpt", help='checkpoint save path')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Full restoration checkpoint path')
    parser.add_argument('--num_experts', type=int, default=5, help='Must match the trained checkpoint')
    parser.add_argument('--save_expert_usage', action='store_true',
                        help='Save per-image and per-degradation expert usage on a single GPU')
    parser.add_argument('--expert_usage_dir', type=str, default=None,
                        help='Statistics root; defaults to output_path/expert_usage')
    parser.add_argument('--expert_audit_images', type=int, default=3,
                        help='Cross-check routing and save raw maps/logits for the first N images per dataset; 0 disables')
    parser.add_argument('--use_diffusion', type=bool, default=True, help='使用潜在扩散模型进行图像恢复')
    parser.add_argument('--diffusion_ckpt', type=str, default=r"/opt/data/private/bfs_files/PromptIR-main/ckpt/v1-5-pruned-emaonly.safetensors", help='潜在扩散模型权重路径')
    parser.add_argument('--diffusion_steps', type=int, default=30, help='扩散模型采样步数')
    parser.add_argument('--use_diff_ckpt', type=bool, default=False, help='使用潜在扩散模型进行图像恢复')
    testopt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    if testopt.num_experts < 1:
        parser.error('--num_experts must be positive')
    if testopt.expert_audit_images < 0:
        parser.error('--expert_audit_images must be nonnegative')
    if testopt.save_expert_usage:
        torch.cuda.set_device(testopt.cuda)
    ckpt_path = testopt.ckpt_path or ("train_MoeDiff260519_5D_ckpt/" + testopt.ckpt_name)
    testopt.ckpt_path = os.path.abspath(ckpt_path)

    denoise_splits = ["BSD68/"]
    derain_splits = ["Rain100L/"]

    denoise_tests = []
    derain_tests = []

    if testopt.mode in (0, 3):
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
        num_experts=testopt.num_experts,
        validate_moe_weights=testopt.save_expert_usage,
    ).cuda()
    net.eval()
    if not testopt.save_expert_usage:
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
        dehaze_set = DerainDehazeDataset(testopt, task="dehaze", addnoise=False, sigma=15)
        test_Derain_Dehaze(net, dehaze_set, task="dehaze")

        print('Start testing GoPro...')
        deblur_set = DeblurLowlightDataset(testopt, task="deblur", addnoise=False, sigma=15)
        test_Derain_Dehaze(net, deblur_set, task="deblur")

        print('Start testing LOL...')
        lowlight_set = DeblurLowlightDataset(testopt, task="lowlight", addnoise=False, sigma=15)
        test_Derain_Dehaze(net, lowlight_set, task="lowlight")

