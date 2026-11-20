# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from .parallel_experts import ParallelExperts

def log_moe_statistics(moe: 'MoE', name: str = "MoE"):
    if not moe.acc_aux_loss:
        return

    gate_counts = moe.acc_gates.detach().cpu().numpy()
    usage_freqs = moe.acc_freq.detach().cpu().numpy()

    print(f"[MoE] Expert Gating Sum: {gate_counts}")
    print(f"[MoE] Expert Usage Frequency (count > 0): {usage_freqs}")
    print(f"[MoE] CV² (均衡性): {moe.compute_cvloss(moe.acc_gates.unsqueeze(0)):.4f}")
    print(f"[MoE] Switch Loss (选择一致性): {moe.compute_switchloss(moe.acc_probs.unsqueeze(0), moe.acc_freq):.4f}")


class MultiHeadGating(nn.Module):
    def __init__(self, input_dim, num_experts, num_heads, dropout=0.0):
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim 必须能被 num_heads 整除"
        self.num_experts = num_experts
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.to_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_dim,  self.num_experts),  
                nn.Dropout(dropout)
            )
            for _ in range(num_heads)
        ])

        for gate_layer in self.to_gates:
            nn.init.normal_(gate_layer[0].weight, std=0.1)
            if gate_layer[0].bias is not None:
                nn.init.zeros_(gate_layer[0].bias)

    def forward(self, x):
        heads = x.reshape(x.shape[0], self.num_heads, self.head_dim)  
        all_logits = []
        for h in range(self.num_heads):
            logits_h = self.to_gates[h](heads[:, h, :])  
            all_logits.append(logits_h.unsqueeze(0))
        all_logits = torch.cat(all_logits, dim=0).transpose(0,1)  
        logits = all_logits.mean(dim=1)  
        return logits, all_logits  

class MoE(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, head_size, num_experts, k,mlp_hidden_dim,
                 cvloss=0, switchloss=0, zloss=0,
                 bias=False, gating_activation=None,
                 activation=None, noisy_gating=True, usage_mem=10000,
                 acc_aux_loss=False):
        super(MoE, self).__init__()

        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = input_size
        self.head_size = head_size
        self.bias = bias
        self.experts = ParallelExperts(num_experts, input_size, mlp_hidden_dim, head_size)
        self.output_experts = ParallelExperts(num_experts, head_size, mlp_hidden_dim, input_size)
        self.k = min(k, self.num_experts)
        self.cvloss = cvloss
        self.switchloss = switchloss
        self.zloss = zloss
        self.activation = activation  
        self.gate_input_dropout = nn.Dropout(p=0.1)
        self.switch_loss_weight = 0.1  
        self.register_buffer("expert_call_count", torch.zeros(self.num_experts, dtype=torch.long))
        self.acc_aux_loss = acc_aux_loss
        if self.acc_aux_loss:
            self.init_aux_statistics()
        if True:
            self.f_gate = MultiHeadGating(input_dim=input_size,
                                          num_experts=num_experts,
                                          num_heads=4,  
                                          dropout=0.1)
        else:
            self.f_gate = nn.Linear(input_size, num_experts, bias=False)
            nn.init.zeros_(self.f_gate.weight)

    @torch.jit.script
    def compute_gating(k: int, probs: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
        zeros = torch.zeros_like(probs)  
        gates = zeros.scatter(1, top_k_indices, top_k_gates)  
        top_k_gates = top_k_gates.flatten()
        top_k_experts = top_k_indices.flatten()
        nonzeros = top_k_gates.nonzero().squeeze(-1)
        top_k_experts_nonzero = top_k_experts[nonzeros]
        _, _index_sorted_experts = top_k_experts_nonzero.sort(0)
        expert_size = (gates > 0).long().sum(0)  
        index_sorted_experts = nonzeros[_index_sorted_experts]
        batch_index = index_sorted_experts.div(k, rounding_mode='trunc')  
        batch_gates = top_k_gates[index_sorted_experts]
        return batch_gates, batch_index, expert_size, gates, index_sorted_experts
    def extra_repr(self):
        return 'k={}, cvloss={}, switchloss={}, zloss={}, noisy_gating={}'.format(
            self.k, self.cvloss, self.switchloss, self.zloss, self.noisy_gating)
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return 0
        return x.float().var() / (x.float().mean() ** 2 + eps)
    def init_aux_statistics(self):
        self.acc_gates = torch.zeros(self.num_experts, dtype=torch.float32)  
        self.acc_probs = torch.zeros(self.num_experts, dtype=torch.float32)
        self.acc_freq = torch.zeros(self.num_experts, dtype=torch.float32)
        self.acc_lsesq = torch.tensor(0., dtype=torch.float32)
        self.acc_lsesq_count = torch.tensor(0., dtype=torch.float32)

    def update_aux_statistics(self, logits, probs, gates):
        logits_cpu = logits.detach().to("cpu")
        probs_cpu = probs.detach().to("cpu")
        gates_cpu = gates.detach().to("cpu")

        lsesq = torch.log(torch.exp(logits_cpu).sum(dim=1) + 1e-6) ** 2

        self.acc_probs += probs_cpu.sum(0)
        self.acc_gates += gates_cpu.sum(0)
        self.acc_freq += (gates_cpu > 0).float().sum(0)
        self.acc_lsesq += lsesq.sum()
        self.acc_lsesq_count += lsesq.size(0)

    def get_aux_loss_and_clear(self):
        cvloss = self.cv_squared(F.normalize(self.acc_gates, p=1, dim=0))
        switchloss = (F.normalize(self.acc_probs, p=1, dim=0) *
                      F.normalize(self.acc_freq, p=1, dim=0)).sum() * self.num_experts
        zloss = self.acc_lsesq / (self.acc_lsesq_count)
        loss = (self.cvloss * cvloss +
                self.switchloss * switchloss +
                self.zloss * zloss)
        self.init_aux_statistics()
        return loss
    def compute_cvloss(self, probs):
        return self.cv_squared(F.normalize(probs.sum(0), p=1, dim=0))
    def compute_switchloss(self, probs, freqs):
        loss = F.normalize(probs.sum(0), p=1, dim=0) * \
               F.normalize(freqs.float(), p=1, dim=0)
        return loss.sum() * self.num_experts
    def compute_zloss(self, logits):
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)
        return zloss
    def top_k_gating(self, x, skip_mask=None, sample_topk=0, noise_epsilon=1e-2):
        x = self.gate_input_dropout(x)
        logits, head_logits = self.f_gate(x)
        self.latest_logits = logits  
        temperature = 2.0
        probs = torch.softmax(logits / temperature, dim=1)
        if skip_mask is not None:
            probs = probs.masked_fill(skip_mask, 0)
        if self.training:
            uniform_noise = torch.rand_like(probs) * 1e-4
            probs = probs + uniform_noise
            probs = probs / probs.sum(dim=1, keepdim=True)
        top_k_gates, top_k_indices = probs.topk(self.k, dim=1)
        (batch_gates, batch_index,
         expert_size, gates,
         index_sorted_experts) = self.compute_gating(self.k, probs,
                                                     top_k_gates, top_k_indices)
        self.expert_size = expert_size
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates
        loss = 0.
        if self.acc_aux_loss:
            self.update_aux_statistics(logits, probs, gates)
            loss = self.compute_aux_loss(gates)
        return loss, top_k_indices

    def compute_aux_loss(self, gates: torch.Tensor):
        """
        计算 MoE 的辅助损失，包括 Z-Loss（对 logits 的正则）和 Switch Loss（门控一致性）
        gates: [B, E]，softmax 后门控输出
        self.latest_logits: [B, E]，softmax 前的 logits，需在 forward 中保存（不detach）
        """
        eps = 1e-6
        z_loss_weight = 0.5
        switch_loss_weight = 0.5

        if hasattr(self, 'latest_logits') and self.latest_logits is not None:
            logits = self.latest_logits  
            lse = torch.logsumexp(logits, dim=1)  
            lse = torch.clamp(lse, min=-1e2, max=50.0)  
            z_loss = torch.mean(lse ** 2)
        else:
            print("⚠️ [Z-Loss] 未找到 logits，跳过 Z-Loss")
            z_loss = torch.tensor(0.0, device=gates.device)

        if torch.isnan(z_loss):
            print("❌ Z-Loss is NaN！请检查 logits 输入是否为 inf 或 nan。")
        probs = gates  
        freq = probs.sum(dim=0) + eps  
        freq = freq / freq.sum()  
        mean_probs = probs.mean(dim=0) + eps
        switch_loss = F.kl_div(mean_probs.log(), freq, reduction="batchmean")
        if torch.isnan(switch_loss):
            print("❌ Switch Loss is NaN！请检查 gates 输入是否为 nan 或 0")
        aux_loss = z_loss_weight * z_loss + switch_loss_weight * switch_loss
        return aux_loss
    def forward(self, x, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        if x.dim() == 4:
            B, C, H, W = x.size()
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        elif x.dim() == 3:
            B, L, C = x.size()
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        bsz, length, emb_size = x.size()
        x_flat = x.reshape(-1, emb_size)


        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)

        loss, topk_indices = self.top_k_gating(x_flat, skip_mask, sample_topk=sample_topk)
        if self.training and hasattr(self, "expert_call_count"):
            with torch.no_grad():
                flat_indices = topk_indices.reshape(-1)
                for i in range(self.num_experts):
                    self.expert_call_count[i] += (flat_indices == i).sum()

        if self.expert_size.sum() == 0:
            return x, loss

        expert_inputs = x_flat[self.batch_index]


        h = self.experts(expert_inputs, self.expert_size)

        if h.numel() > 0:  
            h = self.activation(h)

        expert_outputs = self.output_experts(h, self.expert_size)

        if multiply_by_gates and expert_outputs.numel() > 0:
            expert_outputs = expert_outputs * self.batch_gates[:, None]
        zeros = torch.zeros((bsz * length, self.input_size),
                            dtype=expert_outputs.dtype, device=expert_outputs.device)

        if expert_outputs.numel() > 0:
            y = zeros.index_add(0, self.batch_index, expert_outputs)
        else:
            y = zeros

        y = y.view(bsz, length, self.input_size)
        return y, loss





    def forward_(self, x, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)

        logits = self.f_gate(x)
        probs = torch.softmax(logits, dim=1)

        loss = self.top_k_gating(x, skip_mask, sample_topk=sample_topk)
        assert not self.bias
        hs = [torch.einsum('li,ij->lj', x, self.experts.w[i]) for i in range(self.num_experts)]
        hs = [self.activation(h) for h in hs]
        expert_outputs = [
            torch.einsum('li,ij->lj', hs[i], self.output_experts.w[i]) for i in range(self.num_experts)
        ]
        y = sum(probs[..., i][..., None] * expert_outputs[i] for i in range(self.num_experts))
        y = y.view(bsz, length, self.input_size)

        return y

    def map(self, x, skip_mask=None, sample_topk=0):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.top_k_gating(x, skip_mask, sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        expert_outputs = self.experts(expert_inputs, self.expert_size)

        zeros = torch.zeros((bsz * length * self.k, self.head_size),
                            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.k, -1)
        return y, loss

    def reduce(self, x, multiply_by_gates=True):
        bsz, length, k, emb_size = x.size()
        x = x.view(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_experts(expert_inputs, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size),
                            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        return y



class TaskMoE(MoE):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, head_size, num_experts, k, w_MI=0, limit_k=0, w_topk_loss=0.0, task_num=9,
                 noisy_gating=True, gating_activation=None, **kwargs):
        self.task_num = task_num
        self.w_topk_loss = w_topk_loss
        self.w_MI = w_MI  

        self.limit_k = max(k, limit_k)  
        super(TaskMoE, self).__init__(input_size, head_size, num_experts, k, noisy_gating=noisy_gating,
                                      gating_activation=gating_activation, **kwargs)

        if gating_activation is None:
            gating_activation = nn.ReLU()

        self.f_gate = nn.ModuleList([nn.Sequential(
            nn.Linear(input_size,
                      2 * num_experts if noisy_gating else num_experts,
                      bias=False)
        ) for i in range(task_num)])
        for i in range(task_num):
            nn.init.zeros_(self.f_gate[i][-1].weight)

    def top_k_gating(self, x, task_bh, bsz, skip_mask=None, sample_topk=0, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        B = bsz
        index = x.size(0) // B
        result = []
        for i in range(B):
            clean_logits = self.f_gate[task_bh[i]](x[i * index:(i + 1) * index, :])  
            result.append(clean_logits)
        clean_logits = torch.cat(result, dim=0)
        if self.noisy_gating:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        elif self.noisy_gating:
            logits, _ = clean_logits.chunk(2, dim=-1)
        else:
            logits = clean_logits

        probs = torch.softmax(logits, dim=1) + 1e-4  
        if skip_mask is not None:
            probs = torch.masked_fill(probs, skip_mask, 0)

        if self.training and (sample_topk > 0):
            assert sample_topk <= self.k

            _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
            masked_probs = probs + 1e-6
            masked_probs[torch.arange(probs.size(0)).unsqueeze(
                1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:  
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)  


        top_k_gates = top_k_gates
        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            compute_gating(self.k, probs, top_k_gates, top_k_indices)  
        self.expert_size = expert_size
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates

        loss = 0.
        return loss

    def forward(self, x, task_bh, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        bsz, length, emb_size = x.size()  
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.top_k_gating(x, task_bh, bsz, skip_mask, sample_topk=sample_topk)  
        with open('output.txt', 'w') as file:
            for element in self.batch_index:
                file.write(str(element.item()) + '\n')
        expert_inputs = x[self.batch_index]  

        with open('output8.txt', 'w') as file:
            for element in self.batch_gates:
                file.write(str(element.item()) + '\n')
        expert_inputs = x[self.batch_index]  

        h = self.experts(expert_inputs, self.expert_size)  
        h = self.activation(h)  
        expert_outputs = self.output_experts(h, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]  

        zeros = torch.zeros((bsz * length, self.input_size),
                            dtype=expert_outputs.dtype, device=expert_outputs.device)  
        y = zeros.index_add(0, self.batch_index, expert_outputs)  
        y = y.view(bsz, length, self.input_size)  
        return y, loss

    def forward_(self, x, task_bh, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)

        logits = self.f_gate[task_bh](x)
        probs = torch.softmax(logits, dim=1)

        loss = self.top_k_gating(x, skip_mask, task_bh, sample_topk=sample_topk)
        assert not self.bias
        hs = [torch.einsum('li,ij->lj', x, self.experts.w[i]) for i in range(self.num_experts)]
        hs = [self.activation(h) for h in hs]
        expert_outputs = [
            torch.einsum('li,ij->lj', hs[i], self.output_experts.w[i]) for i in range(self.num_experts)
        ]
        y = sum(probs[..., i][..., None] * expert_outputs[i] for i in range(self.num_experts))
        y = y.view(bsz, length, self.input_size)

        return y

    def map(self, x, task_bh, skip_mask=None, sample_topk=0):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        bsz, length, emb_size = x.size()  
        x = x.reshape(-1, emb_size)
        loss = self.top_k_gating(x, task_bh, bsz, skip_mask, sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]  
        expert_outputs = self.experts(expert_inputs, self.expert_size)

        zeros = torch.zeros((bsz * length * self.k, self.head_size),
                            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.k, -1)
        return y, loss

    def reduce(self, x, multiply_by_gates=True):
        bsz, length, k, emb_size = x.size()
        x = x.view(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_experts(expert_inputs, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size),
                            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        return y


class RandomMoE(MoE):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, *args, **kwargs):
        super(RandomMoE, self).__init__(*args, **kwargs)
        del self.f_gate
        del self.w_noise

    def top_k_gating(self, x, skip_mask=None, sample_topk=0, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = torch.randn(x.shape[0], self.num_experts).to(x.device)
        clean_logits = clean_logits * 0.001 + 1  
        logits = clean_logits

        if skip_mask is not None:
            probs = torch.masked_fill(
                torch.softmax(logits, dim=1), skip_mask, 0)
        else:
            probs = torch.softmax(logits, dim=1)

        if self.training and (sample_topk > 0):
            assert sample_topk <= self.k

            _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
            masked_probs = probs + 1e-6
            masked_probs[torch.arange(probs.size(0)).unsqueeze(
                1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)


        zeros = torch.zeros_like(probs, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        self.expert_size = (gates > 0).long().sum(0)

        top_k_gates = top_k_gates.flatten()
        top_k_experts = top_k_indices.flatten()

        nonzeros = top_k_gates.nonzero().squeeze(-1)
        top_k_experts_nonzero = top_k_experts[nonzeros]

        _, _index_sorted_experts = top_k_experts_nonzero.sort(0)
        self.index_sorted_experts = nonzeros[_index_sorted_experts]
        self.batch_index = self.index_sorted_experts.div(self.k, rounding_mode='trunc')
        self.batch_gates = top_k_gates[self.index_sorted_experts]

        loss = 0
        return loss
