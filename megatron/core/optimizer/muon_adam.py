# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Tuple, Dict

import torch
import math
import torch.distributed as dist
try:
    from torch.distributed.tensor import Shard as DTensorShard
    _HAVE_DTENSOR = True
except ImportError:
    _HAVE_DTENSOR = False


# copy from https://github.com/KellerJordan/Muon/tree/master
# @torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    return X

def normalize_range(range: Tuple[int, int], start):
    return (range[0] - start, range[1] - start)

class MuonDistMeta:

    # which buffer and bucket param belongs to
    buffer_idx: int = 0
    bucket_idx: int = 0
    # param shape after tp
    shape: torch.Size = None
    # param location in global buffer
    global_range: Tuple[int, int] = None
    tp_split_dim: int = -1
    # param location in global buffer (current dp slice)
    local_range: Tuple[int, int] = None

    def __init__(self, buffer_idx: int, bucket_idx: int, shape: torch.Size, global_range: Tuple[int, int], tp_split_dim: int):
        self.buffer_idx = buffer_idx
        self.bucket_idx = bucket_idx
        self.shape = shape
        self.global_range = global_range
        self.tp_split_dim = tp_split_dim
    
    def set_local_buffer_range(self, local_buffer_range: Tuple[int, int]):
        start = max(self.global_range[0], local_buffer_range[0])
        end = min(self.global_range[1], local_buffer_range[1])
        self.local_range = (start, end) if start < end else (local_buffer_range[0], local_buffer_range[0])

# adjust LR based on: https://github.com/MoonshotAI/Moonlight
def adjust_lr_wd_for_muon(lr, matched_adamw_rms, param_shape):
    A, B = param_shape[:2]
    adjusted_ratio = math.sqrt(max(A, B)) * matched_adamw_rms
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr

# copy from https://github.com/KellerJordan/Muon/tree/master and support distributed solution
class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        param_groups: The parameters to be optimized.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        matched_adamw_rms: The AdamW Update RMS that Muon is designed to match. (0.2~0.4 recommended)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (5 is probably always enough)
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """
    def __init__(self, param_groups, lr=2e-2, weight_decay=0.1,
                 matched_adamw_rms=0.2, momentum=0.95, nesterov=True, ns_steps=5,
                 adamw_betas=(0.95, 0.95), adamw_eps=1e-8):

        defaults = dict(lr=lr, weight_decay=weight_decay,
                        matched_adamw_rms=matched_adamw_rms,
                        momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_betas=adamw_betas, adamw_eps=adamw_eps,)

        super().__init__(param_groups, defaults)
        self.distributed_mode = False


    def enable_distributed_mode(self, global_buffer_sizes, dist_group, tp_group,
                                dist_metas: Dict[torch.nn.Parameter, MuonDistMeta]):
        """
        enable distributed mode
        Args:
            global_buffer_size: global buffer size
            dist group: optimizer sharding group
            tp group: param tp group
            dist metas: dist metas for all param
        """

        self.global_buffer_sizes = global_buffer_sizes
        self.dist_group = dist_group
        self.tp_group = tp_group
        self.dist_metas = dist_metas

        world_size = dist.get_world_size(dist_group)
        rank = dist.get_rank(dist_group)

        # calc local buffer range
        self.local_buffer_sizes = []
        self.local_buffer_ranges = []
        for bucket_sizes in global_buffer_sizes:
            local_bucket_sizes = []
            local_bucket_ranges = []
            for (global_bucket_size, bucket_offset) in bucket_sizes:
                assert global_bucket_size % world_size == 0
                local_buffer_size = global_bucket_size // world_size
                local_buffer_start = local_buffer_size * rank + bucket_offset
                local_buffer_range = (local_buffer_start, local_buffer_start + local_buffer_size)
                local_bucket_sizes.append(local_buffer_size)
                local_bucket_ranges.append(local_buffer_range)

            self.local_buffer_sizes.append(local_bucket_sizes)
            self.local_buffer_ranges.append(local_bucket_ranges)

        # calc local range for params
        for dist_meta in dist_metas.values():
            local_buffer_range = self.local_buffer_ranges[dist_meta.buffer_idx][dist_meta.bucket_idx]
            dist_meta.set_local_buffer_range(local_buffer_range)

        self.distributed_mode = True

    def step(self):

        dtype = torch.bfloat16
        device = torch.cuda.current_device()

        ns_inputs = {}

        # update muon momentum first
        for group in self.param_groups:

            if not group.get("use_muon", False):
                continue

            momentum = group['momentum']
            params = group["params"]

            for p in params:

                g = p.grad
                assert g is not None

                # For FSDP DTensor params, work with local shard for momentum accumulation
                is_fsdp = hasattr(p, 'megatron_fsdp_dist_index')
                if is_fsdp:
                    g_local = g._local_tensor
                else:
                    # 1-dim grad for distributed mode
                    assert self.distributed_mode or g.dim() == 2
                    g_local = g

                # prepare muon buffer in state
                state = self.state[p]
                if "muon_buffer" not in state:
                    state["muon_buffer"] = torch.zeros_like(g_local)
                buf = state["muon_buffer"]
                buf.mul_(momentum).add_(g_local)

                # save to ns input (local shard, bfloat16)
                g_ns = g_local.add(buf, alpha=momentum) if group['nesterov'] else buf
                ns_inputs[p] = g_ns.bfloat16()

        # rewrite ns_inputs if ZeRO-1 distributed mode
        if self.distributed_mode:

            # initialize buffers
            ns_input_local_buffers = [
                [ torch.empty((local_buffer_size), device=device, dtype=dtype)
                    for local_buffer_size in local_bucket_sizes ]
                for local_bucket_sizes in self.local_buffer_sizes
            ]
            ns_input_global_buffers = [
                [ torch.empty((global_buffer_size), device=device, dtype=dtype)
                    for (global_buffer_size, bucket_offset) in global_bucket_sizes ]
                for global_bucket_sizes in self.global_buffer_sizes
            ]

            # fill ns input data to local buffer
            for param, ns_input in ns_inputs.items():
                dist_meta = self.dist_metas[param]
                ns_input_local_buffer = ns_input_local_buffers[dist_meta.buffer_idx][dist_meta.bucket_idx]
                local_buffer_range = self.local_buffer_ranges[dist_meta.buffer_idx][dist_meta.bucket_idx]
                local_range = normalize_range(dist_meta.local_range, local_buffer_range[0])
                ns_input_local_buffer[local_range[0]:local_range[1]].copy_(ns_input.view(-1))

            # all gather buffers
            for ns_input_global_buffer, ns_input_local_buffer in zip(ns_input_global_buffers, ns_input_local_buffers):
                for ns_input_global_bucket, ns_input_local_bucket in zip(ns_input_global_buffer, ns_input_local_buffer):
                    dist.all_gather_into_tensor(ns_input_global_bucket, ns_input_local_bucket, group=self.dist_group)

            # overwrite ns input
            for p in ns_inputs.keys():
                dist_meta = self.dist_metas[p]
                ns_input_global_buffer = ns_input_global_buffers[dist_meta.buffer_idx][dist_meta.bucket_idx]
                global_range = dist_meta.global_range
                offset = self.global_buffer_sizes[dist_meta.buffer_idx][dist_meta.bucket_idx][1]
                ns_inputs[p] = ns_input_global_buffer[global_range[0] - offset : global_range[1] - offset].view(dist_meta.shape)

            # set tp info
            tp_world_size = dist.get_world_size(self.tp_group)
            tp_rank = dist.get_rank(self.tp_group)

        # FSDP mode: reconstruct TP-local tensors for NS.
        # For DP-sharded params (ZeRO-2/3): all-gather local DP shards across the FSDP (inner DP)
        # group to get the full TP-local tensor (concatenates along dim 0).
        # For replicated params (no_shard/ZeRO-1): local tensor is already the TP-local tensor.
        fsdp_metas = {}  # p -> (dist_index, tp_split_dim, dp_world_size, dp_rank, is_dp_sharded)
        for p in list(ns_inputs.keys()):
            if not hasattr(p, 'megatron_fsdp_dist_index'):
                continue

            dist_index = p.megatron_fsdp_dist_index
            fsdp_group = dist_index.fsdp_group
            dp_world_size = dist.get_world_size(fsdp_group)
            dp_rank = dist.get_rank(fsdp_group)

            # Check if the DP dimension is actually sharded (ZeRO-2/3) or replicated (no_shard)
            is_dp_sharded = False
            dp_shard_dim_name = dist_index.dp_shard_dim
            if (_HAVE_DTENSOR and dp_shard_dim_name is not None
                    and p.device_mesh.mesh_dim_names is not None):
                mesh_dims = list(p.device_mesh.mesh_dim_names)
                if dp_shard_dim_name in mesh_dims:
                    dp_mesh_idx = mesh_dims.index(dp_shard_dim_name)
                    is_dp_sharded = isinstance(p.placements[dp_mesh_idx], DTensorShard)

            # Determine TP split dimension from original param attributes
            orig_param = p.orig_param
            tp_split_dim = -1
            if (getattr(orig_param, 'tensor_model_parallel', False)
                    and not getattr(orig_param, '_tp_duplicated', False)):
                tp_split_dim = getattr(orig_param, 'partition_dim', -1)

            fsdp_metas[p] = (dist_index, tp_split_dim, dp_world_size, dp_rank, is_dp_sharded)

            if is_dp_sharded:
                # All-gather local DP shard across FSDP group to get TP-local tensor
                local_shard = ns_inputs[p]  # bfloat16, shape: (rows/dp, cols[/tp])
                tp_local_rows = local_shard.shape[0] * dp_world_size
                full_tp_local = torch.empty(
                    (tp_local_rows, *local_shard.shape[1:]),
                    dtype=local_shard.dtype,
                    device=local_shard.device,
                )
                dist.all_gather_into_tensor(full_tp_local, local_shard.contiguous(), group=fsdp_group)
                ns_inputs[p] = full_tp_local  # TP-local 2D tensor
            # else: ns_inputs[p] is already the TP-local tensor (replicated across DP)

        # apply NS updates
        for group in self.param_groups:

            if not group.get('use_muon', False):
                continue

            lr = group["lr"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            matched_adamw_rms = group["matched_adamw_rms"]
            params = group["params"]

            for p in params:

                ns_input = ns_inputs[p]
                tp_split_dim = -1
                is_fsdp = p in fsdp_metas

                if self.distributed_mode:
                    dist_meta = self.dist_metas[p]
                    tp_split_dim = dist_meta.tp_split_dim
                elif is_fsdp:
                    dist_index, tp_split_dim, dp_world_size, dp_rank, is_dp_sharded = fsdp_metas[p]

                # gather tensor parallel ( if tp )
                if tp_split_dim != -1:
                    if self.distributed_mode:
                        ns_input_shards = [ torch.empty_like(ns_input) for _ in range(tp_world_size) ]
                        dist.all_gather(ns_input_shards, ns_input, self.tp_group)
                        ns_input = torch.cat(ns_input_shards, dim=tp_split_dim)
                    elif is_fsdp and dist_index.tp_dim is not None:
                        fsdp_tp_mesh = dist_index.get_submesh(dist_index.tp_dim)
                        fsdp_tp_grp = fsdp_tp_mesh.get_group()
                        fsdp_tp_ws = dist.get_world_size(fsdp_tp_grp)
                        fsdp_tp_rk = dist.get_rank(fsdp_tp_grp)
                        ns_input_shards = [ torch.empty_like(ns_input) for _ in range(fsdp_tp_ws) ]
                        dist.all_gather(ns_input_shards, ns_input, fsdp_tp_grp)
                        ns_input = torch.cat(ns_input_shards, dim=tp_split_dim)
                    else:
                        tp_split_dim = -1  # no TP group available, skip TP gather

                # calc update
                update = zeropower_via_newtonschulz5(ns_input, steps=ns_steps)

                # only local tp part
                if tp_split_dim != -1:
                    if self.distributed_mode:
                        update = update.chunk(tp_world_size, dim=tp_split_dim)[tp_rank]
                    else:  # FSDP
                        update = update.chunk(fsdp_tp_ws, dim=tp_split_dim)[fsdp_tp_rk]

                # apply weight decay and update
                if self.distributed_mode:
                    # only local buffer part
                    local_range_in_global_range = normalize_range(dist_meta.local_range, dist_meta.global_range[0])
                    update = update.reshape(-1)[local_range_in_global_range[0]:local_range_in_global_range[1]]
                    p.data.mul_(1 - lr*weight_decay)
                    adjusted_lr = adjust_lr_wd_for_muon(lr, matched_adamw_rms, ns_input.shape)
                    p.data.add_(update, alpha=-adjusted_lr)
                elif is_fsdp:
                    # Take DP-local shard (along dim 0) when DP-sharded (ZeRO-2/3),
                    # or use the full update for replicated params (no_shard/ZeRO-1)
                    update_local = update.chunk(dp_world_size, dim=0)[dp_rank] if is_dp_sharded else update
                    local_tensor = p._local_tensor
                    adjusted_lr = adjust_lr_wd_for_muon(lr, matched_adamw_rms, ns_input.shape)
                    local_tensor.mul_(1 - lr * weight_decay)
                    local_tensor.add_(update_local.to(local_tensor.dtype), alpha=-adjusted_lr)
                else:
                    # apply weight decay
                    p.data.mul_(1 - lr*weight_decay)
                    #  adjust lr and apply update
                    adjusted_lr = adjust_lr_wd_for_muon(lr, matched_adamw_rms, ns_input.shape)
                    p.data.add_(update, alpha=-adjusted_lr)

        # use adam for other params
        for group in self.param_groups:

            if group.get('use_muon', False):
                continue

            # init step
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            step = group['step']
            params = group["params"]
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']

            for p in params:

                g = p.grad
                assert g is not None
                state = self.state[p]

                if len(state) == 0:
                    state['adamw_exp_avg'] = torch.zeros_like(g)
                    state['adamw_exp_avg_sq'] = torch.zeros_like(g)

                buf1 = state['adamw_exp_avg']
                buf2 = state['adamw_exp_avg_sq']
                buf1.lerp_(g, 1-beta1)
                buf2.lerp_(g.square(), 1-beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr/scale)
