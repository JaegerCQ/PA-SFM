"""Shared acoustic forward-model utilities.

Two forward implementations live here:

  - Triton kernels (`gaussian_sim`)
      Operates on the full 400^3 source-grid Pc. Used by training (steps 1, 2a)
      and dense inference (step 2b).

  - Pure-PyTorch sparse forward (`sparse_acoustic_sim`)
      Operates on the top-K thresholded source points. Used by per-sensor
      localization (step 3) and pose refinement (step 5), where the source
      count is small (<5000) and a vectorized PyTorch loop is faster than
      launching Triton.

Both implement the same physical model:
  out[s, t] = sum_k Pc[k] * 0.5 * ((r_k - vs*t) / r_k) * exp(-(r_k - vs*t)^2 / (2*sigma^2))

with r_k = |sens_pos[s] - src_pos[k]| and t = (t_start_idx + t_rel) * delta_t.
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

VS_DEFAULT = 1500.0          # m/s
SIGMA_DEFAULT = 0.1e-3       # m
GRID_SIZE_DEFAULT = 400
VOXEL_DEFAULT = SIGMA_DEFAULT

# KDE soft-bin forward constants — frozen by validate_kde.py --mode math (Step 1).
# Derived from sensor candidate cube [-0.16, 0.16]^3 × source grid 400^3 × 0.1 mm
# with ±10 mm safety margin. r_max picks the worst-case 2·sqrt(3)·max(sens, src).
KDE_DELTA = 1.25e-5                  # m, = SIGMA_DEFAULT / 8
KDE_R_MIN = -0.01                    # m
KDE_R_MAX = 0.5642562584220407       # m
KDE_N_BINS = 45941                   # = ceil((KDE_R_MAX - KDE_R_MIN) / KDE_DELTA)

STRICT_ATOMICS = os.environ.get("REPRO_STRICT_ATOMICS", "1").strip().lower() not in {"0", "false", "no", "off"}
FIXED_POINT_SCALE = float(os.environ.get("REPRO_FIXED_POINT_SCALE", "10000000000.0"))
FIXED_POINT_INV_SCALE = 1.0 / FIXED_POINT_SCALE


# -------------------------- Triton (full grid) --------------------------

@triton.jit
def _forward_kernel(
    Pc_ptr, src_x_ptr, src_y_ptr, src_z_ptr,
    sens_x_ptr, sens_y_ptr, sens_z_ptr, out_ptr,
    n_sources, n_time_sub, t_start_idx,
    delta_t, vs, sigma,
    stride_out_s, stride_out_t,
    BLOCK_K: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)
    t_idx_rel = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_idx_rel < n_time_sub
    t_vals = (t_start_idx + t_idx_rel) * delta_t

    sx = tl.load(sens_x_ptr + pid_s)
    sy = tl.load(sens_y_ptr + pid_s)
    sz = tl.load(sens_z_ptr + pid_s)

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for k in range(0, n_sources, BLOCK_K):
        idx_k = k + tl.arange(0, BLOCK_K)
        mask_k = idx_k < n_sources
        px = tl.load(src_x_ptr + idx_k, mask=mask_k, other=0.0)
        py = tl.load(src_y_ptr + idx_k, mask=mask_k, other=0.0)
        pz = tl.load(src_z_ptr + idx_k, mask=mask_k, other=0.0)
        pc = tl.load(Pc_ptr + idx_k, mask=mask_k, other=0.0)
        dx, dy, dz = sx - px, sy - py, sz - pz
        r = tl.sqrt(dx * dx + dy * dy + dz * dz + 1e-12)
        r_mat, pc_mat, t_mat = r[:, None], pc[:, None], t_vals[None, :]
        rt = r_mat - vs * t_mat
        contrib = pc_mat * 0.5 * (rt / r_mat) * tl.exp(-(rt * rt) / (2.0 * sigma * sigma))
        contrib = tl.where(mask_k[:, None], contrib, 0.0)
        acc += tl.sum(contrib, axis=0)

    out_ptrs = out_ptr + pid_s * stride_out_s + t_idx_rel * stride_out_t
    tl.store(out_ptrs, acc, mask=t_mask)


@triton.jit
def _backward_kernel(
    src_x_ptr, src_y_ptr, src_z_ptr,
    sens_x_ptr, sens_y_ptr, sens_z_ptr,
    grad_out_ptr, grad_pc_ptr,
    n_sources, n_sensors, n_time_sub, t_start_idx,
    delta_t, vs, sigma,
    stride_go_s, stride_go_t,
    BLOCK_K: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)
    t_idx_rel = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_idx_rel < n_time_sub
    t_vals = (t_start_idx + t_idx_rel) * delta_t

    sx = tl.load(sens_x_ptr + pid_s)
    sy = tl.load(sens_y_ptr + pid_s)
    sz = tl.load(sens_z_ptr + pid_s)

    go_ptrs = grad_out_ptr + pid_s * stride_go_s + t_idx_rel * stride_go_t
    go = tl.load(go_ptrs, mask=t_mask, other=0.0)

    for k in range(0, n_sources, BLOCK_K):
        idx_k = k + tl.arange(0, BLOCK_K)
        mask_k = idx_k < n_sources
        px = tl.load(src_x_ptr + idx_k, mask=mask_k, other=0.0)
        py = tl.load(src_y_ptr + idx_k, mask=mask_k, other=0.0)
        pz = tl.load(src_z_ptr + idx_k, mask=mask_k, other=0.0)
        dx, dy, dz = sx - px, sy - py, sz - pz
        r = tl.sqrt(dx * dx + dy * dy + dz * dz + 1e-12)
        r_mat, t_mat = r[:, None], t_vals[None, :]
        rt = r_mat - vs * t_mat
        dfdpc = 0.5 * (rt / r_mat) * tl.exp(-(rt * rt) / (2.0 * sigma * sigma))
        dfdpc = tl.where(mask_k[:, None], dfdpc, 0.0)
        grad_k = tl.sum(dfdpc * go[None, :], axis=1)
        tl.atomic_add(grad_pc_ptr + idx_k, grad_k, mask=mask_k)


@triton.jit
def _backward_kernel_fixed(
    src_x_ptr, src_y_ptr, src_z_ptr,
    sens_x_ptr, sens_y_ptr, sens_z_ptr,
    grad_out_ptr, grad_pc_ptr,
    n_sources, n_sensors, n_time_sub, t_start_idx,
    delta_t, vs, sigma,
    stride_go_s, stride_go_t,
    fixed_scale,
    BLOCK_K: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)
    t_idx_rel = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_idx_rel < n_time_sub
    t_vals = (t_start_idx + t_idx_rel) * delta_t

    sx = tl.load(sens_x_ptr + pid_s)
    sy = tl.load(sens_y_ptr + pid_s)
    sz = tl.load(sens_z_ptr + pid_s)

    go_ptrs = grad_out_ptr + pid_s * stride_go_s + t_idx_rel * stride_go_t
    go = tl.load(go_ptrs, mask=t_mask, other=0.0)

    for k in range(0, n_sources, BLOCK_K):
        idx_k = k + tl.arange(0, BLOCK_K)
        mask_k = idx_k < n_sources
        px = tl.load(src_x_ptr + idx_k, mask=mask_k, other=0.0)
        py = tl.load(src_y_ptr + idx_k, mask=mask_k, other=0.0)
        pz = tl.load(src_z_ptr + idx_k, mask=mask_k, other=0.0)
        dx, dy, dz = sx - px, sy - py, sz - pz
        r = tl.sqrt(dx * dx + dy * dy + dz * dz + 1e-12)
        r_mat, t_mat = r[:, None], t_vals[None, :]
        rt = r_mat - vs * t_mat
        dfdpc = 0.5 * (rt / r_mat) * tl.exp(-(rt * rt) / (2.0 * sigma * sigma))
        dfdpc = tl.where(mask_k[:, None], dfdpc, 0.0)
        grad_k = tl.sum(dfdpc * go[None, :], axis=1)
        v = grad_k * fixed_scale
        q = tl.where(v >= 0.0, tl.floor(v + 0.5), -tl.floor(-v + 0.5)).to(tl.int64)
        tl.atomic_add(grad_pc_ptr + idx_k, q, mask=mask_k)


class _GaussianSimFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Pc, src_x, src_y, src_z, sens_x, sens_y, sens_z,
                t_start_idx, n_time_sub, delta_t, vs, sigma):
        out = torch.empty((sens_x.numel(), n_time_sub), device=Pc.device, dtype=torch.float32)
        BLOCK_T, BLOCK_K = 32, 32
        grid = (triton.cdiv(n_time_sub, BLOCK_T), sens_x.numel())
        _forward_kernel[grid](
            Pc, src_x, src_y, src_z, sens_x, sens_y, sens_z, out,
            Pc.numel(), n_time_sub, t_start_idx, delta_t, vs, sigma,
            out.stride(0), out.stride(1), BLOCK_K=BLOCK_K, BLOCK_T=BLOCK_T,
            num_warps=4, num_stages=2,
        )
        ctx.save_for_backward(src_x, src_y, src_z, sens_x, sens_y, sens_z)
        ctx.n_sources, ctx.n_sensors, ctx.n_time_sub = Pc.numel(), sens_x.numel(), n_time_sub
        ctx.t_start_idx, ctx.delta_t, ctx.vs, ctx.sigma = t_start_idx, delta_t, vs, sigma
        return out

    @staticmethod
    def backward(ctx, grad_out):
        src_x, src_y, src_z, sens_x, sens_y, sens_z = ctx.saved_tensors
        BLOCK_T, BLOCK_K = 32, 32
        grid = (triton.cdiv(ctx.n_time_sub, BLOCK_T), ctx.n_sensors)
        if STRICT_ATOMICS:
            grad_pc_q = torch.zeros(ctx.n_sources, device=grad_out.device, dtype=torch.int64)
            _backward_kernel_fixed[grid](
                src_x, src_y, src_z, sens_x, sens_y, sens_z,
                grad_out.contiguous(), grad_pc_q,
                ctx.n_sources, ctx.n_sensors, ctx.n_time_sub, ctx.t_start_idx,
                ctx.delta_t, ctx.vs, ctx.sigma,
                grad_out.stride(0), grad_out.stride(1),
                FIXED_POINT_SCALE,
                BLOCK_K=BLOCK_K, BLOCK_T=BLOCK_T, num_warps=4, num_stages=2,
            )
            grad_pc = grad_pc_q.to(torch.float32) * FIXED_POINT_INV_SCALE
            del grad_pc_q
        else:
            grad_pc = torch.zeros(ctx.n_sources, device=grad_out.device, dtype=torch.float32)
            _backward_kernel[grid](
                src_x, src_y, src_z, sens_x, sens_y, sens_z,
                grad_out.contiguous(), grad_pc,
                ctx.n_sources, ctx.n_sensors, ctx.n_time_sub, ctx.t_start_idx,
                ctx.delta_t, ctx.vs, ctx.sigma,
                grad_out.stride(0), grad_out.stride(1),
                BLOCK_K=BLOCK_K, BLOCK_T=BLOCK_T, num_warps=4, num_stages=2,
            )
        return grad_pc, None, None, None, None, None, None, None, None, None, None, None


def gaussian_sim(Pc, src_x, src_y, src_z, sens_x, sens_y, sens_z,
                 t_start_idx, n_time_sub, delta_t, vs=VS_DEFAULT, sigma=SIGMA_DEFAULT):
    """Triton-backed forward simulation over the full source grid."""
    return _GaussianSimFunction.apply(Pc, src_x, src_y, src_z, sens_x, sens_y, sens_z,
                                      t_start_idx, n_time_sub, delta_t, vs, sigma)


# -------------------------- pure-PyTorch (sparse) --------------------------

def sparse_acoustic_sim(sens_pos, src_pos, Pc, t_start_idx, n_time_sub, delta_t,
                        vs=VS_DEFAULT, sigma=SIGMA_DEFAULT):
    """Sparse forward — sens_pos: (3,); src_pos: (K,3); Pc: (K,).

    Returns shape (n_time_sub,). Used when K is small (top-K thresholded
    sources, typically <5000). For batched sensors, use `sparse_acoustic_sim_batch`.
    """
    diff = sens_pos.unsqueeze(0) - src_pos       # (K, 3)
    r = torch.norm(diff, dim=1) + 1e-12          # (K,)
    t_rel = torch.arange(n_time_sub, device=Pc.device, dtype=torch.float32)
    t_phys = (t_start_idx + t_rel) * delta_t     # (T,)
    rt = r.unsqueeze(1) - vs * t_phys.unsqueeze(0)               # (K, T)
    envelope = torch.exp(-(rt ** 2) / (2.0 * sigma * sigma))
    amplitude = Pc.unsqueeze(1) * 0.5 * (rt / r.unsqueeze(1))
    return torch.sum(amplitude * envelope, dim=0)


def sparse_acoustic_sim_batch(sens_pos_batch, src_pos, Pc, t_start_idx, n_time_sub,
                              delta_t, vs=VS_DEFAULT, sigma=SIGMA_DEFAULT):
    """Vectorized sparse forward over a batch of sensors.

    sens_pos_batch: (B, 3); src_pos: (K, 3); Pc: (K,).
    Returns (B, n_time_sub). Memory ~ B*K*T*float32 — pick B accordingly.
    """
    diff = sens_pos_batch.unsqueeze(1) - src_pos.unsqueeze(0)    # (B, K, 3)
    r = torch.norm(diff, dim=2) + 1e-12                          # (B, K)
    t_rel = torch.arange(n_time_sub, device=Pc.device, dtype=torch.float32)
    t_phys = (t_start_idx + t_rel) * delta_t
    rt = r.unsqueeze(2) - vs * t_phys.reshape(1, 1, -1)          # (B, K, T)
    envelope = torch.exp(-(rt ** 2) / (2.0 * sigma * sigma))
    amplitude = Pc.reshape(1, -1, 1) * 0.5 * (rt / r.unsqueeze(2))
    return torch.sum(amplitude * envelope, dim=1)


def kde_acoustic_sim(sens_pos, src_pos, Pc, t_start_idx, n_time_sub, delta_t,
                     vs=VS_DEFAULT, sigma=SIGMA_DEFAULT,
                     delta_bin=None, r_min=None, n_bins=None):
    """KDE soft-bin + 1D-conv forward — equivalent to sparse_acoustic_sim up
    to a Δ²/6 σ² inflation (~0.26% at Δ=σ_target/8). Memory drops from
    O(K·T) (~256 MB) to O(n_bins) (~360 KB), trading HBM bandwidth for
    arithmetic.

    Math:  p(t) = Σ_s Pc_s · 0.5·(r_s - vs·t)/r_s · exp(-(r_s - vs·t)²/2σ²)
                = Σ_s w_s · f(r_s - vs·t),  w_s = Pc_s/(2 r_s),
                                            f(d) = d·exp(-d²/2σ²)

    sens_pos: (3,); src_pos: (K, 3); Pc: (K,)
    Returns (n_time_sub,) with dtype matching inputs.
    """
    delta_bin = KDE_DELTA if delta_bin is None else delta_bin
    r_min_v = KDE_R_MIN if r_min is None else r_min
    n_bins_v = KDE_N_BINS if n_bins is None else n_bins
    device = Pc.device
    dtype = Pc.dtype

    diff = sens_pos.unsqueeze(0) - src_pos
    r = torch.norm(diff, dim=1) + 1e-12
    w = Pc / (2.0 * r)

    pos = (r - r_min_v) / delta_bin
    i0 = pos.floor().long().clamp_(0, n_bins_v - 1)
    alpha = pos - i0.to(dtype)

    # Extend by 1 slot to absorb i0+1 at the right edge, then trim.
    if STRICT_ATOMICS:
        val0 = (1.0 - alpha) * w
        val1 = alpha * w
        h_q = torch.zeros(n_bins_v + 1, device=device, dtype=torch.int64)
        v0 = val0 * FIXED_POINT_SCALE
        v1 = val1 * FIXED_POINT_SCALE
        q0 = torch.where(v0 >= 0.0, torch.floor(v0 + 0.5), -torch.floor(-v0 + 0.5)).to(torch.int64)
        q1 = torch.where(v1 >= 0.0, torch.floor(v1 + 0.5), -torch.floor(-v1 + 0.5)).to(torch.int64)
        h_q.index_add_(0, i0, q0)
        h_q.index_add_(0, i0 + 1, q1)
        h_det = h_q[:n_bins_v].to(dtype) * FIXED_POINT_INV_SCALE
        del h_q, q0, q1

        if torch.is_grad_enabled() and (val0.requires_grad or val1.requires_grad):
            h_grad = torch.zeros(n_bins_v + 1, device=device, dtype=dtype)
            h_grad.index_add_(0, i0, val0)
            h_grad.index_add_(0, i0 + 1, val1)
            h_grad = h_grad[:n_bins_v]
            h = h_det.detach() + (h_grad - h_grad.detach())
        else:
            h = h_det
    else:
        h = torch.zeros(n_bins_v + 1, device=device, dtype=dtype)
        h.index_add_(0, i0, (1.0 - alpha) * w)
        h.index_add_(0, i0 + 1, alpha * w)
        h = h[:n_bins_v]

    K_half = int(np.ceil(5.0 * sigma / delta_bin))
    d = torch.arange(-K_half, K_half + 1, device=device, dtype=dtype) * delta_bin
    k = d * torch.exp(-d * d / (2.0 * sigma * sigma))
    # F.conv1d is cross-correlation: out[j] = Σ_n input[j+n-pad] * weight[n].
    # We want p_grid[j] = Σ_m h[j+m]·f(m·Δ). With pad=K_half and m=n-K_half,
    # we need weight[n] = f((n-K_half)·Δ) = d[n]·exp(...) — i.e., k itself,
    # NO flip. (The numpy reference flips because np.convolve is true
    # convolution, not cross-correlation.)
    p_grid = F.conv1d(h.view(1, 1, -1), k.view(1, 1, -1), padding=K_half).view(-1)

    t_rel = torch.arange(n_time_sub, device=device, dtype=dtype)
    u = vs * (t_start_idx + t_rel) * delta_t
    pos_t = (u - r_min_v) / delta_bin
    j0 = pos_t.floor().long().clamp_(0, n_bins_v - 2)
    beta = pos_t - j0.to(dtype)
    return (1.0 - beta) * p_grid[j0] + beta * p_grid[j0 + 1]


# -------------------------- helpers --------------------------

def build_source_grid(grid_size, voxel_size, device, dtype=torch.float32):
    """Return contiguous (src_x, src_y, src_z) tensors for a centered cubic grid."""
    center = (grid_size - 1) / 2.0
    coords = (np.arange(grid_size) - center) * voxel_size
    xg, yg, zg = np.meshgrid(coords, coords, coords, indexing="ij")
    return (
        torch.tensor(xg.astype(np.float32).ravel(), device=device, dtype=dtype).contiguous(),
        torch.tensor(yg.astype(np.float32).ravel(), device=device, dtype=dtype).contiguous(),
        torch.tensor(zg.astype(np.float32).ravel(), device=device, dtype=dtype).contiguous(),
    )


def load_phantom_topk(ckpt_path, device, keep_ratio=0.002,
                      grid_size=GRID_SIZE_DEFAULT, voxel_size=VOXEL_DEFAULT):
    """Load a trained Pc checkpoint and return the top-K (by |Pc|) sparse source set.

    Returns:
      xyz_src : (K, 3) source positions
      Pc      : (K,)   intensities
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    Pc_raw = ckpt["Pc_state"].to(device).contiguous()
    src_x, src_y, src_z = build_source_grid(grid_size, voxel_size, device)
    n_keep = int(Pc_raw.numel() * keep_ratio)
    _, idx_top = torch.topk(torch.abs(Pc_raw), n_keep)
    xyz_src = torch.stack([src_x[idx_top], src_y[idx_top], src_z[idx_top]], dim=1).contiguous()
    Pc = Pc_raw[idx_top].contiguous()
    return xyz_src, Pc


def tgv2_regularization(A_flat, grid_shape, alpha0=2.0, alpha1=1.0, eps=1e-8):
    """Total Generalized Variation (order 2) on a 3D volume reshape of A_flat."""
    A = A_flat.view(*grid_shape)

    def fwd_diff(t, dim):
        d = torch.roll(t, shifts=-1, dims=dim) - t
        idx = [slice(None)] * t.ndim
        idx[dim] = slice(-1, None)
        d[tuple(idx)] = 0.0
        return d

    gx, gy, gz = fwd_diff(A, 0), fwd_diff(A, 1), fwd_diff(A, 2)
    grad_mag = torch.sqrt(gx * gx + gy * gy + gz * gz + eps)
    gxx, gyy, gzz = fwd_diff(gx, 0), fwd_diff(gy, 1), fwd_diff(gz, 2)
    gxy = 0.5 * (fwd_diff(gx, 1) + fwd_diff(gy, 0))
    gxz = 0.5 * (fwd_diff(gx, 2) + fwd_diff(gz, 0))
    gyz = 0.5 * (fwd_diff(gy, 2) + fwd_diff(gz, 1))
    second_sq = gxx ** 2 + gyy ** 2 + gzz ** 2 + 2.0 * (gxy ** 2 + gxz ** 2 + gyz ** 2)
    return alpha1 * grad_mag.mean() + alpha0 * torch.sqrt(second_sq + eps).mean()
