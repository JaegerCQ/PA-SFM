"""Step 3 — single-sensor coarse-to-fine localization.

For each sensor, optimize its 3D position by matching its simulated signal
(against the trained pose0 phantom Pc) to its observed pose10-aligned signal.

  - Coarse: dense grid search → keep top-K candidates by negative-correlation
  - Fine: from each candidate, Adam gradient descent with annealed sigma
  - Pick the candidate with lowest final loss

Inputs:
  data/simulated_signals_full_4096_pose10.txt  — from step 2b
  data/sensor_location_pose10.txt              — ground-truth pose10 sensors (for error report)
  checkpoints/pose0/model.pt                   — phantom volume from step 1

Output:
  outputs/predicted_locations_pose10_10MHz.txt — (n_sensors, 3) predicted positions (m)

Source: legacy/in_vivo_liver_array_predict_sh_downsample_pose10.ipynb
"""
import argparse
import time
from pathlib import Path

from lib.runtime_config import DEFAULT_SEED, seed_everything

import numpy as np
import torch

from lib.forward import (
    VS_DEFAULT, kde_acoustic_sim, load_phantom_topk, sparse_acoustic_sim,
)

ROOT = Path(__file__).resolve().parent
SEED = seed_everything(DEFAULT_SEED)


def neg_corr_loss(pred, target):
    p = pred - pred.mean()
    t = target - target.mean()
    return 1.0 - torch.sum(p * t) / (torch.norm(p) * torch.norm(t) + 1e-9)


def coarse_search(xyz_src, Pc, target_sig, bounds, step, k, t_start, n_time, delta_t,
                  device, forward_fn):
    xs = torch.arange(bounds[0][0], bounds[0][1], step)
    ys = torch.arange(bounds[1][0], bounds[1][1], step)
    zs = torch.arange(bounds[2][0], bounds[2][1], step)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    candidates = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=1).to(device)

    losses = []
    with torch.no_grad():
        for pos in candidates:
            pred = forward_fn(pos, xyz_src, Pc, t_start, n_time, delta_t,
                              vs=VS_DEFAULT, sigma=0.5e-3)
            losses.append(neg_corr_loss(pred, target_sig).item())
    top = torch.topk(torch.tensor(losses), k, largest=False).indices
    return candidates[top].cpu().numpy()


def fine_search(xyz_src, Pc, target_sig, start_pos, total_epochs, lr,
                sigma_start, sigma_target, t_start, n_time, delta_t,
                device, forward_fn):
    param = torch.nn.Parameter(torch.tensor(start_pos, device=device, dtype=torch.float32),
                               requires_grad=True)
    opt = torch.optim.Adam([param], lr=lr)
    final_loss = None
    for epoch in range(total_epochs):
        opt.zero_grad()
        sigma = sigma_start * (sigma_target / sigma_start) ** min(1.0, epoch / (total_epochs * 0.8))
        pred = forward_fn(param, xyz_src, Pc, t_start, n_time, delta_t,
                          vs=VS_DEFAULT, sigma=sigma)
        loss = neg_corr_loss(pred, target_sig)
        loss.backward()
        opt.step()
        final_loss = loss.item()
    return param.detach().cpu().numpy(), final_loss


def solve_sensor(xyz_src, Pc, true_pos, full_sig, args, device, forward_fn):
    target_sig = full_sig[args.t_start:args.t_end]
    starts = coarse_search(xyz_src, Pc, target_sig,
                           bounds=[[-0.16, 0.16]] * 3, step=0.02, k=args.top_k,
                           t_start=args.t_start, n_time=args.t_end - args.t_start,
                           delta_t=args.delta_t, device=device, forward_fn=forward_fn)
    best_pos, best_loss = None, float("inf")
    for start in starts:
        pos, loss = fine_search(xyz_src, Pc, target_sig, start,
                                args.total_epochs, args.lr,
                                args.sigma_start, args.sigma_target,
                                args.t_start, args.t_end - args.t_start,
                                args.delta_t, device, forward_fn)
        if loss < best_loss:
            best_pos, best_loss = pos, loss
    err_mm = np.linalg.norm(best_pos - true_pos) * 1000
    return best_pos, 1.0 - best_loss, err_mm


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--signal", default=str(ROOT / "data/simulated_signals_full_4096_pose10.txt"))
    ap.add_argument("--sensor_gt", default=str(ROOT / "data/sensor_location_pose10.txt"),
                    help="ground-truth positions, used only for the per-sensor error printout")
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/pose0/model.pt"))
    ap.add_argument("--out", default=str(ROOT / "outputs/predicted_locations_pose10_10MHz.txt"))
    ap.add_argument("--keep_ratio", type=float, default=0.002,
                    help="fraction of source-grid voxels kept by |Pc| top-K")
    ap.add_argument("--top_k", type=int, default=15, help="coarse candidates per sensor")
    ap.add_argument("--total_epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.0005)
    ap.add_argument("--sigma_start", type=float, default=1.5e-3)
    ap.add_argument("--sigma_target", type=float, default=0.1e-3)
    ap.add_argument("--delta_t", type=float, default=100e-9, help="10MHz default")
    ap.add_argument("--t_start", type=int, default=500)
    ap.add_argument("--t_end", type=int, default=1000)
    ap.add_argument("--downsample", type=int, default=4, help="time decimation (1=already-10MHz)")
    ap.add_argument("--sensor_limit", type=int, default=0, help="stop after N sensors (0=all)")
    ap.add_argument("--sensor_ids", default="",
                    help="comma-separated indices; overrides --sensor_limit when set")
    ap.add_argument("--use_kde", action="store_true",
                    help="use kde_acoustic_sim forward in coarse and fine search")
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    forward_fn = kde_acoustic_sim if args.use_kde else sparse_acoustic_sim
    print(f"[INIT] device={device} forward={forward_fn.__name__}")

    print(f"[LOAD] phantom: {args.ckpt}")
    xyz_src, Pc = load_phantom_topk(args.ckpt, device, keep_ratio=args.keep_ratio)
    print(f"[INFO] kept {xyz_src.shape[0]} source points")

    signals = np.loadtxt(args.signal)
    if args.downsample > 1:
        signals = signals[:, ::args.downsample]
    sensor_gt = np.loadtxt(args.sensor_gt)
    total = signals.shape[0]
    if args.sensor_ids.strip():
        sensor_ids = [int(x) for x in args.sensor_ids.split(",")]
    elif args.sensor_limit > 0:
        sensor_ids = list(range(min(args.sensor_limit, total)))
    else:
        sensor_ids = list(range(total))
    print(f"[INFO] {len(sensor_ids)} sensors selected (of {total} total), "
          f"{signals.shape[1]} time samples")

    signals_t = torch.tensor(signals, dtype=torch.float32, device=device)
    preds = np.full((total, 3), np.nan, dtype=np.float64)
    t0 = time.time()
    for step, i in enumerate(sensor_ids):
        pos, corr, err_mm = solve_sensor(xyz_src, Pc, sensor_gt[i], signals_t[i],
                                         args, device, forward_fn)
        preds[i] = pos
        elapsed = time.time() - t0
        eta = elapsed / (step + 1) * (len(sensor_ids) - step - 1)
        print(f"[{step+1}/{len(sensor_ids)}] sensor#{i} "
              f"pred=[{pos[0]*1000:+.2f},{pos[1]*1000:+.2f},{pos[2]*1000:+.2f}]mm "
              f"corr={corr:.4f} err={err_mm:.2f}mm  ({elapsed:.0f}s elapsed, eta {eta:.0f}s)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, preds, fmt="%.6f",
               header="Predicted x, y, z (meters); NaN = sensor not solved")
    print(f"[DONE] saved {out_path}")


if __name__ == "__main__":
    main()
