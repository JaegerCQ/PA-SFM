import os
import argparse

from lib.runtime_config import DEFAULT_SEED, seed_everything

# ===============================================================
# 0. 命令行参数：先解析 device-id，再设置 CUDA_VISIBLE_DEVICES
# ===============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--device-id", type=int, default=0, help="物理 GPU ID")
parser.add_argument("--input-dir", type=str, default="./", help="输入文件目录")
parser.add_argument("--output-dir", type=str, default="./", help="输出目录")
parser.add_argument("--signal-files", nargs="+", required=True, help="信号 txt 文件列表")
parser.add_argument("--location-files", nargs="+", required=True, help="探头坐标 txt 文件列表")
parser.add_argument("--output-prefix", type=str, required=True, help="输出 mat 文件名前缀")
parser.add_argument("--method", type=str, default="DAS", choices=["DAS", "FBP"], help="重建方法")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

import time
import numpy as np
import scipy.io as sio
import torch
import triton
import triton.language as tl

SEED = seed_everything(DEFAULT_SEED)

# ==========================================
# 1. 路径与文件配置
# ==========================================
input_dir = args.input_dir
output_dir = args.output_dir

signal_files = args.signal_files
location_files = args.location_files

assert len(signal_files) == len(location_files), (
    "错误：信号文件名列表和位置文件名列表的数量不一致！"
)

output_file_prefix = args.output_prefix

# ==========================================
# 2. 算法选择与参数配置
# ==========================================
recon_method = args.method

vs = 1500.0
fs = 40e6
num_times = 4096
res = 0.1e-3

x_start, x_end = -12.5e-3, 17.5e-3
y_start, y_end = -10.0e-3, 40.0e-3
z_start, z_end = -20.0e-3, 0.0e-3

num_x = np.around((x_end - x_start) / res).astype(np.int32)
num_y = np.around((y_end - y_start) / res).astype(np.int32)
num_z = np.around((z_end - z_start) / res).astype(np.int32)

num_voxels = int(num_x) * int(num_y) * int(num_z)

print("=" * 80)
print("[CONFIG]")
print(f"device_id={args.device_id}")
print(f"input_dir={input_dir}")
print(f"output_dir={output_dir}")
print(f"method={recon_method}")
print(f"signal_files={signal_files}")
print(f"location_files={location_files}")
print(f"output_file_prefix={output_file_prefix}")
print(f"grid={int(num_x)}x{int(num_y)}x{int(num_z)}")
print("=" * 80)

# ==========================================
# 3. Triton Kernels
# ==========================================
@triton.jit
def recon_kernel_triton(
    signal_ptr,
    location_ptr,
    output_ptr,
    x_start: tl.constexpr,
    y_start: tl.constexpr,
    z_start: tl.constexpr,
    res: tl.constexpr,
    vs: tl.constexpr,
    fs: tl.constexpr,
    NUM_X: tl.constexpr,
    NUM_Y: tl.constexpr,
    NUM_Z: tl.constexpr,
    NUM_DETECTORS: tl.constexpr,
    NUM_TIMES: tl.constexpr,
    METHOD: tl.constexpr,
    BLOCK_VOXELS: tl.constexpr,
    BLOCK_DETECTORS: tl.constexpr,
):
    pid = tl.program_id(0)

    voxel_offsets = pid * BLOCK_VOXELS + tl.arange(0, BLOCK_VOXELS)
    voxel_mask = voxel_offsets < (NUM_X * NUM_Y * NUM_Z)

    k = voxel_offsets % NUM_Z
    j = (voxel_offsets // NUM_Z) % NUM_Y
    i = voxel_offsets // (NUM_Y * NUM_Z)

    x = x_start + i.to(tl.float32) * res
    y = y_start + j.to(tl.float32) * res
    z = z_start + k.to(tl.float32) * res

    acc = tl.zeros((BLOCK_VOXELS,), dtype=tl.float32)

    det_block = tl.arange(0, BLOCK_DETECTORS)

    for det_start in range(0, NUM_DETECTORS, BLOCK_DETECTORS):
        det_idx = det_start + det_block
        det_mask = det_idx < NUM_DETECTORS

        loc_base = det_idx * 3

        det_x = tl.load(location_ptr + loc_base + 0, mask=det_mask, other=0.0)
        det_y = tl.load(location_ptr + loc_base + 1, mask=det_mask, other=0.0)
        det_z = tl.load(location_ptr + loc_base + 2, mask=det_mask, other=0.0)

        dx = x[:, None] - det_x[None, :]
        dy = y[:, None] - det_y[None, :]
        dz = z[:, None] - det_z[None, :]

        d2 = dx * dx + dy * dy + dz * dz
        d = tl.sqrt(d2)

        vector_n0 = tl.sqrt(det_x * det_x + det_y * det_y + det_z * det_z)

        dot_val = -dx * det_x[None, :] - dy * det_y[None, :] - dz * det_z[None, :]
        denom = vector_n0[None, :] * d

        angle_cos = dot_val / denom
        angle_cos = tl.maximum(angle_cos, 0.0)

        idx_float = d / vs * fs
        idx = idx_float.to(tl.int32)

        if METHOD == 0:
            valid = (
                voxel_mask[:, None]
                & det_mask[None, :]
                & (idx >= 0)
                & (idx < NUM_TIMES)
                & (d2 > 0.0)
                & (denom > 0.0)
            )

            signal_val = tl.load(
                signal_ptr + det_idx[None, :] * NUM_TIMES + idx,
                mask=valid,
                other=0.0,
            )

            contrib = signal_val * angle_cos / d2

        else:
            valid = (
                voxel_mask[:, None]
                & det_mask[None, :]
                & (idx >= 0)
                & (idx < NUM_TIMES - 2)
                & (d2 > 0.0)
                & (denom > 0.0)
            )

            signal_0 = tl.load(
                signal_ptr + det_idx[None, :] * NUM_TIMES + idx,
                mask=valid,
                other=0.0,
            )

            signal_1 = tl.load(
                signal_ptr + det_idx[None, :] * NUM_TIMES + idx + 1,
                mask=valid,
                other=0.0,
            )

            derivative = signal_1 - signal_0
            contrib = (signal_0 - idx_float * derivative) * angle_cos / d2

        contrib = tl.where(valid, contrib, 0.0)
        acc += tl.sum(contrib, axis=1)

    tl.store(output_ptr + voxel_offsets, acc, mask=voxel_mask)

# ==========================================
# 4. 读取并拼接多组 txt 数据
# ==========================================
print(f"Loading and fusing {len(signal_files)} data pairs...")
start_load = time.time()

all_signals = []
all_locations = []

for sig_file, loc_file in zip(signal_files, location_files):
    signal_txt_path = os.path.join(input_dir, sig_file)
    location_txt_path = os.path.join(input_dir, loc_file)

    temp_signal = np.loadtxt(signal_txt_path, dtype=np.float32)
    temp_location = np.loadtxt(location_txt_path, dtype=np.float32)

    if temp_signal.ndim == 1:
        temp_signal = temp_signal[None, :]

    if temp_location.ndim == 1:
        temp_location = temp_location[None, :]

    if temp_signal.shape[1] != num_times:
        raise ValueError(
            f"{sig_file} 的时间采样点数为 {temp_signal.shape[1]}，但 num_times={num_times}"
        )

    if temp_location.shape[1] != 3:
        raise ValueError(
            f"{loc_file} 的位置列数为 {temp_location.shape[1]}，应为 3 列 x/y/z"
        )

    if temp_signal.shape[0] != temp_location.shape[0]:
        raise ValueError(
            f"{sig_file} 和 {loc_file} 的探测器数量不一致："
            f"{temp_signal.shape[0]} vs {temp_location.shape[0]}"
        )

    all_signals.append(temp_signal)
    all_locations.append(temp_location)

    print(f"  - Loaded: {sig_file} (Detectors: {temp_location.shape[0]})")

real_signal = np.ascontiguousarray(np.vstack(all_signals), dtype=np.float32)
sensor_location = np.ascontiguousarray(np.vstack(all_locations), dtype=np.float32)

num_detectors = int(sensor_location.shape[0])

print(f"Data loading and fusion finished in {time.time() - start_load:.2f} seconds.")
print(
    f"==> Equivalent DENSE Array Detectors: {num_detectors}, "
    f"Fused Signal shape: {real_signal.shape}"
)

# ==========================================
# 5. 拷贝到 GPU
# ==========================================
if not torch.cuda.is_available():
    raise RuntimeError("当前环境没有可用 CUDA GPU，Triton 版本需要 CUDA。")

device = torch.device("cuda:0")
print(f"[INIT] 当前进程可见设备: {device}, physical CUDA_VISIBLE_DEVICES={args.device_id}")

signal_gpu = torch.from_numpy(real_signal).to(device=device, dtype=torch.float32).contiguous()
location_gpu = torch.from_numpy(sensor_location).to(device=device, dtype=torch.float32).contiguous()

output_gpu = torch.empty((num_voxels,), device=device, dtype=torch.float32)

# ==========================================
# 6. 执行 Triton 重建
# ==========================================
if recon_method == "DAS":
    method_id = 0
elif recon_method == "FBP":
    method_id = 1
else:
    raise ValueError("Invalid recon_method. Please choose 'DAS' or 'FBP'.")

print(
    f"Starting [{recon_method}] reconstruction on Grid: "
    f"{num_x}x{num_y}x{num_z} using Triton CUDA..."
)

start_recon = time.time()

BLOCK_VOXELS = 128
BLOCK_DETECTORS = 32

grid = (triton.cdiv(num_voxels, BLOCK_VOXELS),)

recon_kernel_triton[grid](
    signal_gpu,
    location_gpu,
    output_gpu,
    float(x_start),
    float(y_start),
    float(z_start),
    float(res),
    float(vs),
    float(fs),
    int(num_x),
    int(num_y),
    int(num_z),
    int(num_detectors),
    int(num_times),
    int(method_id),
    BLOCK_VOXELS=BLOCK_VOXELS,
    BLOCK_DETECTORS=BLOCK_DETECTORS,
    num_warps=4,
)

torch.cuda.synchronize()

end_recon = time.time()
print("Reconstruction time: {:.2f}s".format(end_recon - start_recon))

# ==========================================
# 7. 保存结果
# ==========================================
signal_recon = output_gpu.detach().cpu().numpy().reshape(
    int(num_x), int(num_y), int(num_z)
)

signal_recon_abs = np.abs(signal_recon).astype(np.float32)

num_fused = len(signal_files)
save_filename = f"{output_file_prefix}_{recon_method}_fused_{num_fused}files_triton.mat"
save_path = os.path.join(output_dir, save_filename)

os.makedirs(output_dir, exist_ok=True)

sio.savemat(save_path, {"signal_recon": signal_recon_abs})

print(f"Successfully saved reconstructed volume to {save_path}")
