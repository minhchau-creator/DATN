#!/usr/bin/env python3
"""
VRAM Monitor — Bảo vệ tiến trình liver training.

Theo dõi VRAM GPU liên tục. Nếu VRAM tự do xuống dưới ngưỡng,
tự động gửi SIGTERM để dừng tiến trình DATN training (train_models.py)
một cách an toàn (lưu checkpoint rồi thoát).

Chạy:
    python monitor_vram.py                        # mặc định
    python monitor_vram.py --threshold 4096       # ngưỡng 4 GB
    python monitor_vram.py --interval 5           # kiểm tra mỗi 5 giây
"""

import os
import sys
import time
import signal
import logging
import argparse
import subprocess
from pathlib import Path

# ── Cấu hình mặc định ─────────────────────────────────────────────────────────
FREE_VRAM_THRESHOLD_MIB = 3072   # Dưới mức này → dừng DATN (3 GB)
CHECK_INTERVAL_SEC      = 10     # Kiểm tra mỗi N giây
TRAIN_SCRIPT_NAME       = "train_models.py"
GPU_INDEX               = 0
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("vram_monitor.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

try:
    import pynvml
    _PYNVML = True
    pynvml.nvmlInit()
except ImportError:
    _PYNVML = False

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


# ── GPU helpers ───────────────────────────────────────────────────────────────

def _nvml_handle():
    return pynvml.nvmlDeviceGetHandleByIndex(GPU_INDEX)


def get_vram_info() -> tuple[int, int]:
    """Trả về (total_mib, free_mib) của GPU."""
    if _PYNVML:
        info = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle())
        return info.total // (1024 * 1024), info.free // (1024 * 1024)
    result = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=memory.total,memory.free",
         "--format=csv,noheader,nounits",
         f"--id={GPU_INDEX}"],
        capture_output=True, text=True, check=True,
    )
    total, free = result.stdout.strip().split(",")
    return int(total.strip()), int(free.strip())


def get_gpu_processes() -> dict[int, int]:
    """Trả về {pid: used_mib} cho tất cả process đang dùng GPU."""
    if _PYNVML:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(_nvml_handle())
        return {p.pid: p.usedGpuMemory // (1024 * 1024) for p in procs}
    result = subprocess.run(
        ["nvidia-smi",
         "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits",
         f"--id={GPU_INDEX}"],
        capture_output=True, text=True, check=True,
    )
    out: dict[int, int] = {}
    for line in result.stdout.strip().splitlines():
        if line.strip():
            pid_s, mem_s = line.split(",")
            out[int(pid_s.strip())] = int(mem_s.strip())
    return out


# ── Process helpers ───────────────────────────────────────────────────────────

def find_datn_pid() -> int | None:
    """Tìm PID của tiến trình train_models.py đang chạy."""
    if _PSUTIL:
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = " ".join(proc.info["cmdline"] or [])
                if TRAIN_SCRIPT_NAME in cmdline:
                    return proc.info["pid"]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None

    result = subprocess.run(
        ["pgrep", "-f", TRAIN_SCRIPT_NAME],
        capture_output=True, text=True,
    )
    pids = [p for p in result.stdout.strip().splitlines() if p]
    return int(pids[0]) if pids else None


def send_sigterm(pid: int) -> bool:
    """Gửi SIGTERM đến pid. Trả về True nếu thành công."""
    try:
        os.kill(pid, signal.SIGTERM)
        logger.warning(f"SIGTERM → PID {pid} ({TRAIN_SCRIPT_NAME})")
        return True
    except ProcessLookupError:
        logger.info(f"PID {pid} không còn tồn tại.")
    except PermissionError:
        logger.error(f"Không có quyền gửi signal đến PID {pid}.")
    return False


# ── Monitor loop ──────────────────────────────────────────────────────────────

def monitor(threshold_mib: int, interval: int) -> None:
    total_mib, _ = get_vram_info()
    logger.info(
        f"VRAM Monitor khởi động | GPU {GPU_INDEX} | "
        f"Tổng VRAM: {total_mib} MiB | "
        f"Ngưỡng dừng: {threshold_mib} MiB tự do | "
        f"Kiểm tra mỗi {interval}s"
    )
    if not _PYNVML:
        logger.info("(pynvml không có — dùng nvidia-smi)")
    if not _PSUTIL:
        logger.info("(psutil không có — dùng pgrep)")

    terminated_pid: int | None = None

    while True:
        try:
            _, free_mib = get_vram_info()
            gpu_procs   = get_gpu_processes()
            datn_pid    = find_datn_pid()
            datn_vram   = gpu_procs.get(datn_pid, 0) if datn_pid else 0
            used_mib    = total_mib - free_mib

            logger.info(
                f"VRAM: {used_mib}/{total_mib} MiB dùng | "
                f"Tự do: {free_mib} MiB | "
                f"DATN PID: {datn_pid or '—'} ({datn_vram} MiB) | "
                f"Ngưỡng: {threshold_mib} MiB"
            )

            if datn_pid and datn_pid != terminated_pid:
                if free_mib < threshold_mib:
                    logger.warning(
                        f"⚠  VRAM tự do ({free_mib} MiB) < ngưỡng ({threshold_mib} MiB)! "
                        f"Đang dừng DATN training (PID {datn_pid})..."
                    )
                    if send_sigterm(datn_pid):
                        terminated_pid = datn_pid
                        logger.info("   DATN sẽ lưu checkpoint và thoát an toàn.")
                        logger.info("   Chạy lại train_models.py sau khi liver training xong.")

            elif not datn_pid and terminated_pid:
                logger.info(
                    f"DATN training (PID {terminated_pid}) đã dừng hẳn. "
                    f"VRAM tự do hiện tại: {free_mib} MiB."
                )
                terminated_pid = None

        except Exception as exc:
            logger.error(f"Lỗi khi đọc VRAM: {exc}")

        time.sleep(interval)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitor VRAM, tự động dừng DATN training khi sắp hết bộ nhớ GPU"
    )
    parser.add_argument(
        "--threshold", type=int, default=FREE_VRAM_THRESHOLD_MIB,
        help=f"Ngưỡng VRAM tự do (MiB) để dừng DATN training. Mặc định: {FREE_VRAM_THRESHOLD_MIB}",
    )
    parser.add_argument(
        "--interval", type=int, default=CHECK_INTERVAL_SEC,
        help=f"Chu kỳ kiểm tra (giây). Mặc định: {CHECK_INTERVAL_SEC}",
    )
    args = parser.parse_args()

    try:
        monitor(args.threshold, args.interval)
    except KeyboardInterrupt:
        logger.info("Monitor dừng (Ctrl+C).")
