#!/usr/bin/env python3
"""vllmtop — top-like TUI for a running vLLM server.

Read-only side-car: scrapes /metrics, nvidia-smi, and `docker logs -f`.
No restart of the target. Stdlib only.
"""

import argparse
import collections
import ctypes
import curses
import glob
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional


VERSION = "0.1"
DEFAULT_URL = "http://localhost:8000"


# ───────────────────────────────────────────────────────────── HTTP ──

def http_get(url, timeout=2.0):
    req = urllib.request.Request(url, headers={"User-Agent": f"vllmtop/{VERSION}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def normalize_url(s):
    s = s.strip()
    if not s:
        return s
    if not s.startswith(("http://", "https://")):
        s = "http://" + s
    return s.rstrip("/")


def is_local_host(host):
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return True
    try:
        local_ips = {info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None)}
        local_ips.update({"127.0.0.1", "::1"})
        return host in local_ips
    except OSError:
        return False


# ───────────────────────────────────────── Prometheus exposition parser ──

PROM_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{([^}]*)\})?\s+([0-9.eE+\-NainfINF]+)')
PROM_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"')


def parse_prom_labels(s):
    if not s:
        return {}
    return {m.group(1): m.group(2) for m in PROM_LABEL.finditer(s)}


def parse_prom_text(text):
    """Return dict: metric_name -> list[(labels_dict, value_float)]."""
    out = collections.defaultdict(list)
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] == "#":
            continue
        m = PROM_LINE.match(line)
        if not m:
            continue
        try:
            val = float(m.group(4))
        except ValueError:
            continue
        out[m.group(1)].append((parse_prom_labels(m.group(3)), val))
    return out


def m_sum(metrics, name):
    return sum(v for _, v in metrics.get(name, []))


def m_max(metrics, name):
    return max((v for _, v in metrics.get(name, [])), default=0.0)


def m_by_label(metrics, name, label_key):
    """Return dict[label_value(str) -> summed float] for rows of `name` carrying `label_key`."""
    out = {}
    for labels, val in metrics.get(name, []):
        k = labels.get(label_key)
        if k is None:
            continue
        out[k] = out.get(k, 0.0) + val
    return out


def m_avg(metrics, name):
    rows = metrics.get(name, [])
    if not rows:
        return 0.0
    return sum(v for _, v in rows) / len(rows)


def m_sum_filter(metrics, name, label, value):
    return sum(v for labs, v in metrics.get(name, []) if labs.get(label) == value)


# ─────────────────────────────────────────────────── histograms ──

def collect_histogram(metrics, base_name):
    """Sum cumulative buckets across engines: returns {le_float: count} or None."""
    rows = metrics.get(base_name + "_bucket")
    if not rows:
        return None
    out = collections.defaultdict(float)
    for labels, val in rows:
        le = labels.get("le")
        if le is None:
            continue
        try:
            le_f = float("inf") if le in ("+Inf", "Inf") else float(le)
        except ValueError:
            continue
        out[le_f] += val
    if not out:
        return None
    return dict(sorted(out.items()))


def histogram_diff(now, then):
    if not now:
        return None
    if not then:
        return dict(now)
    return {le: cnt - then.get(le, 0.0) for le, cnt in now.items()}


def histogram_total(b):
    if not b:
        return 0.0
    return max(b.values()) if b else 0.0


def percentile_from_buckets(buckets, q):
    """Linear-interpolate the q-quantile from cumulative {le: count}."""
    if not buckets:
        return None
    items = sorted(buckets.items())
    total = items[-1][1]
    if total <= 0:
        return None
    target = q * total
    prev_le = 0.0
    prev_count = 0.0
    for le, cnt in items:
        if cnt >= target:
            if le == float("inf"):
                return prev_le
            if cnt == prev_count:
                return le
            frac = (target - prev_count) / (cnt - prev_count)
            return prev_le + frac * (le - prev_le)
        prev_le = le
        prev_count = cnt
    return items[-1][0] if items[-1][0] != float("inf") else prev_le


# ─────────────────────────────────────────────────── nvidia-smi ──

NVIDIA_SMI = shutil.which("nvidia-smi")


def _smi_run(args, timeout=0.8):
    if not NVIDIA_SMI:
        return None
    try:
        cp = subprocess.run([NVIDIA_SMI] + args,
                            capture_output=True, text=True, timeout=timeout)
        if cp.returncode != 0:
            return None
        return cp.stdout
    except (subprocess.TimeoutExpired, OSError):
        return None


def nvidia_query_gpu():
    out = _smi_run([
        "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,"
        "temperature.gpu,power.draw,clocks.current.sm,"
        "pcie.link.gen.current,pcie.link.width.current",
        "--format=csv,noheader,nounits",
    ])
    if out is None:
        return None
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 11:
            continue
        def _f(s, default=0.0):
            try:
                return float(s)
            except (ValueError, TypeError):
                return default
        def _i(s, default=0):
            try:
                return int(s)
            except (ValueError, TypeError):
                return default
        rows.append({
            "index": _i(parts[0]),
            "uuid": parts[1],
            "name": parts[2],
            "util": _f(parts[3]),
            "mem_used": _f(parts[4]),
            "mem_total": _f(parts[5]),
            "temp": _f(parts[6]),
            "power": _f(parts[7]),
            "clock": _i(parts[8]),
            "pcie_gen": _i(parts[9]),
            "pcie_width": _i(parts[10]),
        })
    return rows


def nvidia_query_compute_apps():
    out = _smi_run([
        "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
        "--format=csv,noheader,nounits",
    ])
    if out is None:
        return None
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            rows.append({
                "pid": int(parts[0]),
                "name": parts[1],
                "mem": float(parts[2]),
                "gpu_uuid": parts[3],
            })
        except (ValueError, IndexError):
            continue
    return rows


# ─────────────────────────────────────────────────── PCIe ──
#
# vllmtop is a passive monitor, so we cannot measure true cudaMemcpy round-trip
# latency. We surface three proxies that move in lockstep with it:
#   • Rx/Tx MB/s          (live throughput, tail of `nvidia-smi dmon -s te`)
#   • saturation %         (max(rx,tx) / theoretical link bw — queuing-delay proxy)
#   • PCIe replays /s      (`pci` column from dmon — every replay = a latency tax)
#   • AER correctable Δ    (cumulative `aer_dev_correctable` delta — link health)

# Per-lane per-direction throughput in MB/s. Gen3+ uses 128b/130b encoding.
_PCIE_LANE_MBPS = {
    1: 250.0,    # Gen1: 2.5 GT/s · 8b/10b
    2: 500.0,    # Gen2: 5.0 GT/s · 8b/10b
    3: 985.0,    # Gen3: 8.0 GT/s · 128b/130b
    4: 1969.0,   # Gen4: 16  GT/s · 128b/130b
    5: 3938.0,   # Gen5: 32  GT/s · 128b/130b
    6: 7563.0,   # Gen6: 64  GT/s · PAM4 + FLIT
}


def pcie_theoretical_mbps(gen, width):
    """Per-direction theoretical bandwidth in MB/s for a PCIe gen×width link."""
    if not gen or not width:
        return 0.0
    return _PCIE_LANE_MBPS.get(int(gen), 0.0) * int(width)


def read_aer_correctable_total(bdf):
    """Sum AER correctable error counts for one PCI device, or None if unreadable.

    File format: lines of '<ErrName> <count>'. Readable as a normal user on
    modern kernels (since the AER attributes are mode 0444).
    """
    try:
        with open(f"/sys/bus/pci/devices/{bdf}/aer_dev_correctable") as f:
            total = 0
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        total += int(parts[1])
                    except ValueError:
                        continue
            return total
    except OSError:
        return None


def discover_gpu_bdfs():
    """Return {gpu_index: 'DDDD:BB:DD.F'} mapping suitable for /sys/bus/pci/devices/."""
    out = _smi_run([
        "--query-gpu=index,pci.bus_id",
        "--format=csv,noheader",
    ])
    if not out:
        return {}
    bdfs = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        # nvidia-smi reports '00000000:04:00.0' — sysfs uses '0000:04:00.0'.
        bid = parts[1].lower()
        if len(bid) > 12 and bid.startswith("0000"):
            bid = bid[4:]
        bdfs[idx] = bid
    return bdfs


# NVML constants we care about
_NVML_PCIE_UTIL_TX_BYTES = 0   # nvmlPcieUtilCounter_t
_NVML_PCIE_UTIL_RX_BYTES = 1
_NVML_SUCCESS = 0


class NvmlPcieSampler:
    """Sub-tick PCIe throughput sampler via libnvidia-ml.so (no nvidia-smi fork).

    Each `nvmlDeviceGetPcieThroughput` call returns a KB/s value averaged over
    the last 20 ms and blocks for ~20 ms while it collects the window. By
    polling in a tight background loop we get one fresh 20-ms-window sample
    per GPU per direction roughly every 80 ms (4 calls × 20 ms = 80 ms for two
    GPUs), i.e. ~6 samples per GPU per 500 ms tick. Peak across those samples
    is what queuing-latency-proxy `sat` is computed against.
    """

    def __init__(self):
        self._rings = {}                     # idx -> deque[(ts, rx_kbps, tx_kbps)]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._lib = None
        self._handles = []                   # list of (idx, ctypes.c_void_p handle)
        self.status = "init"

    # ── public API ────────────────────────────────────────────────
    def start(self):
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
        except OSError as e:
            self.status = f"libnvidia-ml not found: {e}"
            return
        # Declare prototypes (defensive; argtypes prevents pointer truncation on 64-bit).
        lib.nvmlInit_v2.restype = ctypes.c_int
        lib.nvmlShutdown.restype = ctypes.c_int
        lib.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        lib.nvmlDeviceGetCount_v2.restype = ctypes.c_int
        lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [
            ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        lib.nvmlDeviceGetPcieThroughput.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
        lib.nvmlDeviceGetPcieThroughput.restype = ctypes.c_int

        rc = lib.nvmlInit_v2()
        if rc != _NVML_SUCCESS:
            self.status = f"nvmlInit_v2 rc={rc}"
            return
        count = ctypes.c_uint()
        if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != _NVML_SUCCESS:
            lib.nvmlShutdown()
            self.status = "nvmlDeviceGetCount failed"
            return
        for i in range(count.value):
            h = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) != _NVML_SUCCESS:
                continue
            self._handles.append((i, h))
            # Ring of ~5 s at typical sample cadence; sub-tick queries pull from the tail.
            self._rings[i] = collections.deque(maxlen=128)
        self._lib = lib
        self._thread = threading.Thread(target=self._run, daemon=True, name="nvml-pcie")
        self._thread.start()
        self.status = f"active ({len(self._handles)} GPUs)"

    def stop(self):
        self._stop.set()
        # Don't call nvmlShutdown here — _run holds the lib; let the thread exit cleanly.

    def window_stats(self, idx, window_s):
        """Return dict {rx_peak, rx_mean, rx_last, tx_peak, tx_mean, tx_last,
        n_samples} in MB/s for the given GPU index over the last window_s seconds.
        Returns None if we have no samples for this GPU.
        """
        with self._lock:
            ring = self._rings.get(idx)
            if not ring:
                return None
            now = time.time()
            cutoff = now - window_s
            recent = [s for s in ring if s[0] >= cutoff]
            if not recent:
                recent = [ring[-1]]
        rxs = [s[1] for s in recent]
        txs = [s[2] for s in recent]
        return {
            "rx_peak": max(rxs) / 1000.0,
            "rx_mean": sum(rxs) / len(rxs) / 1000.0,
            "rx_last": recent[-1][1] / 1000.0,
            "tx_peak": max(txs) / 1000.0,
            "tx_mean": sum(txs) / len(txs) / 1000.0,
            "tx_last": recent[-1][2] / 1000.0,
            "n_samples": len(recent),
        }

    # ── sampler thread ────────────────────────────────────────────
    def _run(self):
        tx_val = ctypes.c_uint()
        rx_val = ctypes.c_uint()
        try:
            while not self._stop.is_set():
                for idx, h in self._handles:
                    rc = self._lib.nvmlDeviceGetPcieThroughput(
                        h, _NVML_PCIE_UTIL_TX_BYTES, ctypes.byref(tx_val))
                    tx_kbps = tx_val.value if rc == _NVML_SUCCESS else 0
                    rc = self._lib.nvmlDeviceGetPcieThroughput(
                        h, _NVML_PCIE_UTIL_RX_BYTES, ctypes.byref(rx_val))
                    rx_kbps = rx_val.value if rc == _NVML_SUCCESS else 0
                    with self._lock:
                        self._rings[idx].append((time.time(), rx_kbps, tx_kbps))
                # No sleep needed: each nvml call already blocks ~20 ms.
        except Exception:
            pass
        finally:
            try:
                self._lib.nvmlShutdown()
            except Exception:
                pass
            self.status = "stopped"


# ─────────────────────────────────────────────────── CPU temp ──

SENSORS_BIN = shutil.which("sensors")


def _read_hwmon_cpu_temp():
    """Read CPU temperature from /sys/class/hwmon/hwmon*/temp*_input."""
    cpu_drivers = {"k10temp", "coretemp", "cpu_thermal", "zenpower"}
    temps = []
    for hwmon in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            with open(hwmon + "/name") as f:
                devname = f.read().strip().lower()
        except OSError:
            continue
        if devname not in cpu_drivers:
            continue
        for temp_path in glob.glob(hwmon + "/temp*_input"):
            try:
                with open(temp_path) as f:
                    raw = f.read().strip()
                temps.append(int(raw) / 1e3)
            except (OSError, ValueError):
                continue
    return temps if temps else None


def _read_sysfs_cpu_temp():
    """Read CPU temperature from /sys/class/thermal/thermal_zone*/temp."""
    temps = []
    for zone_path in glob.glob("/sys/class/thermal/thermal_zone*"):
        type_path = zone_path + "/type"
        temp_path = zone_path + "/temp"
        try:
            with open(type_path) as f:
                ztype = f.read().strip().lower()
            if "cpu" not in ztype and "x86" not in ztype:
                continue
            with open(temp_path) as f:
                raw = f.read().strip()
            temps.append(int(raw) / 1e3)
        except (OSError, ValueError):
            continue
    return temps if temps else None


def _read_sensors_cpu_temp():
    """Read CPU temperature via the `sensors` command."""
    if not SENSORS_BIN:
        return None
    try:
        cp = subprocess.run([SENSORS_BIN], capture_output=True, text=True, timeout=1.0)
        if cp.returncode != 0:
            return None
    except (subprocess.TimeoutExpired, OSError):
        return None
    temps = []
    for line in cp.stdout.splitlines():
        low = line.lower()
        if "core" in low or "cpu" in low:
            m = re.search(r"([+-]?\d+\.?\d*)\s*[°\xb0]?\s*c", low)
            if m:
                try:
                    temps.append(float(m.group(1)))
                except ValueError:
                    pass
    return temps if temps else None


def get_cpu_temps():
    """Return list of CPU core temperatures (°C), or None if unavailable."""
    temps = _read_hwmon_cpu_temp()
    if temps is not None:
        return temps
    temps = _read_sysfs_cpu_temp()
    if temps is not None:
        return temps
    temps = _read_sensors_cpu_temp()
    return temps


# ─────────────────────────────────────────────────── docker ──

DOCKER = shutil.which("docker")


def _docker_ps(filter_args=None):
    if not DOCKER:
        return None
    args = [DOCKER, "ps", "--format", "{{.Names}}\t{{.Image}}"]
    if filter_args:
        for f in filter_args:
            args += ["--filter", f]
    try:
        cp = subprocess.run(args, capture_output=True, text=True, timeout=2.0)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if cp.returncode != 0:
        return None
    rows = []
    for line in cp.stdout.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        rows.append({"name": parts[0], "image": parts[1]})
    return rows


def autodetect_container(url):
    """Return (name|None, status_message). Empty message means clean match."""
    if not DOCKER:
        return None, "docker unavailable"
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not is_local_host(host):
        return None, "remote endpoint — log pane disabled"

    rows = _docker_ps([f"publish={port}"])
    if rows is None:
        return None, "docker permission denied"

    def _narrow_vllm(rs):
        return [r for r in rs if "vllm" in r["name"].lower() or "vllm" in r["image"].lower()]

    if len(rows) == 1:
        return rows[0]["name"], ""
    if len(rows) > 1:
        narrowed = _narrow_vllm(rows)
        if len(narrowed) == 1:
            return narrowed[0]["name"], ""
        pool = narrowed or rows
        extra = len(pool) - 1
        return pool[0]["name"], (f"+{extra} more, press c" if extra else "")

    all_rows = _docker_ps()
    if all_rows is None:
        return None, "docker permission denied"
    matches = _narrow_vllm(all_rows)
    if len(matches) == 1:
        return matches[0]["name"], ""
    if len(matches) > 1:
        return matches[0]["name"], f"+{len(matches)-1} more, press c"
    return None, f"no vllm container on :{port}"


# ─────────────────────────────────────────────────── log tailer ──

ENGINE_RE = re.compile(
    r"Engine\s+\d+:\s*Avg prompt throughput:\s*([\d.]+)\s*tokens/s.*?"
    r"generation throughput:\s*([\d.]+)\s*tokens/s.*?"
    r"Running:\s*(\d+).*?"
    r"Waiting:\s*(\d+).*?"
    r"GPU KV cache usage:\s*([\d.]+)%.*?"
    r"Prefix cache hit rate:\s*([\d.]+)%"
)
SPECDEC_RE = re.compile(
    r"SpecDecoding metrics:\s*Mean acceptance length:\s*([\d.]+).*?"
    r"Accepted throughput:\s*([\d.]+)\s*tokens/s.*?"
    r"Drafted throughput:\s*([\d.]+)\s*tokens/s.*?"
    r"Per-position acceptance rate:\s*([\d.,\s]+?)(?:,\s*Avg|$)"
)



class LogRotator:
    """Manages log file rotation based on model changes."""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.current_model = None
        self.active_file = None
        self.lock = threading.Lock()
        self._ensure_log_dir()
    
    def _ensure_log_dir(self):
        """Create log directory if it doesn't exist."""
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except OSError as e:
            pass  # Silently ignore errors
    
    def _close_file(self):
        """Internal: close the current log file without acquiring lock."""
        if self.active_file:
            try:
                self.active_file.close()
            except OSError:
                pass
            self.active_file = None
    
    def set_model(self, model_id):
        """Rotate to a new model if model_id changed."""
        with self.lock:
            if model_id == self.current_model:
                return
            
            # Close current file if open
            self._close_file()
            
            self.current_model = model_id
            if not model_id:
                return
            
            # Open new log file
            log_filename = os.path.join(self.log_dir, f"{model_id}.log")
            try:
                self.active_file = open(log_filename, 'a', buffering=1)  # line buffered
            except OSError:
                self.active_file = None
    
    def write(self, line):
        """Write a line to the current model's log file."""
        with self.lock:
            if self.active_file:
                try:
                    self.active_file.write(line)
                    self.active_file.flush()
                except OSError:
                    pass
    
    def close(self):
        """Close the current log file."""
        with self.lock:
            self._close_file()
    
    @property
    def status(self):
        """Return current status string."""
        if self.current_model:
            if self.active_file:
                return "tailing"
            return "rotated"
        return "idle"

class LogTailer:
    def __init__(self, container, buffer, log_rotator=None):
        self.container = container
        self.buffer = buffer
        self.log_rotator = log_rotator
        self.stop_flag = threading.Event()
        self.thread = None
        self.proc = None
        self.dead = False
        self.status = "starting"

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True, name="logtailer")
        self.thread.start()

    def stop(self):
        self.stop_flag.set()
        if self.proc:
            try:
                self.proc.terminate()
            except OSError:
                pass

    def _run(self):
        if not DOCKER:
            self.status = "docker unavailable"
            self.dead = True
            return
        attempts = 0
        while not self.stop_flag.is_set() and attempts < 3:
            try:
                self.proc = subprocess.Popen(
                    [DOCKER, "logs", "-f", "--tail", "0", self.container],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except OSError as e:
                self.status = f"docker logs failed: {e}"
                attempts += 1
                time.sleep(1.0)
                continue
            self.status = "tailing"
            for line in self.proc.stdout:
                if self.stop_flag.is_set():
                    break
                self._parse_line(line)
            rc = self.proc.wait() if self.proc else -1
            if self.stop_flag.is_set():
                return
            attempts += 1
            self.status = f"tailer exited rc={rc}, restart {attempts}/3"
            time.sleep(0.5)
        self.dead = True
        self.status = "log tailer dead"

    def _parse_line(self, line):
        ts = time.strftime("%H:%M:%S")
        # Write to log rotator if available
        if self.log_rotator:
            self.log_rotator.write(line)
        m = ENGINE_RE.search(line)
        if m:
            self.buffer.append({
                "ts": ts, "kind": "engine",
                "prompt_tps": float(m.group(1)),
                "gen_tps": float(m.group(2)),
                "running": int(m.group(3)),
                "waiting": int(m.group(4)),
                "kv": float(m.group(5)),
                "prefix_hit": float(m.group(6)),
            })
            return
        m = SPECDEC_RE.search(line)
        if m:
            per_pos = [float(x.strip()) for x in m.group(4).split(",") if x.strip()]
            self.buffer.append({
                "ts": ts, "kind": "spec",
                "alpha": float(m.group(1)),
                "acc_tps": float(m.group(2)),
                "draft_tps": float(m.group(3)),
                "per_pos": per_pos,
            })


# ─────────────────────────────────────────────────── state / sampling ──

@dataclass
class Sample:
    ts: float
    process_start_time: float = 0.0
    metrics: dict = field(default_factory=dict)
    health_ok: Optional[bool] = None
    model_id: str = ""
    max_ctx: int = 0
    engines: int = 1


# Counter metrics we display rates for.
COUNTERS = [
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:num_preemptions_total",
]
HISTOGRAMS = [
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
]


@dataclass
class State:
    url: str
    container: Optional[str]
    container_status: str
    tick: float
    debug_log: Optional[str]
    no_logs: bool

    last_sample: Optional[Sample] = None
    last_ok_ts: float = 0.0
    counter_history: dict = field(default_factory=dict)
    hist_history: dict = field(default_factory=dict)

    gpu_rows: list = field(default_factory=list)
    gpu_apps: list = field(default_factory=list)
    gpu_stale_ts: float = 0.0

    pcie_sampler: Optional["NvmlPcieSampler"] = None
    gpu_bdfs: dict = field(default_factory=dict)
    pcie_aer_baseline: dict = field(default_factory=dict)

    log_buf: collections.deque = field(default_factory=lambda: collections.deque(maxlen=200))
    log_rotator: Optional["LogRotator"] = None
    current_model: str = ""
    log_tailer: Optional[LogTailer] = None

    # Time-series buffers for graphs (one float per tick; capped at 240 samples = 4 min @ 1 s).
    graph_history: dict = field(default_factory=dict)

    show_cumulative_percentiles: bool = False
    err_msg: str = ""
    start_ts: float = field(default_factory=time.time)
    gen_tps_global_max: float = 0.0
    gpu_clock_global_max: float = 0.0

    def push_graph(self, name, value):
        dq = self.graph_history.setdefault(name, collections.deque(maxlen=240))
        dq.append(float(value))

    def push_counter(self, name, value, now):
        dq = self.counter_history.setdefault(name, collections.deque())
        dq.append((now, value))
        cutoff = now - 15.0
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def push_histogram(self, name, buckets, now):
        if buckets is None:
            return
        dq = self.hist_history.setdefault(name, collections.deque())
        dq.append((now, buckets))
        cutoff = now - 12.0
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def rate_1s(self, name):
        dq = self.counter_history.get(name)
        if not dq or len(dq) < 2:
            return None
        (t0, v0), (t1, v1) = dq[-2], dq[-1]
        dt = t1 - t0
        if dt <= 0:
            return None
        return max(0.0, (v1 - v0) / dt)

    def rate_10s(self, name):
        dq = self.counter_history.get(name)
        if not dq or len(dq) < 2:
            return None
        t_now, v_now = dq[-1]
        target = t_now - 10.0
        oldest = None
        for ts, v in dq:
            if ts >= target:
                oldest = (ts, v)
                break
        if oldest is None or oldest is dq[-1]:
            return None
        t0, v0 = oldest
        dt = t_now - t0
        if dt < 2.0:
            return None
        return max(0.0, (v_now - v0) / dt)

    def windowed_hist(self, name):
        dq = self.hist_history.get(name)
        if not dq or len(dq) < 2:
            return None, 0.0
        t_now, b_now = dq[-1]
        target = t_now - 10.0
        oldest = None
        for ts, b in dq:
            if ts >= target:
                oldest = (ts, b)
                break
        if oldest is None or oldest is dq[-1]:
            return None, 0.0
        diff = histogram_diff(b_now, oldest[1])
        return diff, t_now - oldest[0]

    def clear_history(self):
        self.counter_history.clear()
        self.hist_history.clear()
        self.gen_tps_global_max = 0.0
        self.gpu_clock_global_max = 0.0


def fetch_sample(url, timeout=2.0):
    try:
        status, body = http_get(url + "/metrics", timeout=timeout)
        if status != 200:
            return None, f"/metrics HTTP {status}"
        metrics = parse_prom_text(body.decode("utf-8", errors="replace"))
    except Exception as e:
        return None, str(e).split("\n")[0][:80]

    health_ok = None
    model_id = ""
    max_ctx = 0
    try:
        st, _ = http_get(url + "/health", timeout=1.0)
        health_ok = (st == 200)
    except Exception:
        health_ok = False
    try:
        st, body = http_get(url + "/v1/models", timeout=1.0)
        if st == 200:
            data = json.loads(body)
            arr = data.get("data") or []
            if arr:
                model_id = str(arr[0].get("id", ""))
                max_ctx = int(arr[0].get("max_model_len") or 0)
    except Exception:
        pass

    pst = 0.0
    if "process_start_time_seconds" in metrics:
        pst = metrics["process_start_time_seconds"][0][1]

    engines = set()
    for rows in metrics.values():
        for labels, _ in rows:
            if "engine" in labels:
                engines.add(labels["engine"])
    n_engines = max(1, len(engines))

    return Sample(
        ts=time.time(),
        process_start_time=pst,
        metrics=metrics,
        health_ok=health_ok,
        model_id=model_id,
        max_ctx=max_ctx,
        engines=n_engines,
    ), ""


def update_state(state: State, sample: Sample):
    # Detect server restart
    prev = state.last_sample
    if prev and sample.process_start_time and prev.process_start_time \
            and sample.process_start_time > prev.process_start_time + 1.0:
        state.clear_history()

    now = sample.ts
    for c in COUNTERS:
        state.push_counter(c, m_sum(sample.metrics, c), now)
    for h in HISTOGRAMS:
        state.push_histogram(h, collect_histogram(sample.metrics, h), now)

    state.last_sample = sample
    state.last_ok_ts = now

    # Graph time-series — one value per sample. rate_1s/rate_10s read from the freshly
    # pushed counter_history above, so they reflect this tick.
    gen_tps = state.rate_1s("vllm:generation_tokens_total") or 0.0
    state.push_graph("gen_tps", gen_tps)
    if gen_tps > state.gen_tps_global_max:
        state.gen_tps_global_max = gen_tps
    state.push_graph("prompt_tps", state.rate_1s("vllm:prompt_tokens_total") or 0.0)
    state.push_graph("kv_pct", m_max(sample.metrics, "vllm:kv_cache_usage_perc") * 100.0)
    state.push_graph("running", m_sum(sample.metrics, "vllm:num_requests_running"))


# ─────────────────────────────────────────────────── formatting ──

def fmt_num(n, unit=""):
    if n is None:
        return "—"
    a = abs(n)
    if a >= 1e9:
        return f"{n/1e9:.2f} G{unit}"
    if a >= 1e6:
        return f"{n/1e6:.2f} M{unit}"
    if a >= 1e3:
        return f"{n/1e3:.1f} k{unit}"
    if a >= 10:
        return f"{n:.0f}{unit}"
    if a >= 1:
        return f"{n:.1f}{unit}"
    return f"{n:.2f}{unit}"


def fmt_rate(n):
    if n is None:
        return "    —"
    if n < 0.05:
        return "   +0"
    if n >= 100000:
        return f"{n/1000:>4.0f}k"
    if n >= 1000:
        return f"{n:>5.0f}"
    if n >= 10:
        return f"{n:>5.1f}"
    return f"{n:>5.2f}"


def fmt_uptime(secs):
    if secs <= 0:
        return "—"
    d, r = divmod(int(secs), 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def fmt_seconds(s):
    if s is None:
        return "—"
    if s < 0.001:
        return f"{s*1e6:.0f}µs"
    if s < 1.0:
        return f"{s*1000:.0f}ms"
    if s < 60:
        return f"{s:.2f}s"
    m, sec = divmod(s, 60)
    return f"{int(m)}m{int(sec):02d}s"


def fmt_pct(x):
    if x is None:
        return "—"
    return f"{x:5.1f}%"


# ─────────────────────────────────────────────────── curses helpers ──

# Color pair indices
CP_OK = 1       # green
CP_WARN = 2     # yellow
CP_BAD = 3      # red
CP_DIM = 4
CP_HEADER = 5   # cyan
CP_PLOT_A = 6   # green   (series 1)
CP_PLOT_B = 7   # cyan    (series 2)
CP_PLOT_C = 8   # yellow  (series 3)
CP_PLOT_D = 9   # magenta (series 4)
CP_PLOT_E = 10  # blue    (series 5)
CP_PLOT_F = 11  # red     (series 6)


def init_colors():
    has_color = curses.has_colors()
    if has_color:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(CP_OK, curses.COLOR_GREEN, -1)
        curses.init_pair(CP_WARN, curses.COLOR_YELLOW, -1)
        curses.init_pair(CP_BAD, curses.COLOR_RED, -1)
        curses.init_pair(CP_DIM, 8 if curses.COLORS >= 16 else curses.COLOR_WHITE, -1)
        curses.init_pair(CP_HEADER, curses.COLOR_CYAN, -1)
        curses.init_pair(CP_PLOT_A, curses.COLOR_GREEN, -1)
        curses.init_pair(CP_PLOT_B, curses.COLOR_CYAN, -1)
        curses.init_pair(CP_PLOT_C, curses.COLOR_YELLOW, -1)
        curses.init_pair(CP_PLOT_D, curses.COLOR_MAGENTA, -1)
        curses.init_pair(CP_PLOT_E, curses.COLOR_BLUE, -1)
        curses.init_pair(CP_PLOT_F, curses.COLOR_RED, -1)
    return has_color


def attr(pair):
    try:
        return curses.color_pair(pair)
    except Exception:
        return 0


def addstr_safe(stdscr, y, x, s, a=0):
    rows, cols = stdscr.getmaxyx()
    if y < 0 or y >= rows or x >= cols:
        return
    if x < 0:
        s = s[-x:]
        x = 0
    max_len = cols - x
    if max_len <= 0:
        return
    if len(s) > max_len:
        s = s[:max_len - 1] + "…" if max_len > 1 else s[:max_len]
    try:
        stdscr.addstr(y, x, s, a)
    except curses.error:
        pass


def draw_kv_bar(width, pct):
    """Return a unicode bar string of given width filled to pct (0-100)."""
    if width < 4:
        return ""
    inner = width - 2
    filled = int(round(inner * max(0.0, min(100.0, pct)) / 100.0))
    return "[" + ("█" * filled) + ("░" * (inner - filled)) + "]"


def _data_level(value, max_val, height):
    """Map data value (0..max_val) to row index 0..height-1, top-anchored (smaller row = higher value)."""
    if max_val <= 0 or height <= 0:
        return height - 1
    v = max(0.0, min(max_val, value))
    # row = height-1 means bottom (zero); row = 0 means top (max).
    lvl = int(round(height - 1 - (v / max_val) * (height - 1)))
    return max(0, min(height - 1, lvl))


def render_line_plot(stdscr, y0, x0, width, height, series, max_val=100.0):
    """nvtop-style line plot: overlay N series as colored line traces.

    series: list of (color_attr, deque-of-values).
    All series share the same Y-axis scaled to max_val (so normalize before passing).
    Uses box-drawing chars (─ │ ┌ ┐ └ ┘) and lets later series overwrite earlier
    ones on collision (nvtop uses junction chars; we keep it simple).
    """
    if width <= 0 or height <= 0:
        return

    for color, dq in series:
        vals = list(dq)[-width:]
        pad = width - len(vals)
        # left-pad with None to keep the latest sample at the rightmost column
        vals = [None] * pad + vals
        prev = None
        for col, v in enumerate(vals):
            if v is None:
                prev = None
                continue
            lvl = _data_level(v, max_val, height)
            if prev is None or prev == lvl:
                addstr_safe(stdscr, y0 + lvl, x0 + col, "─", color)
            else:
                # Smaller row index = higher data value. "drawing_down" on screen
                # means the row number got bigger (value decreased).
                drawing_down = prev < lvl
                if drawing_down:
                    addstr_safe(stdscr, y0 + prev, x0 + col, "┐", color)
                    addstr_safe(stdscr, y0 + lvl, x0 + col, "└", color)
                else:
                    addstr_safe(stdscr, y0 + prev, x0 + col, "┘", color)
                    addstr_safe(stdscr, y0 + lvl, x0 + col, "┌", color)
                lo, hi = min(prev, lvl), max(prev, lvl)
                for r in range(lo + 1, hi):
                    addstr_safe(stdscr, y0 + r, x0 + col, "│", color)
            prev = lvl


def draw_one_plot(stdscr, y, x, width, height, title, series_specs, right_axis=None):
    """Draw one labeled plot inside a (width × height) rectangle starting at (y, x).

    series_specs: list of (label, color_attr, max_val, value_fmt, deque). Each series is
    normalized to 0..100 internally so they share a common Y-axis; value_fmt(last_value)
    returns the human-readable current reading shown in the legend (e.g. "67.4%", "3", "38 t/s").

    right_axis: optional tuple (color_attr, max_val, label) for a secondary right-side Y-axis.
        When provided, a second scale is drawn on the right side of the plot body.
        The right-axis series is still normalized to 0..100 internally, but the right-side
        tick labels reflect the actual scale (0..max_val) in the given color.

    Layout:
        row y      → legend
        rows y+1 … y+height-1 → plot body (with Y-axis labels '100 75 50 25 0' in left margin)
    """
    if height < 4 or width < 18:
        return

    # Top: legend with colored labels "●label NN" per series (nvtop-style)
    col = x
    if title:
        s = f"{title}  "
        addstr_safe(stdscr, y, col, s, attr(CP_HEADER) | curses.A_BOLD)
        col += len(s)
    for label, color, _mv, vfmt, dq in series_specs:
        last_v = dq[-1] if dq else 0.0
        try:
            val_str = vfmt(last_v)
        except Exception:
            val_str = "—"
        s = f"●{label} {val_str}  "
        addstr_safe(stdscr, y, col, s, color | curses.A_BOLD)
        col += len(s)

    # Reserve margins for Y-axis labels
    label_w = 4   # left: "100", "75 ", "50 ", "25 ", "0  "
    right_axis_w = 0
    if right_axis is not None:
        right_axis_w = 9  # right: " 1000 " + space

    body_y = y + 1
    body_h = height - 1
    body_x = x + label_w + 1
    body_w = width - label_w - 1 - right_axis_w
    if body_w <= 0 or body_h <= 0:
        return

    # Y-axis ticks at 0/25/50/75/100
    for tick in (0, 25, 50, 75, 100):
        lvl = _data_level(tick, 100, body_h)
        addstr_safe(stdscr, body_y + lvl, x, f"{tick:>3d}", attr(CP_DIM))
        addstr_safe(stdscr, body_y + lvl, x + label_w, "┤", attr(CP_DIM))
    # vertical axis on remaining body rows
    for r in range(body_h):
        # only draw │ where we didn't already draw ┤
        addstr_safe(stdscr, body_y + r, x + label_w, "│", attr(CP_DIM))
    # repaint ticks on top of the bare axis
    for tick in (0, 25, 50, 75, 100):
        lvl = _data_level(tick, 100, body_h)
        addstr_safe(stdscr, body_y + lvl, x + label_w, "┤", attr(CP_DIM))

    # Right axis: skeleton (vertical line at right edge)
    if right_axis is not None:
        ra_color, ra_max, ra_label = right_axis
        ra_x = body_x + body_w
        for r in range(body_h):
            addstr_safe(stdscr, body_y + r, ra_x, "│", ra_color)

    # Normalize series to 0..100
    normalized = []
    for _label, color, mv, _vfmt, dq in series_specs:
        if mv <= 0:
            mv = 1.0
        norm = [min(100.0, max(0.0, (v / mv) * 100.0)) for v in dq]
        normalized.append((color, norm))

    render_line_plot(stdscr, body_y, body_x, body_w, body_h, normalized, max_val=100.0)

    # Right axis: tick labels at 0/25/50/75/100 of the secondary scale
    if right_axis is not None:
        ra_color, ra_max, ra_label = right_axis
        ra_x = body_x + body_w
        ra_label_x = ra_x + 1
        for tick in (0, 25, 50, 75, 100):
            lvl = _data_level(tick, 100, body_h)
            raw_val = (tick / 100.0) * ra_max
            if raw_val >= 1000:
                lbl = f"{raw_val/1000:.1f}k"
            elif raw_val >= 100:
                lbl = f"{raw_val:.0f}"
            else:
                lbl = f"{raw_val:.1f}"
            addstr_safe(stdscr, body_y + lvl, ra_label_x, f"{lbl:<8s}", ra_color)
            # redraw junction on top
            addstr_safe(stdscr, body_y + lvl, ra_x, "├", ra_color)


# ─────────────────────────────────────────────────── modal input ──

def curses_input_modal(stdscr, title, prompt, default=""):
    """Show a centered modal asking for a single-line string. Returns input or None on ESC."""
    curses.curs_set(1)
    curses.echo()
    try:
        rows, cols = stdscr.getmaxyx()
        w = max(50, min(cols - 4, 80))
        h = 7
        y = (rows - h) // 2
        x = (cols - w) // 2
        win = curses.newwin(h, w, y, x)
        win.keypad(True)
        win.box()
        win.addstr(0, 2, f" {title} ", curses.A_BOLD)
        win.addstr(2, 2, prompt[: w - 4])
        win.addstr(4, 2, "▸ ")
        if default:
            win.addstr(4, 4, default[: w - 6])
        win.addstr(5, 2, "[Enter to accept, ESC to cancel]"[: w - 4], attr(CP_DIM))
        win.refresh()
        try:
            s = win.getstr(4, 4 + len(default), w - 6 - len(default)).decode("utf-8", "replace")
        except curses.error:
            s = ""
        if s == "":
            s = default
        # ESC -> getstr returns empty too; treat empty as cancel only if default was empty
        return s.strip() or None
    finally:
        curses.curs_set(0)
        curses.noecho()
        stdscr.touchwin()
        stdscr.refresh()


# ─────────────────────────────────────────────────── panels ──

def color_for_health(state, sample):
    if sample is None or sample.health_ok is None:
        return 0
    return attr(CP_OK) if sample.health_ok else attr(CP_BAD)


def color_for_kv(pct):
    if pct >= 85:
        return attr(CP_BAD)
    if pct >= 70:
        return attr(CP_WARN)
    return 0


def color_for_waiting(n):
    if n >= 5:
        return attr(CP_BAD)
    if n >= 1:
        return attr(CP_WARN)
    return attr(CP_OK)


def color_for_temp(t):
    if t >= 85:
        return attr(CP_BAD)
    if t >= 80:
        return attr(CP_WARN)
    return 0


def color_for_mem_pct(pct):
    if pct >= 99:
        return attr(CP_BAD)
    if pct >= 95:
        return attr(CP_WARN)
    return 0


def overall_pill(state, sample):
    if sample is None:
        return "INIT", attr(CP_DIM)
    now = time.time()
    stale = now - state.last_ok_ts if state.last_ok_ts else 0
    if stale >= 30:
        return "BAD", attr(CP_BAD)
    if sample.health_ok is False:
        return "BAD", attr(CP_BAD)
    # Compute the worst color in play
    kv = m_max(sample.metrics, "vllm:kv_cache_usage_perc") * 100.0
    waiting = m_sum(sample.metrics, "vllm:num_requests_waiting")
    bad = (kv >= 85) or (waiting >= 5)
    if bad:
        return "BAD", attr(CP_BAD)
    warn = (kv >= 70) or (waiting >= 1) or (stale >= 2)
    if warn:
        return "WARN", attr(CP_WARN)
    if sample.health_ok and waiting == 0:
        return "OK", attr(CP_OK)
    return "OK", 0


def draw_header(stdscr, y, x, w, state):
    sample = state.last_sample
    now = time.time()
    stale = now - state.last_ok_ts if state.last_ok_ts else 0
    pill, pill_attr = overall_pill(state, sample)
    model = (sample.model_id if sample else "") or "?"
    ctx = (sample.max_ctx if sample else 0) or 0
    if sample and sample.process_start_time > 0:
        up = fmt_uptime(now - sample.process_start_time)
    else:
        up = "—"
    eng_badge = ""
    if sample and sample.engines > 1:
        eng_badge = f" engines:{sample.engines}"

    left = f" vllmtop {state.url}  {model}  ctx {ctx}  up {up}{eng_badge}  {state.tick:.1f}s "
    right = f" [{pill}] [q e c h] "
    addstr_safe(stdscr, y, x, left, curses.A_BOLD | attr(CP_HEADER))
    # right-justify pill
    if len(left) + len(right) < w:
        addstr_safe(stdscr, y, x + w - len(right), right, pill_attr | curses.A_BOLD)

    # Status line for stale / errors / container
    extras = []
    if stale >= 2 and state.last_sample is not None:
        c = attr(CP_BAD) if stale >= 30 else attr(CP_WARN)
        extras.append(("DISCONNECTED" if stale >= 30 else f"stale {int(stale)}s", c))
    if state.err_msg:
        extras.append((state.err_msg, attr(CP_BAD)))
    if state.no_logs:
        extras.append(("logs: disabled (--no-logs)", attr(CP_DIM)))
    elif state.log_rotator and state.current_model:
        # Model-based log status
        s = f"logs: {state.current_model} [{state.log_rotator.status}]"
        extras.append((s, attr(CP_DIM)))
    elif state.container:
        s = f"logs: {state.container}"
        if state.log_tailer and state.log_tailer.dead:
            s += " [dead]"
        elif state.log_tailer:
            s += f" [{state.log_tailer.status}]"
        if state.container_status:
            s += f"  {state.container_status}"
        extras.append((s, attr(CP_DIM)))
    elif state.container_status:
        extras.append((f"logs: {state.container_status}", attr(CP_WARN)))

    if extras:
        col = x
        for text, a in extras:
            addstr_safe(stdscr, y + 1, col, " " + text, a)
            col += len(text) + 2
            if col >= w - 2:
                break


def draw_engine_panel(stdscr, y, x, w, state):
    sample = state.last_sample
    if not sample:
        return 0
    m = sample.metrics
    running = int(m_sum(m, "vllm:num_requests_running"))
    waiting = int(m_sum(m, "vllm:num_requests_waiting"))
    cap = int(m_sum_filter(m, "vllm:num_requests_waiting_by_reason", "reason", "capacity"))
    defr = int(m_sum_filter(m, "vllm:num_requests_waiting_by_reason", "reason", "deferred"))
    preempt = m_sum(m, "vllm:num_preemptions_total")
    preempt_rate = state.rate_1s("vllm:num_preemptions_total") or 0.0
    kv = m_max(m, "vllm:kv_cache_usage_perc") * 100.0

    addstr_safe(stdscr, y, x, " ENGINE   ", attr(CP_HEADER) | curses.A_BOLD)
    col = x + 10
    addstr_safe(stdscr, y, col, "run ")
    col += 4
    addstr_safe(stdscr, y, col, f"{running}", curses.A_BOLD)
    col += len(str(running)) + 2

    addstr_safe(stdscr, y, col, " wait ")
    col += 6
    addstr_safe(stdscr, y, col, f"{waiting}", color_for_waiting(waiting) | curses.A_BOLD)
    col += len(str(waiting)) + 1
    if waiting > 0:
        s = f" (cap {cap} / defer {defr})"
        addstr_safe(stdscr, y, col, s, attr(CP_DIM))
        col += len(s)

    addstr_safe(stdscr, y, col, "   preempt ")
    col += 11
    pa = attr(CP_BAD) if preempt_rate > 0 else 0
    addstr_safe(stdscr, y, col, f"{int(preempt)}", pa)
    col += len(str(int(preempt))) + 1
    if preempt_rate > 0:
        s = f"(+{preempt_rate:.1f}/s)"
        addstr_safe(stdscr, y, col, s, attr(CP_BAD))
        col += len(s) + 1

    # KV bar on the right
    kv_str = f"KV {kv:5.1f}%"
    bar_w = max(8, w - (col - x) - len(kv_str) - 4)
    bar_x = x + w - bar_w - len(kv_str) - 2
    addstr_safe(stdscr, y, bar_x, kv_str, color_for_kv(kv) | curses.A_BOLD)
    addstr_safe(stdscr, y, bar_x + len(kv_str) + 1, draw_kv_bar(bar_w, kv), color_for_kv(kv))
    return 1


def draw_tokens_panel(stdscr, y, x, w, state):
    sample = state.last_sample
    if not sample:
        return 0
    pt = m_sum(sample.metrics, "vllm:prompt_tokens_total")
    gt = m_sum(sample.metrics, "vllm:generation_tokens_total")
    p1 = state.rate_1s("vllm:prompt_tokens_total")
    p10 = state.rate_10s("vllm:prompt_tokens_total")
    g1 = state.rate_1s("vllm:generation_tokens_total")
    g10 = state.rate_10s("vllm:generation_tokens_total")

    addstr_safe(stdscr, y, x, " TOKENS   ", attr(CP_HEADER) | curses.A_BOLD)
    addstr_safe(stdscr, y, x + 10,
                f"prompt {fmt_num(pt, 'tok'):>10}   {fmt_rate(p1)} tok/s (1s)   {fmt_rate(p10)} tok/s (10s)")
    addstr_safe(stdscr, y + 1, x + 10,
                f"output {fmt_num(gt, 'tok'):>10}   {fmt_rate(g1)} tok/s (1s)   {fmt_rate(g10)} tok/s (10s)")
    return 2


def draw_cache_panel(stdscr, y, x, w, state):
    sample = state.last_sample
    if not sample:
        return 0
    q = m_sum(sample.metrics, "vllm:prefix_cache_queries_total")
    h = m_sum(sample.metrics, "vllm:prefix_cache_hits_total")
    hr = (h / q * 100.0) if q > 0 else None
    addstr_safe(stdscr, y, x, " CACHE    ", attr(CP_HEADER) | curses.A_BOLD)
    msg = f"prefix {fmt_pct(hr):>6} hit     {fmt_num(q,'tok')} queries / {fmt_num(h,'tok')} hits"
    addstr_safe(stdscr, y, x + 10, msg)
    return 1


def draw_spec_panel(stdscr, y, x, w, state, h_budget):
    sample = state.last_sample
    if not sample:
        return 0
    drafts = m_sum(sample.metrics, "vllm:spec_decode_num_drafts_total")
    draft_toks = m_sum(sample.metrics, "vllm:spec_decode_num_draft_tokens_total")
    accepted = m_sum(sample.metrics, "vllm:spec_decode_num_accepted_tokens_total")
    alpha = (accepted / drafts) if drafts > 0 else None
    acc_rate = (accepted / draft_toks * 100.0) if draft_toks > 0 else None
    # mean tokens emitted per spec cycle = accepted-draft + 1 bonus from target verify
    accept_len = (alpha + 1.0) if alpha is not None else None
    n_spec = int(round(draft_toks / drafts)) if drafts > 0 else None

    if drafts <= 0 and draft_toks <= 0:
        return 0

    addstr_safe(stdscr, y, x, " SPEC     ", attr(CP_HEADER) | curses.A_BOLD)
    if alpha is not None and acc_rate is not None:
        n_lbl = f"N={n_spec}  " if n_spec else ""
        line = (f"{n_lbl}α {alpha:.2f} tok/draft   accept-len {accept_len:.2f}   "
                f"acc-rate {acc_rate:.1f}%   "
                f"drafts {fmt_num(drafts)}  draft-tok {fmt_num(draft_toks)}  acc-tok {fmt_num(accepted)}")
    else:
        line = "MTP n/a"
    addstr_safe(stdscr, y, x + 10, line)
    rows = 1

    # Row 2: lifetime per-position acceptance from Prometheus (more stable than the log line)
    if h_budget > 1:
        per_pos = m_by_label(sample.metrics,
                             "vllm:spec_decode_num_accepted_tokens_per_pos_total", "position")
        if per_pos and drafts > 0:
            try:
                pos_keys = sorted(per_pos.keys(), key=lambda s: int(s))
            except ValueError:
                pos_keys = sorted(per_pos.keys())
            parts = []
            for k in pos_keys:
                pct = per_pos[k] / drafts * 100.0
                parts.append(f"p{k} {pct:.1f}%")
            addstr_safe(stdscr, y + 1, x + 10,
                        "per-pos lifetime  " + "  ".join(parts))
            rows += 1

    # Row 3: most recent log-line snapshot (rolling window from vLLM's own metrics dump)
    if h_budget > rows:
        spec = None
        for entry in reversed(state.log_buf):
            if entry["kind"] == "spec":
                spec = entry
                break
        if spec:
            pp = " / ".join(f"{x:.2f}" for x in spec["per_pos"])
            addstr_safe(stdscr, y + rows, x + 10,
                        f"last log: α {spec['alpha']:.2f}   acc {spec['acc_tps']:.1f} t/s   "
                        f"draft {spec['draft_tps']:.1f} t/s   per-pos {pp}",
                        attr(CP_DIM))
            rows += 1
    return rows


def hist_summary(state, name, want_cumulative):
    """Return (p50, p99, n_samples) for the chosen view."""
    if want_cumulative:
        dq = state.hist_history.get(name)
        if not dq:
            return None, None, 0
        b = dq[-1][1]
        n = histogram_total(b)
        return percentile_from_buckets(b, 0.5), percentile_from_buckets(b, 0.99), n
    diff, _dt = state.windowed_hist(name)
    if diff is None:
        return None, None, 0
    n = histogram_total(diff)
    if n < 5:
        return None, None, n
    return percentile_from_buckets(diff, 0.5), percentile_from_buckets(diff, 0.99), n


def draw_lat_panel(stdscr, y, x, w, state):
    sample = state.last_sample
    if not sample:
        return 0
    cum = state.show_cumulative_percentiles
    ttft_p50, ttft_p99, ttft_n = hist_summary(state, "vllm:time_to_first_token_seconds", cum)
    itl_p50, itl_p99, itl_n = hist_summary(state, "vllm:inter_token_latency_seconds", cum)
    e2e_p50, e2e_p99, e2e_n = hist_summary(state, "vllm:e2e_request_latency_seconds", cum)

    def _ph(p50, p99, n, label):
        if p50 is None:
            return f"{label} n={int(n)}"
        return f"{label} p50 {fmt_seconds(p50)} p99 {fmt_seconds(p99)}"

    addstr_safe(stdscr, y, x, " LAT      ", attr(CP_HEADER) | curses.A_BOLD)
    window_lbl = "(cumulative)" if cum else "(10s window)"
    line = (f"{_ph(ttft_p50, ttft_p99, ttft_n, 'TTFT')}    "
            f"{_ph(itl_p50, itl_p99, itl_n, 'ITL')}    "
            f"{_ph(e2e_p50, e2e_p99, e2e_n, 'E2E')}    {window_lbl}")
    addstr_safe(stdscr, y, x + 10, line, attr(CP_DIM) if cum else 0)
    return 1


def _vllm_gpu_indices(state):
    """Return sorted list of GPU indices that have a vllm compute-app attached, or []."""
    apps = state.gpu_apps or []
    rows = state.gpu_rows or []
    vllm_uuids = {a["gpu_uuid"] for a in apps if "vllm" in a["name"].lower()}
    if not vllm_uuids:
        return []
    return sorted({r["index"] for r in rows if r["uuid"] in vllm_uuids})


def _reduce_series(state, key_pattern, indices, reducer):
    """Build a deque by applying `reducer(list)` per-tick across per-GPU history.

    Aligns by tail (latest sample) and returns the min-length intersection.
    """
    dqs = [state.graph_history.get(key_pattern.format(i)) for i in indices]
    dqs = [d for d in dqs if d and len(d) >= 2]
    if not dqs:
        return collections.deque(maxlen=240)
    n = min(len(d) for d in dqs)
    out = collections.deque(maxlen=240)
    sliced = [list(d)[-n:] for d in dqs]
    for i in range(n):
        out.append(reducer([s[i] for s in sliced]))
    return out


def _mean_series(state, key_pattern, indices):
    return _reduce_series(state, key_pattern, indices,
                          lambda vs: sum(vs) / len(vs))


def _max_series(state, key_pattern, indices):
    return _reduce_series(state, key_pattern, indices, max)


def draw_graphs_panel(stdscr, y, x, w, state, h_budget):
    """nvtop-style line plots, side-by-side.

    Two plots:
      • engine  — KV% / running / gen tok/s   (vLLM-side)
      • GPUs    — mean util / mean mem% / mean temp°C  (averaged across vLLM-attached GPUs)

    Plot priority when narrow: engine first, then GPUs.
    """
    if h_budget < 5:
        return 0

    gpu_indices = _vllm_gpu_indices(state)

    # Value formatters for each series shown in the legend.
    def _pct(v):   return f"{v:.1f}%"
    def _int(v):   return f"{int(round(v))}"
    def _tps(v):   return f"{v:.1f} t/s"
    def _temp(v):  return f"{int(round(v))}°C"
    def _mhz(v):   return f"{int(round(v))}MHz"

    all_plots = []

    # ── engine plot ──
    eng_specs = []
    eng_right_axis = None
    if "kv_pct" in state.graph_history and len(state.graph_history["kv_pct"]) >= 2:
        eng_specs.append(("KV", attr(CP_PLOT_A), 100.0, _pct, state.graph_history["kv_pct"]))
    if "running" in state.graph_history and len(state.graph_history["running"]) >= 2:
        # Normalize: 5 concurrent → 100% on Y. Legend shows the raw integer.
        eng_specs.append(("running", attr(CP_PLOT_B), 5.0, _int, state.graph_history["running"]))
    if "gen_tps" in state.graph_history and len(state.graph_history["gen_tps"]) >= 2:
        global_max = state.gen_tps_global_max
        # Mode 2: if tok/s > 105, use global max as scale + right axis
        # Mode 1: tok/s ≤ 105, use 100.0 so raw values map directly to 0-100 Y-axis
        if global_max > 105:
            eng_specs.append(("gen", attr(CP_PLOT_C), global_max, _tps, state.graph_history["gen_tps"]))
            eng_right_axis = (attr(CP_PLOT_C), global_max, "tok/s")
        else:
            eng_specs.append(("gen", attr(CP_PLOT_C), 100.0, _tps, state.graph_history["gen_tps"]))
    if eng_specs:
        all_plots.append(("engine", eng_specs, eng_right_axis))

    # ── single mean-of-GPUs plot (util / mem / temp / cpu / clock) ──
    if gpu_indices:
        mean_util  = _mean_series(state, "gpu_{}_util",     gpu_indices)
        mean_mem   = _mean_series(state, "gpu_{}_mem",      gpu_indices)
        mean_temp  = _mean_series(state, "gpu_{}_temp",     gpu_indices)
        mean_clock = _mean_series(state, "gpu_{}_clock",    gpu_indices)
        gpu_specs = []
        gpu_right_axis = None
        if len(mean_util) >= 2:
            gpu_specs.append(("util", attr(CP_PLOT_A), 100.0, _pct, mean_util))
        if len(mean_mem) >= 2:
            gpu_specs.append(("mem",  attr(CP_PLOT_B), 100.0, _pct, mean_mem))
        if len(mean_temp) >= 2:
            gpu_specs.append(("temp", attr(CP_PLOT_C), 100.0, _temp, mean_temp))
        # CPU temp overlaid on the GPU plot
        cpu_temp_dq = state.graph_history.get("cpu_temp")
        if cpu_temp_dq and len(cpu_temp_dq) >= 2:
            gpu_specs.append(("cpu", attr(CP_PLOT_D), 100.0, _temp, cpu_temp_dq))
        if len(mean_clock) >= 2:
            clock_max = max(state.gpu_clock_global_max, 500)
            gpu_specs.append(("clk", attr(CP_PLOT_F), clock_max, _mhz, mean_clock))
            gpu_right_axis = (attr(CP_PLOT_F), clock_max, "MHz")
        if gpu_specs:
            n_gpus = len(gpu_indices)
            title = f"GPUs (mean ×{n_gpus})" if n_gpus > 1 else f"GPU{gpu_indices[0]}"
            all_plots.append((title, gpu_specs, gpu_right_axis))

    if not all_plots:
        return 0

    # Layout. With only 2 plots we can afford taller bodies.
    min_pw = 40
    gap = 2
    section_header_rows = 1
    plot_h = max(6, h_budget - section_header_rows)   # taller than before
    plot_h = min(plot_h, 14)

    addstr_safe(stdscr, y, x, " PLOTS", attr(CP_HEADER) | curses.A_BOLD)

    inner_x = x + 1
    inner_w = w - 1
    max_plots_fit = max(1, (inner_w + gap) // (min_pw + gap))
    n_plots = min(len(all_plots), max_plots_fit)
    plots = all_plots[:n_plots]
    if n_plots == 0:
        return 1
    pw = (inner_w - gap * (n_plots - 1)) // n_plots
    pw = max(min_pw, pw)

    for i, (title, specs, ra) in enumerate(plots):
        px = inner_x + i * (pw + gap)
        try:
            draw_one_plot(stdscr, y + section_header_rows, px, pw, plot_h, title, specs, right_axis=ra)
        except Exception:
            if state.debug_log:
                _log_debug(state, traceback.format_exc())

    return section_header_rows + plot_h


def draw_gpu_panel(stdscr, y, x, w, state, h_budget):
    rows = state.gpu_rows or []
    apps = state.gpu_apps or []
    if not rows and not apps:
        addstr_safe(stdscr, y, x, " GPU      ", attr(CP_HEADER) | curses.A_BOLD)
        addstr_safe(stdscr, y, x + 10, "[no vllm GPU workers attached]", attr(CP_DIM))
        return 1

    # Filter to GPUs that have a vllm compute-app attached.
    vllm_apps = [a for a in apps if "vllm" in a["name"].lower()]
    vllm_uuids = {a["gpu_uuid"] for a in vllm_apps}
    rows = [r for r in rows if r["uuid"] in vllm_uuids] if vllm_uuids else []
    if not rows:
        addstr_safe(stdscr, y, x, " GPU      ", attr(CP_HEADER) | curses.A_BOLD)
        addstr_safe(stdscr, y, x + 10, "[no vllm GPU workers attached]", attr(CP_DIM))
        return 1

    addstr_safe(stdscr, y, x, " GPU NAME            UTIL  MEM (GB)         TEMP    POWER   CLK    PCIe",
                attr(CP_HEADER) | curses.A_BOLD)
    written = 1
    budget = h_budget
    for r in rows:
        if written >= budget:
            break
        mem_used_gb = r["mem_used"] / 1024.0
        mem_total_gb = r["mem_total"] / 1024.0
        mem_pct = (r["mem_used"] / r["mem_total"] * 100.0) if r["mem_total"] > 0 else 0
        line = (f"{r['index']:3d} {r['name'][:16]:<16}  {int(r['util']):3d}%  "
                f"{mem_used_gb:5.1f} / {mem_total_gb:5.1f}   ")
        addstr_safe(stdscr, y + written, x, line)
        # Temp + power get conditional color
        col = x + len(line)
        temp_str = f"{int(r['temp']):3d} C"
        addstr_safe(stdscr, y + written, col, temp_str, color_for_temp(r["temp"]))
        col += len(temp_str) + 3
        addstr_safe(stdscr, y + written, col, f"{int(r['power']):4d} W")
        col += 7
        addstr_safe(stdscr, y + written, col, f"{int(r['clock']):4d} MHz")
        col += 8
        pcie = f"  Gen{r['pcie_gen']} x{r['pcie_width']}"
        addstr_safe(stdscr, y + written, col, pcie, attr(CP_DIM))
        # Color whole row by mem pct if high
        if mem_pct >= 95:
            # repaint name highlight
            addstr_safe(stdscr, y + written, x + 4, f"{r['name'][:16]:<16}", color_for_mem_pct(mem_pct))
        written += 1

        # PCIe sub-line: per-tick rx/tx (last / peak / mean MB/s), saturation, AER.
        if written >= budget:
            continue
        rx_last = r.get("pcie_rx_last")
        tx_last = r.get("pcie_tx_last")
        rx_peak = r.get("pcie_rx_peak")
        tx_peak = r.get("pcie_tx_peak")
        rx_mean = r.get("pcie_rx_mean")
        tx_mean = r.get("pcie_tx_mean")
        sat = r.get("pcie_sat_pct", 0.0)
        aer = r.get("pcie_aer_delta", 0)
        theo = r.get("pcie_theo_mbps", 0.0)
        n = r.get("pcie_n_samples", 0)
        if rx_last is None and tx_last is None and not r.get("pcie_aer_delta"):
            sub = f"    PCIe  [{(state.pcie_sampler.status if state.pcie_sampler else 'off')}]"
            addstr_safe(stdscr, y + written, x, sub, attr(CP_DIM))
        else:
            theo_s = f" / {int(theo)}" if theo else ""
            rx_s = (f"Rx {int(rx_last):>4} ▲{int(rx_peak):>4} μ{int(rx_mean):>4}"
                    if rx_last is not None else "Rx  ----")
            tx_s = (f"Tx {int(tx_last):>4} ▲{int(tx_peak):>4} μ{int(tx_mean):>4}"
                    if tx_last is not None else "Tx  ----")
            sub_left = f"    PCIe  {rx_s}  {tx_s} MB/s{theo_s}  "
            addstr_safe(stdscr, y + written, x, sub_left, attr(CP_DIM))
            col2 = x + len(sub_left)
            sat_s = f"sat ▲{sat:5.1f}%"
            sat_col = (attr(CP_BAD) if sat >= 95 else
                       attr(CP_WARN) if sat >= 80 else
                       attr(CP_DIM))
            addstr_safe(stdscr, y + written, col2, sat_s, sat_col)
            col2 += len(sat_s) + 2
            aer_s = f"AER +{aer}"
            aer_col = attr(CP_WARN) if aer > 0 else attr(CP_DIM)
            addstr_safe(stdscr, y + written, col2, aer_s, aer_col)
            col2 += len(aer_s) + 2
            n_s = f"(n={n})"
            addstr_safe(stdscr, y + written, col2, n_s, attr(CP_DIM))
        written += 1

    # Compute apps subsection
    if written < budget and vllm_apps:
        addstr_safe(stdscr, y + written, x, "", 0)
        written += 1
        if written < budget:
            addstr_safe(stdscr, y + written, x,
                        "   PID PROCESS                  GPU MEM",
                        attr(CP_HEADER) | curses.A_BOLD)
            written += 1
        for a in vllm_apps:
            if written >= budget:
                break
            mem_gb = a["mem"] / 1024.0
            addstr_safe(stdscr, y + written, x,
                        f"{a['pid']:>6} {a['name'][:24]:<24}  {mem_gb:5.1f} GB",
                        attr(CP_OK))
            written += 1
    return written


def draw_log_panel(stdscr, y, x, w, state, h_budget):
    title = " LOG"
    if state.log_rotator and state.current_model:
        # Model-based log file
        log_file = os.path.join(state.log_rotator.log_dir, f"{state.current_model}.log")
        title += f"  ({log_file})"
    elif state.container:
        title += f"  (docker logs -f {state.container})"
    elif state.container_status:
        title += f"  [{state.container_status}]"
    addstr_safe(stdscr, y, x, title, attr(CP_HEADER) | curses.A_BOLD)
    if h_budget <= 1:
        return 1
    body_rows = h_budget - 1
    entries = list(state.log_buf)[-body_rows:]
    n_entries = len(entries)
    for i, entry in enumerate(entries):
        ts = entry["ts"]
        if entry["kind"] == "engine":
            line = (f"{ts} Engine:  prompt {entry['prompt_tps']:>6.1f} t/s  "
                    f"gen {entry['gen_tps']:>6.1f} t/s  "
                    f"run {entry['running']}  wait {entry['waiting']}  "
                    f"KV {entry['kv']:5.1f}%  prefix {entry['prefix_hit']:5.1f}%")
        elif entry["kind"] == "spec":
            pp = "/".join(f"{x:.2f}" for x in entry["per_pos"])
            line = (f"{ts} SpecDec: α {entry['alpha']:.2f}  "
                    f"acc {entry['acc_tps']:>5.1f} t/s  "
                    f"draft {entry['draft_tps']:>5.1f} t/s  "
                    f"per-pos {pp}")
        else:
            line = f"{ts} {entry}"
        a = 0 if i == n_entries - 1 else attr(CP_DIM)
        addstr_safe(stdscr, y + 1 + i, x, line, a)
    return 1 + n_entries


def draw_separator(stdscr, y, x, w):
    addstr_safe(stdscr, y, x, "─" * (w - x), attr(CP_DIM))


# ─────────────────────────────────────────────────── render loop ──

def render(stdscr, state):
    stdscr.erase()
    rows, cols = stdscr.getmaxyx()
    if rows < 6 or cols < 30:
        addstr_safe(stdscr, 0, 0, "terminal too small", attr(CP_BAD))
        stdscr.refresh()
        return

    # Apply staleness dimming by drawing into a buffer first conceptually — here we just
    # dim individual lines via the panel renderers when stale > 2s.
    y = 0
    draw_header(stdscr, 0, 0, cols, state)
    y = 2  # header takes 2 rows

    panels_top = [
        ("engine", draw_engine_panel, 1),
        ("tokens", draw_tokens_panel, 2),
        ("cache", draw_cache_panel, 1),
        ("spec", lambda s, yy, xx, ww, st: draw_spec_panel(s, yy, xx, ww, st, 3), 3),
        ("lat", draw_lat_panel, 1),
    ]
    for name, fn, _need in panels_top:
        if y >= rows - 1:
            break
        try:
            consumed = fn(stdscr, y, 0, cols, state)
        except Exception:
            consumed = 0
            if state.debug_log:
                _log_debug(state, traceback.format_exc())
        y += consumed

    if y < rows - 1:
        draw_separator(stdscr, y, 0, cols)
        y += 1

    # PLOTS panel: aim for ≥ 8 rows but grow up to 15 when the terminal is tall.
    # Reserve a minimum below it for GPU summary + LOG so we don't starve them.
    remaining = rows - y
    log_min = 0 if state.no_logs else 4
    gpu_min = 4
    sep_below = 1
    if remaining >= 8 + gpu_min + log_min + sep_below:
        graphs_budget = min(15, remaining - gpu_min - log_min - sep_below)
        try:
            consumed = draw_graphs_panel(stdscr, y, 0, cols, state, graphs_budget)
        except Exception:
            consumed = 0
            if state.debug_log:
                _log_debug(state, traceback.format_exc())
        if consumed:
            y += consumed
            if y < rows - 1:
                draw_separator(stdscr, y, 0, cols)
                y += 1

    # Allocate remaining rows: GPU gets up to 6, then LOG takes rest.
    remaining = rows - y
    if remaining > 1:
        gpu_budget = min(8, max(2, remaining - 4))
        if state.no_logs:
            gpu_budget = remaining
        consumed = draw_gpu_panel(stdscr, y, 0, cols, state, gpu_budget)
        y += consumed
        if y < rows - 1:
            draw_separator(stdscr, y, 0, cols)
            y += 1

    if not state.no_logs and y < rows:
        draw_log_panel(stdscr, y, 0, cols, state, rows - y)

    stdscr.refresh()


def _log_debug(state, msg):
    if not state.debug_log:
        return
    try:
        with open(state.debug_log, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except OSError:
        pass


# ─────────────────────────────────────────────────── startup probe / modal ──

def show_connecting(stdscr, url):
    stdscr.erase()
    rows, cols = stdscr.getmaxyx()
    msg = f"Connecting to {url}…"
    addstr_safe(stdscr, rows // 2, max(0, (cols - len(msg)) // 2), msg, curses.A_BOLD)
    stdscr.refresh()


def show_error(stdscr, msg):
    stdscr.erase()
    rows, cols = stdscr.getmaxyx()
    addstr_safe(stdscr, rows // 2 - 1, max(0, (cols - len(msg)) // 2), msg, attr(CP_BAD) | curses.A_BOLD)
    hint = "Press any key to enter a URL, q to quit"
    addstr_safe(stdscr, rows // 2 + 1, max(0, (cols - len(hint)) // 2), hint, attr(CP_DIM))
    stdscr.refresh()


def initial_probe_loop(stdscr, state):
    """Probe state.url; on failure show modal, retry. Returns False if user quits."""
    try:
        while True:
            show_connecting(stdscr, state.url)
            sample, err = fetch_sample(state.url, timeout=2.0)
            if sample:
                update_state(state, sample)
                return True
            show_error(stdscr, f"Cannot reach {state.url}: {err}")
            stdscr.timeout(-1)
            ch = stdscr.getch()
            if ch in (ord("q"), ord("Q"), 27):
                return False
            new_url = curses_input_modal(stdscr, "Set vLLM endpoint",
                                         "Enter vLLM URL (host:port or http://host:port):",
                                         state.url)
            if new_url is None:
                return False
            state.url = normalize_url(new_url)
            # Re-resolve container for new URL
            if not state.no_logs:
                _rebind_log_tailer(state)
    except KeyboardInterrupt:
        return False


def _rebind_log_tailer(state):
    if state.log_tailer:
        state.log_tailer.stop()
        state.log_tailer = None
    state.log_buf.clear()
    name, status = autodetect_container(state.url)
    state.container = name
    state.container_status = status
    if name and state.log_rotator:
        state.log_tailer = LogTailer(name, state.log_buf, state.log_rotator)
        state.log_tailer.start()


# ─────────────────────────────────────────────────── main loop ──

def run_tui(stdscr, state):
    curses.curs_set(0)
    init_colors()
    stdscr.nodelay(True)
    stdscr.keypad(True)

    if not initial_probe_loop(stdscr, state):
        return 0

    if not state.no_logs and state.log_tailer is None:
        _rebind_log_tailer(state)

    # Host-level PCIe telemetry: NVML sub-tick throughput sampler + AER counters.
    state.gpu_bdfs = discover_gpu_bdfs()
    state.pcie_sampler = NvmlPcieSampler()
    state.pcie_sampler.start()

    last_render = 0.0
    tick_ms = max(100, int(state.tick * 1000))
    stdscr.timeout(tick_ms)

    last_sample_attempt = 0.0
    last_gpu_attempt = 0.0

    try:
      while True:
        now = time.time()

        # Sample /metrics every tick
        if now - last_sample_attempt >= state.tick * 0.95:
            last_sample_attempt = now
            sample, err = fetch_sample(state.url, timeout=max(1.0, state.tick))
            if sample:
                update_state(state, sample)
                # Check for model change and rotate logs if needed
                if state.log_rotator and sample.model_id:
                    if state.current_model != sample.model_id:
                        state.current_model = sample.model_id
                        state.log_rotator.set_model(sample.model_id)
                state.err_msg = ""
            else:
                state.err_msg = err if err else "fetch failed"

        # GPU every tick (cheap)
        if now - last_gpu_attempt >= state.tick * 0.95:
            last_gpu_attempt = now
            gpus = nvidia_query_gpu()
            apps = nvidia_query_compute_apps()
            if gpus is not None:
                sampler = state.pcie_sampler
                for r in gpus:
                    idx = r["index"]
                    state.push_graph(f"gpu_{idx}_util", r["util"])
                    state.push_graph(f"gpu_{idx}_mem", r["mem_used"] / r["mem_total"] * 100.0 if r["mem_total"] > 0 else 0.0)
                    state.push_graph(f"gpu_{idx}_temp", r["temp"])
                    state.push_graph(f"gpu_{idx}_clock", r["clock"])
                    if r["clock"] > state.gpu_clock_global_max:
                        state.gpu_clock_global_max = r["clock"]

                    # PCIe theoretical (gen×width) for saturation calc
                    theo = pcie_theoretical_mbps(r["pcie_gen"], r["pcie_width"])
                    r["pcie_theo_mbps"] = theo

                    stats = sampler.window_stats(idx, state.tick) if sampler else None
                    if stats:
                        r["pcie_rx_last"] = stats["rx_last"]
                        r["pcie_rx_mean"] = stats["rx_mean"]
                        r["pcie_rx_peak"] = stats["rx_peak"]
                        r["pcie_tx_last"] = stats["tx_last"]
                        r["pcie_tx_mean"] = stats["tx_mean"]
                        r["pcie_tx_peak"] = stats["tx_peak"]
                        r["pcie_n_samples"] = stats["n_samples"]
                        peak = max(stats["rx_peak"], stats["tx_peak"])
                        sat = (peak / theo * 100.0) if theo > 0 else 0.0
                        r["pcie_sat_pct"] = sat

                    # AER correctable cumulative delta (link-health latency proxy)
                    bdf = state.gpu_bdfs.get(idx)
                    if bdf:
                        cur = read_aer_correctable_total(bdf)
                        if cur is not None:
                            base = state.pcie_aer_baseline.setdefault(idx, cur)
                            r["pcie_aer_delta"] = cur - base

                state.gpu_rows = gpus
                state.gpu_stale_ts = now
            if apps is not None:
                state.gpu_apps = apps

            # CPU temperature
            cpu_temps = get_cpu_temps()
            if cpu_temps is not None:
                state.push_graph("cpu_temp", max(cpu_temps))

        try:
            render(stdscr, state)
        except curses.error:
            pass
        except Exception:
            _log_debug(state, traceback.format_exc())

        ch = stdscr.getch()
        if ch == -1:
            continue
        if ch in (ord("q"), ord("Q")):
            return 0
        if ch == curses.KEY_RESIZE:
            curses.update_lines_cols()
            continue
        if ch in (ord("h"), ord("H")):
            state.show_cumulative_percentiles = not state.show_cumulative_percentiles
            continue
        if ch in (ord("e"), ord("E")):
            new_url = curses_input_modal(stdscr, "Set vLLM endpoint",
                                         "Enter vLLM URL (host:port or http://host:port):",
                                         state.url)
            if new_url:
                state.url = normalize_url(new_url)
                state.clear_history()
                state.last_sample = None
                state.last_ok_ts = 0.0
                if not state.no_logs:
                    _rebind_log_tailer(state)
            continue
        if ch in (ord("c"), ord("C")):
            new_c = curses_input_modal(stdscr, "Set log container",
                                       "Container name (must be running locally):",
                                       state.container or "")
            if new_c:
                if state.log_tailer:
                    state.log_tailer.stop()
                state.container = new_c
                state.container_status = ""
                state.log_buf.clear()
                state.log_tailer = LogTailer(new_c, state.log_buf, state.log_rotator)
                state.log_tailer.start()
            continue
    except KeyboardInterrupt:
        return 0
    finally:
        if state.pcie_sampler:
            state.pcie_sampler.stop()


def main():
    ap = argparse.ArgumentParser(prog="vllmtop",
                                 description="top-like TUI for a running vLLM server")
    ap.add_argument("--url", default=DEFAULT_URL,
                    help=f"vLLM endpoint (default: {DEFAULT_URL}). host:port shorthand accepted.")
    ap.add_argument("--container", default=None,
                    help="docker container name for the log pane (default: auto-detect)")
    ap.add_argument("--tick", type=float, default=1.0,
                    help="refresh interval in seconds (default: 1.0, capped at 10)")
    ap.add_argument("--no-logs", action="store_true",
                    help="disable the docker-logs pane (no background thread)")
    ap.add_argument("--debug", default=None, metavar="FILE",
                    help="append internal errors / tracebacks to FILE")
    args = ap.parse_args()

    url = normalize_url(args.url)
    tick = min(10.0, max(0.1, args.tick))

    container = args.container
    container_status = ""
    if not args.no_logs and not container:
        container, container_status = autodetect_container(url)
    elif args.no_logs:
        container_status = "disabled by --no-logs"

    # Initialize log rotator for model-based logging
    log_rotator = LogRotator() if not args.no_logs else None

    state = State(
        url=url,
        container=container,
        container_status=container_status,
        tick=tick,
        debug_log=args.debug,
        no_logs=args.no_logs,
        log_rotator=log_rotator,
    )

    if container and not args.no_logs and log_rotator:
        state.log_tailer = LogTailer(container, state.log_buf, log_rotator)
        state.log_tailer.start()

    rc = 0
    try:
        rc = curses.wrapper(lambda s: run_tui(s, state))
    except KeyboardInterrupt:
        rc = 0
    finally:
        if state.log_tailer:
            state.log_tailer.stop()
        if state.log_rotator:
            state.log_rotator.close()
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
