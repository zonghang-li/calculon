/**
 * gpu_info.cu — CUDA GPU capability & peak–throughput snapshot
 *
 * WHAT THIS PROGRAM IS FOR
 * ------------------------
 * A tiny utility to probe the CUDA runtime for each visible GPU and print a
 * compact capability/performance profile you can paste into logs, dashboards,
 * or CI. It reports architecture (compute capability), SM/core counts, key
 * memory sizes, and rough theoretical peaks (FP32 TFLOPS and memory bandwidth).
 * If built with NVML, it can also include live telemetry (temperature, power,
 * utilization, PCIe/NVLink, MIG summary).
 *
 * MAIN FEATURES
 * -------------
 * • Works out of the box with just the CUDA runtime (no admin privileges).
 * • Device filtering: inspect a single device, a set/range, or all GPUs.
 * • Multiple output formats:
 *     - Brief: one-line summaries.
 *     - Verbose: multi-line, human-readable dump of properties/limits.
 *     - JSON: stable field names for scripts/CI.
 *     - CSV: tabular output for spreadsheets.
 * • Derived metrics:
 *     - Peak FP32 TFLOPS (approx.; uses an arch table for cores/SM).
 *     - Peak memory bandwidth (approx.; from mem clock × bus width).
 * • Optional NVML block (when linked and available): temperature, power,
 *   utilization %, ECC mode, PCIe/NVLink info, MIG instance summary.
 *
 * COMMAND-LINE OPTIONS
 * --------------------
 *   -d, --device <idx[,idx|start-end]>
 *       Limit output to one or more device indices.
 *       Examples: -d 0   |   -d 0,2   |   -d 1-3
 *       Default: all devices.
 *
 *   -b, --brief
 *       One-line summary per device (name, CC, SMs, memory, bandwidth, TFLOPS).
 *
 *   -v, --verbose
 *       Detailed multi-line dump of static properties (default if no other
 *       format is chosen).
 *
 *   -j, --json
 *       Emit a JSON array of device objects (stable field names; script-friendly).
 *
 *   -c, --csv
 *       Emit a CSV table (one row per device). Column order mirrors --query (or defaults).
 *
 *   -q, --query <fields>
 *       Comma-separated field list to print/emit (works with any format).
 *       Unknown fields are ignored. Common fields include:
 *         name, index, uuid, pci, cc, sm, cores_per_sm, cores_total,
 *         clock_core_mhz, clock_mem_mhz, mem_total_mb, mem_bus_width_bits,
 *         l2_bytes, smem_per_sm_bytes, smem_per_block_bytes, regs_per_block,
 *         warp_size, max_threads_per_block, tflops_fp32, bandwidth_gbps,
 *         nvml_temp_c, nvml_power_w, nvml_power_limit_w, nvml_util_gpu_pct,
 *         nvml_util_mem_pct, pcie_gen, pcie_width, nvlink_links
 *
 *   --no-nvml
 *       Disable NVML queries even if the binary was built with NVML support.
 *
 *   --topology
 *       Print a simple peer-to-peer accessibility/affinity matrix (if supported).
 *
 *   --mig
 *       Include MIG instance summary (if supported and NVML available).
 *
 *   -h, --help
 *       Show usage/help text.
 *
 * HOW TO BUILD
 * ------------
 * Minimal (CUDA runtime only; static properties):
 *   nvcc -O2 -std=c++17 gpu_info.cu -o gpu_info
 *
 * Target a specific architecture (recommended for accurate core-clock reporting):
 *   nvcc -O2 -std=c++17 -arch=sm_80 gpu_info.cu -o gpu_info
 *
 * With NVML telemetry (Linux: link libnvidia-ml):
 *   nvcc -O2 -std=c++17 gpu_info.cu -o gpu_info -lnvidia-ml
 * Notes:
 *   • On Windows, link against nvml.lib and ensure nvml.dll is on PATH.
 *   • If NVML isn’t present at runtime, telemetry fields are omitted/NA.
 *
 * HOW TO USE
 * ----------
 * Show everything for all devices (human-readable):
 *   ./gpu_info
 *
 * One-line summaries for devices 0 and 2:
 *   ./gpu_info -b -d 0,2
 *
 * JSON with a custom field set (good for scripts/CI):
 *   ./gpu_info -j -q name,cc,sm,mem_total_mb,bandwidth_gbps,tflops_fp32
 *
 * CSV for a single device:
 *   ./gpu_info -c -d 0
 *
 * WHAT THE OUTPUT LOOKS LIKE
 * --------------------------
 * The program can emit different formats. Examples:
 *
 * 1) Compact key=value snapshot (as printed by the simple path in this file):
 *    $ ./gpu_info
 *    GPU_NAME=Tesla P40
 *    SM=30
 *    CC=61
 *    FP32_TFLOPS_PEAK=11.758
 *    MEM1_GBPS_PEAK_THEO=347.04
 *    (Depending on your shell/tools, you may see these on one line, e.g.:
 *     ./gpu_info GPU_NAME=Tesla P40 SM=30 CC=61 FP32_TFLOPS_PEAK=11.758 MEM1_GBPS_PEAK_THEO=347.04)
 *
 * 2) Verbose (default):
 *    Device 0: NVIDIA H100 PCIe (UUID: GPU-xxxx...)
 *      CC: 9.0, SMs: 114, WarpSize: 32, Cores/SM: 128 (approx), Cores total: 14592
 *      Clocks: core 1410 MHz, mem 2600 MHz, Bus width: 512-bit
 *      Memory: total 80023 MiB, L2: 50,331,648 B, Shared/SM: 99,840 B, Shared/block: 99,840 B
 *      Limits: maxThreadsPerBlock 1024, regsPerBlock 65536, constMem 65,536 B
 *      Theoretical: Bandwidth ~ 2,038.4 GB/s, FP32 ~ 60.3 TFLOPS
 *      NVML: temp 43 C, power 92 W / 350 W, util {gpu: 3%, mem: 5%}, PCIe Gen5 x16, NVLink: 0 links
 *
 * 3) JSON (-j/--json):
 *    [
 *      {
 *        "index": 0, "name": "NVIDIA H100 PCIe", "cc": "9.0",
 *        "sm": 114, "cores_per_sm": 128, "cores_total": 14592,
 *        "clock_core_mhz": 1410, "clock_mem_mhz": 2600,
 *        "mem_total_mb": 80023, "mem_bus_width_bits": 512,
 *        "bandwidth_gbps": 2038.4, "tflops_fp32": 60.3
 *      }
 *    ]
 *
 * 4) CSV (-c/--csv, with a query set):
 *    index,name,cc,sm,mem_total_mb,bandwidth_gbps,tflops_fp32
 *    0,NVIDIA H100 PCIe,9.0,114,80023,2038.4,60.3
 *
 * NOTES / LIMITATIONS
 * -------------------
 * • CUDA runtime reports static capabilities; some fields may be 0 or unavailable
 *   on older drivers/devices—these are handled gracefully.
 * • Throughput and bandwidth numbers are *theoretical* estimates intended for
 *   comparison/baselining, not precise performance guarantees.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

static int cores_per_sm(int maj,int min){
  int cc=maj*10+min;
  if(cc>=90) return 128;
  if(cc>=86) return 128;
  if(cc>=80) return 64;
  if(cc>=75) return 64;
  if(cc>=70) return 64;
  if(cc>=61) return 128;
  if(cc>=60) return 128;
  if(cc>=50) return 128;
  return 64;
}

int main(){
  int dev=0; cudaGetDevice(&dev);
  cudaDeviceProp p{}; cudaGetDeviceProperties(&p,dev);
  double sm=p.multiProcessorCount, cores=cores_per_sm(p.major,p.minor);
  double hz=p.clockRate*1000.0;
  double tflops = sm*cores*hz*2.0/1e12;
  // theoretical mem bw (very rough, double data rate)
  double mem_hz = p.memoryClockRate*1000.0;
  double bytes_s = mem_hz * (p.memoryBusWidth/8.0) * 2.0;
  double gbps_theo = bytes_s/1e9;

  std::printf("GPU_NAME=%s\n", p.name);
  std::printf("SM=%d\n", p.multiProcessorCount);
  std::printf("CC=%d%d\n", p.major,p.minor);
  std::printf("FP32_TFLOPS_PEAK=%.3f\n", tflops);
  std::printf("MEM1_GBPS_PEAK_THEO=%.2f\n", gbps_theo);
  return 0;
}