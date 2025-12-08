/**
 * gpu_info.cu — CUDA GPU capability & peak-throughput snapshot
 *
 * PURPOSE
 * -------
 * A tiny utility that queries the CUDA runtime for the *current* device and prints a
 * compact capability/performance profile you can paste into logs or CI. It reports:
 *   GPU_NAME, SM (SMs on NVIDIA), CC (major/minor), FP32_TFLOPS_PEAK,
 *   MEM1_GBPS_PEAK_THEO.
 *
 * SCOPE
 * -----
 * • Single device, static properties only (no topology/telemetry).
 * • On multi-GPU systems, each GPU appears as its own CUDA device; run once per
 *   visible device.
 * • This program does not probe PCIe/NVLink/InfiniBand fabrics; interconnect
 *   topology matters for multi-GPU apps but is outside the scope of this snapshot.
 *
 * BUILD
 * -----
 * Recommended:
 *   nvcc -O2 -std=c++17 gpu_info.cu -o gpu_info
 *
 * USAGE
 * -----
 *   ./gpu_info_cuda
 *   # To choose which device the process sees:
 *   export CUDA_VISIBLE_DEVICES=0
 *
 * OUTPUT EXAMPLE
 * --------------
 *   GPU_NAME=NVIDIA H100 PCIe
 *   SM=132
 *   CC=90
 *   FP32_TFLOPS_PEAK=67.000
 *   MEM1_GBPS_PEAK_THEO=2000.00
 *
 * HOW NUMBERS ARE DERIVED
 * -----------------------
 *   FP32_TFLOPS_PEAK ≈ (multiProcessorCount × cores_per_MP × core_clock_Hz × 2) / 1e12
 *   MEM1_GBPS_PEAK_THEO ≈ mem_clock_Hz × (bus_width_bits / 8) × 2 / 1e9
 *   • cores_per_MP is a coarse mapping:
 *       NVIDIA: 128/64 depending on architecture (fallback table).
 *
 * LIMITATIONS
 * -----------
 * • CC is the CUDA compute capability (major/minor).
 * • On some stacks memory clock/bus width are unavailable; bandwidth prints as 0.00.
 * • Values are theoretical upper bounds for quick comparison, not guarantees.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static inline void ckCuda(cudaError_t e, const char* where){
  if(e != cudaSuccess){
    std::fprintf(stderr, "CUDA error %d (%s) at %s\n",
                 (int)e, cudaGetErrorString(e), where);
    std::exit(1);
  }
}

// Coarse FP32 cores per SM
static int fp32_cores_per_mp(const cudaDeviceProp& p){
  const int cc = p.major*10 + p.minor;
  if(cc >= 90) return 128; // Hopper/Ada fallback
  if(cc >= 86) return 128; // Ampere GA10x
  if(cc >= 80) return  64; // A100
  if(cc >= 75) return  64; // Turing
  if(cc >= 70) return  64; // Volta
  if(cc >= 61) return 128; // Pascal GP10x
  if(cc >= 60) return 128; // Pascal
  if(cc >= 50) return 128; // Maxwell fallback
  return 64;
}

static int get_attr_i(int dev, cudaDeviceAttr a){
  int v = 0;
  cudaError_t e = cudaDeviceGetAttribute(&v, a, dev);
  return (e == cudaSuccess) ? v : 0;
}

int main(){
  // Use the current device (index 0 unless CUDA_VISIBLE_DEVICES is set)
  int dev = 0;
  ckCuda(cudaGetDevice(&dev), "cudaGetDevice");
  cudaDeviceProp p{};
  ckCuda(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");

  // Core clock (kHz) via attribute, fallback to prop.clockRate
  int core_khz = get_attr_i(dev, cudaDevAttrClockRate);
  if(core_khz <= 0) core_khz = p.clockRate;

  // Memory clock (kHz) and bus width (bits) — attributes preferred
  int mem_khz  = get_attr_i(dev, cudaDevAttrMemoryClockRate);
  int bus_bits = get_attr_i(dev, cudaDevAttrGlobalMemoryBusWidth);
  if(mem_khz  <= 0) mem_khz  = p.memoryClockRate;   // may be 0 on some stacks
  if(bus_bits <= 0) bus_bits = p.memoryBusWidth;    // may be 0 on some stacks

  // TFLOPs (coarse)
  const double mps   = (double)p.multiProcessorCount;
  const double cores = (double)fp32_cores_per_mp(p);
  const double hz    = (double)core_khz * 1000.0;                 // Hz
  const double tflops = mps * cores * hz * 2.0 / 1e12;            // FP32 FMA => *2

  // Bandwidth (very rough, "DDR" x2)
  const double mem_hz    = (double)mem_khz * 1000.0;              // Hz
  const double bytes_s   = mem_hz * (bus_bits / 8.0) * 2.0;       // B/s
  const double gbps_theo = (mem_khz > 0 && bus_bits > 0) ? (bytes_s / 1e9) : 0.0;

  // Output (same keys as the HIP snippet)
  std::printf("GPU_NAME=%s\n", p.name);
  std::printf("SM=%d\n", p.multiProcessorCount);
  std::printf("CC=%d%d\n", p.major, p.minor);
  std::printf("FP32_TFLOPS_PEAK=%.3f\n", tflops);
  std::printf("MEM1_GBPS_PEAK_THEO=%.2f\n", gbps_theo);
  return 0;
}
