/**
 * gpu_info.cpp — HIP GPU capability & peak–throughput snapshot (ROCm + NVIDIA via HIP)
 *
 * PURPOSE
 * -------
 * A tiny utility that queries the HIP runtime for the *current* device and prints a
 * compact capability/performance profile you can paste into logs or CI. It reports:
 *   GPU_NAME, SM (CUs on AMD / SMs on NVIDIA), CC (major/minor), FP32_TFLOPS_PEAK,
 *   MEM1_GBPS_PEAK_THEO.
 *
 * SCOPE
 * -----
 * • Single device, static properties only (no topology/telemetry).
 * • On MI250X each GCD appears as its own HIP device; run once per visible GCD.
 * • This program does not probe PCIe/IF/Ethernet fabrics. The MI250x8 two‑tier
 *   topology (intra‑socket IF, inter‑socket PCIe4, cross‑node Ethernet) matters
 *   for multi‑GPU apps but is outside the scope of this snapshot.
 *
 * BUILD
 * -----
 * Recommended for MI250X (gfx90a):
 *   hipcc -O2 -std=c++17 --offload-arch=gfx90a gpu_info.cpp -o gpu_info_amd
 *
 * Note: --amdgpu-target is deprecated; use --offload-arch.
 *
 * USAGE
 * -----
 *   ./gpu_info_amd
 *   # To choose which device the process sees:
 *   export HIP_VISIBLE_DEVICES=0      # or
 *   export ROCR_VISIBLE_DEVICES=2
 *
 * OUTPUT EXAMPLE
 * --------------
 *   GPU_NAME=AMD Instinct MI250X (gfx90a)
 *   SM=110
 *   CC=00
 *   FP32_TFLOPS_PEAK=47.500
 *   MEM1_GBPS_PEAK_THEO=3276.80
 *
 * HOW NUMBERS ARE DERIVED
 * -----------------------
 *   FP32_TFLOPS_PEAK ≈ (multiProcessorCount × cores_per_MP × core_clock_Hz × 2) / 1e12
 *   MEM1_GBPS_PEAK_THEO ≈ mem_clock_Hz × (bus_width_bits / 8) × 2 / 1e9
 *   • cores_per_MP is a coarse mapping:
 *       AMD: 64 FP32 lanes per CU (approx.)
 *       NVIDIA: 128/64 depending on architecture (fallback table).
 *
 * LIMITATIONS
 * -----------
 * • CC may print 00 on AMD (HIP does not expose a CUDA‑style “compute capability”).
 * • On some stacks memory clock/bus width are unavailable; bandwidth prints as 0.00.
 * • Values are theoretical upper bounds for quick comparison, not guarantees.
 */


#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

static inline void ckHip(hipError_t e, const char* where){
  if(e != hipSuccess){
    std::fprintf(stderr, "HIP error %d (%s) at %s\n",
                 (int)e, hipGetErrorString(e), where);
    std::exit(1);
  }
}

// Coarse FP32 cores per MP/CU
static int fp32_cores_per_mp(const hipDeviceProp_t& p){
#if defined(__HIP_PLATFORM_NVIDIA__)
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
#else
  // AMD CDNA/RDNA (coarse): 64 FP32 lanes per CU
  return 64;
#endif
}

static int get_attr_i(int dev, hipDeviceAttribute_t a){
  int v = 0;
  hipError_t e = hipDeviceGetAttribute(&v, a, dev);
  return (e == hipSuccess) ? v : 0;
}

int main(){
  // Use the current device (index 0 unless HIP_VISIBLE_DEVICES is set)
  int dev = 0;
  ckHip(hipGetDevice(&dev), "hipGetDevice");
  hipDeviceProp_t p{};
  ckHip(hipGetDeviceProperties(&p, dev), "hipGetDeviceProperties");

  // Core clock (kHz) via attribute, fallback to prop.clockRate
  int core_khz = get_attr_i(dev, hipDeviceAttributeClockRate);
  if(core_khz <= 0) core_khz = p.clockRate;

  // Memory clock (kHz) and bus width (bits) — attributes preferred
  int mem_khz  = get_attr_i(dev, hipDeviceAttributeMemoryClockRate);
  int bus_bits = get_attr_i(dev, hipDeviceAttributeMemoryBusWidth);
  if(mem_khz  <= 0) mem_khz  = p.memoryClockRate;   // may be 0 on some AMD stacks
  if(bus_bits <= 0) bus_bits = p.memoryBusWidth;    // may be 0 on some AMD stacks

  // TFLOPs (coarse)
  const double mps   = (double)p.multiProcessorCount;
  const double cores = (double)fp32_cores_per_mp(p);
  const double hz    = (double)core_khz * 1000.0;                 // Hz
  const double tflops = mps * cores * hz * 2.0 / 1e12;            // FP32 FMA => *2

  // Bandwidth (very rough, "DDR" x2)
  const double mem_hz    = (double)mem_khz * 1000.0;              // Hz
  const double bytes_s   = mem_hz * (bus_bits / 8.0) * 2.0;       // B/s
  const double gbps_theo = (mem_khz > 0 && bus_bits > 0) ? (bytes_s / 1e9) : 0.0;

  // Output (same keys as the CUDA snippet)
  std::printf("GPU_NAME=%s\n", p.name);
  std::printf("SM=%d\n", p.multiProcessorCount);
  std::printf("CC=%d%d\n", p.major, p.minor); // may be 00 on AMD; acceptable for this snapshot
  std::printf("FP32_TFLOPS_PEAK=%.3f\n", tflops);
  std::printf("MEM1_GBPS_PEAK_THEO=%.2f\n", gbps_theo);
  return 0;
}
