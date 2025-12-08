/**
 * mem_bw.cu — GPU memory bandwidth microbenchmark (CUDA)
 *
 * PURPOSE
 * -------
 * Measure two bandwidths across a small, LLM-oriented sweep of payload sizes and
 * report each as an absolute GB/s and an efficiency vs a *theoretical* peak:
 *
 *   (1) mem1 — device DRAM bandwidth via a STREAM-like triad (two reads + one write).
 *   (2) mem2 — best of host→device (H2D) and device→host (D2H) copy bandwidth using
 *               pinned host memory.
 *
 * SCOPE & ASSUMPTIONS
 * -------------------
 * • Single-device benchmark. It does not exercise GPU↔GPU fabrics (e.g., NVLink)
 *   or any cluster/network interconnects. It measures only GPU local DRAM (mem1)
 *   and the CPU↔GPU link (mem2).
 * • On multi-GPU systems, each GPU appears as its own CUDA device. Run once per
 *   visible device (use CUDA_VISIBLE_DEVICES to select).
 * • Results depend on CPU NUMA placement, PCIe/NVLink settings, and pinned-memory
 *   availability. For stable mem2, pin the process near the GPU’s NUMA node.
 *
 * HOW THE “THEORETICAL PEAKS” ARE DERIVED
 * ---------------------------------------
 * • DRAM (mem1):
 *     peak_GBps ≈ 2 * memoryClockRate(Hz) * memoryBusWidth(bytes) / 1e9
 *   (factor 2 for DDR). Values are read from CUDA device attributes/properties.
 *   If unavailable, a conservative fallback is used to keep efficiencies defined.
 *
 * • CPU↔GPU interconnect (mem2):
 *   The measured large-payload plateau is snapped to the nearest canonical PCIe
 *   peak to normalize efficiency:
 *       {7.88, 15.75, 31.50, 63.00} GB/s   // Gen3x8, Gen3x16, Gen4x16, Gen5x16
 *   (This is a heuristic for easy comparison across systems.)
 *
 * WHAT IT DOES
 * ------------
 * • Payload sizes (MB): 256, 64, 8, 1, 0.25, 0.0625.
 * • mem1: STREAM-style triad kernel; throughput computed as 3×bytes / time
 *   (two reads + one write per element).
 * • mem2: pinned host buffers + cudaMemcpyAsync; report max(H2D, D2H) per size.
 * • Warm-ups and size-dependent repetition counts for stable timing.
 * • Prints per-size rows and two compact summary lines that are easy to parse.
 *
 * BUILD
 * -----
 * CUDA toolchain:
 *   nvcc -O3 -std=c++14 mem_bw.cu -o mem_bw
 *
 * RUN
 * ---
 *   ./mem_bw_cuda
 *   # Select device(s):
 *   export CUDA_VISIBLE_DEVICES=0
 *
 * SAMPLE OUTPUT
 * -------------
 *   === Measuring Memory Bandwidths (mem1/mem2) ===
 *        256 MB : mem1(stream)=   850.12 GB/s (eff=0.9428)  |  mem2(best H2D/D2H)=   31.20 GB/s (eff=0.9906)
 *         64 MB : mem1(stream)=   812.34 GB/s (eff=0.9000)  |  mem2(best H2D/D2H)=   30.85 GB/s (eff=0.9808)
 *         ...
 *
 *   MEM1_SUMMARY GiB=<device_VRAM_GiB> GBps=<theory_DRAM_GBps> MB_eff=[(256,0.94),(64,0.90),...,(0,tail)]
 *   MEM2_SUMMARY GiB=<host_RAM_GiB>    GBps=<theory_PCIe_GBps> MB_eff=[(256,1.00),(64,0.98),...,(0,tail)]
 *
 * IMPLEMENTATION NOTES
 * --------------------
 * • Kernel: __global__ triad(float* a, const float* b, const float* c, int n) with
 *   <<<grid,block>>>; minimal arithmetic to expose memory bandwidth.
 * • mem2 uses cudaMallocHost (pinned) and cudaMemcpyAsync for both directions.
 * • Timing uses host wall-clock; short warm-ups precede measurement loops.
 * • Total memory sizes for summaries:
 *     - Device VRAM: cudaDeviceProp::totalGlobalMem (GiB).
 *     - Host RAM: parsed from /proc/meminfo (Linux).
 *
 * LIMITATIONS
 * -----------
 * • Not a full STREAM implementation; intended for quick baselining.
 * • On some stacks memoryClockRate/memoryBusWidth may be 0; we fall back to
 *   a conservative DRAM peak so efficiencies remain meaningful.
 * • mem2 reflects CPU↔GPU link only; it does not measure GPU↔GPU NVLink bandwidth.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>

// ---------- helpers ----------
static inline void ckCuda(cudaError_t e, const char* w){
  if(e != cudaSuccess){
    std::fprintf(stderr, "CUDA error %d (%s) at %s\n",
                 (int)e, cudaGetErrorString(e), w);
    std::exit(1);
  }
}

static inline double now_ms(){
  using clk = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
           clk::now().time_since_epoch()
         ).count();
}

static inline size_t MB(double x){
  return (size_t) llround(x * 1024.0 * 1024.0);
}

// Host RAM GiB from /proc/meminfo
static long long host_total_gib(){
  FILE* f = std::fopen("/proc/meminfo","r");
  if(!f) return 0;
  char key[64] = {0}, unit[32] = {0};
  long long kB = 0, val = 0;
  while(std::fscanf(f, "%63s %lld %31s", key, &val, unit) == 3){
    if(std::strcmp(key,"MemTotal:") == 0){
      kB = val;
      break;
    }
  }
  std::fclose(f);
  if(kB <= 0) return 0;
  return (long long) llround((double)kB / (1024.0 * 1024.0)); // kB -> GiB
}

// Snap a measured plateau to canonical PCIe peaks (GB/s).
static double snap_pcie_peak(double plateau_gbps){
  const double cand[] = {7.88, 15.75, 31.50, 63.00}; // Gen3x8, Gen3x16, Gen4x16, Gen5x16
  double best = cand[0];
  double bestd = std::fabs(plateau_gbps - cand[0]);
  for(size_t i=1;i<sizeof(cand)/sizeof(cand[0]);++i){
    double d = std::fabs(plateau_gbps - cand[i]);
    if(d < bestd){
      bestd = d;
      best = cand[i];
    }
  }
  return best;
}

// STREAM-like triad kernel (two reads + one write)
__global__ void triad(float* __restrict__ a,
                      const float* __restrict__ b,
                      const float* __restrict__ c,
                      int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n) a[i] = b[i] + c[i];
}

struct SizeCase { double mb; int blk; };

// Fetch CUDA attribute with fallback
static int get_attr_i(int dev, cudaDeviceAttr attr){
  int v = 0;
  cudaError_t e = cudaDeviceGetAttribute(&v, attr, dev);
  return (e == cudaSuccess) ? v : 0;
}

int main(){
  // Sizes representative of LLM tensor shards (MB)
  const SizeCase sizes[] = {
    {256.0,    64},
    { 64.0,   128},
    {  8.0,   512},
    {  1.0,   256},
    {0.25,   1024},
    {0.0625, 1024}
  };
  const int S = (int)(sizeof(sizes)/sizeof(sizes[0]));
  std::vector<double> sizes_mb; sizes_mb.reserve(S);
  for(int i=0;i<S;i++) sizes_mb.push_back(sizes[i].mb);

  int dev = 0;
  ckCuda(cudaGetDevice(&dev), "cudaGetDevice");
  cudaDeviceProp p{};
  ckCuda(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");

  // --- Theoretical mem1 (DRAM) ---
  // Try attributes first; fall back to cudaDeviceProp. Units expected ~kHz, bits.
  int mem_khz  = get_attr_i(dev, cudaDevAttrMemoryClockRate);
  int bus_bits = get_attr_i(dev, cudaDevAttrGlobalMemoryBusWidth);
  if(mem_khz  <= 0) mem_khz  = p.memoryClockRate; // may be 0 on some stacks
  if(bus_bits <= 0) bus_bits = p.memoryBusWidth;  // may be 0 on some stacks

  double dram_theory_gbps = 0.0;
  if(mem_khz > 0 && bus_bits > 0){
    const double mem_clk_hz = (double)mem_khz * 1000.0;  // kHz -> Hz
    const double bus_bytes  = (double)bus_bits / 8.0;    // bits -> bytes
    dram_theory_gbps = (2.0 * mem_clk_hz * bus_bytes) / 1e9; // GB/s (DDR x2)
  }else{
    // Conservative fallback (rare): a safe constant rather than 0.
    dram_theory_gbps = 250.0;
  }

  const long long vram_gib = (long long) llround(
      (double)p.totalGlobalMem / (1024.0*1024.0*1024.0));
  const long long host_gib = host_total_gib();

  // --- Measure mem1 STREAM triad ---
  std::vector<double> mem1_gbps; mem1_gbps.reserve(S);
  for(int i=0;i<S;i++){
    const size_t bytes = MB(sizes[i].mb);
    const int n = (int)(bytes / sizeof(float));
    if(n <= 0){
      mem1_gbps.push_back(0.0);
      continue;
    }

    float *a=nullptr, *b=nullptr, *c=nullptr;
    ckCuda(cudaMalloc(&a, bytes), "cudaMalloc a");
    ckCuda(cudaMalloc(&b, bytes), "cudaMalloc b");
    ckCuda(cudaMalloc(&c, bytes), "cudaMalloc c");

    const int blk  = sizes[i].blk;
    const int grid = std::max(1, (n + blk - 1) / blk);

    // warmup
    for(int w=0; w<3; ++w){
      triad<<<grid, blk>>>(a, b, c, n);
    }
    ckCuda(cudaDeviceSynchronize(), "warmup");

    // reps tuned per size
    int reps = 200;
    if(sizes[i].mb >= 256.0)      reps = 120;
    else if(sizes[i].mb >= 64.0) reps = 160;
    else if(sizes[i].mb <= 0.25) reps = 800;

    double t0 = now_ms();
    for(int r=0; r<reps; ++r){
      triad<<<grid, blk>>>(a, b, c, n);
    }
    ckCuda(cudaDeviceSynchronize(), "sync");
    double ms = (now_ms() - t0) / reps;

    // 3*bytes per iter (2 reads + 1 write)
    double gbps = (3.0 * (double)bytes) / (ms * 1e6);
    mem1_gbps.push_back(gbps);

    cudaFree(c); cudaFree(b); cudaFree(a);
  }

  // --- Measure mem2 H2D/D2H (pinned) ---
  std::vector<double> mem2_best; mem2_best.reserve(S);
  for(int i=0;i<S;i++){
    const size_t bytes = MB(sizes[i].mb);

    float* d = nullptr;
    ckCuda(cudaMalloc(&d, bytes), "cudaMalloc d");

    void* h = nullptr;
    ckCuda(cudaMallocHost(&h, bytes), "cudaMallocHost h");

    // warmup
    ckCuda(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, 0), "h2d-w");
    ckCuda(cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, 0), "d2h-w");
    ckCuda(cudaDeviceSynchronize(), "warm");

    int reps = 400;
    if(sizes[i].mb >= 256.0)      reps = 120;
    else if(sizes[i].mb >= 64.0) reps = 240;

    // H2D
    double t0 = now_ms();
    for(int r=0;r<reps;r++){
      ckCuda(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, 0), "h2d");
    }
    ckCuda(cudaDeviceSynchronize(), "h2d-s");
    double ms = (now_ms() - t0) / reps;
    double gbps_h2d = (double)bytes / (ms * 1e6);

    // D2H
    t0 = now_ms();
    for(int r=0;r<reps;r++){
      ckCuda(cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, 0), "d2h");
    }
    ckCuda(cudaDeviceSynchronize(), "d2h-s");
    ms = (now_ms() - t0) / reps;
    double gbps_d2h = (double)bytes / (ms * 1e6);

    mem2_best.push_back(std::max(gbps_h2d, gbps_d2h));

    cudaFreeHost(h);
    cudaFree(d);
  }

  // Infer theoretical interconnect GB/s from large-payload plateau
  double mem2_plateau = 0.0;
  for(int i=0;i<S;i++){
    if(sizes[i].mb >= 8.0)
      mem2_plateau = std::max(mem2_plateau, mem2_best[i]);
  }
  if(mem2_plateau <= 0.0){
    for(double v : mem2_best)
      mem2_plateau = std::max(mem2_plateau, v);
  }
  if(mem2_plateau <= 0.0) mem2_plateau = 12.0; // very conservative fallback
  const double interconnect_theory_gbps = snap_pcie_peak(mem2_plateau);

  // ---- Print ----
  std::puts("=== Measuring Memory Bandwidths (mem1/mem2) ===");
  for(int i=0;i<S;i++){
    const double eff1 = std::min(1.0, mem1_gbps[i] / dram_theory_gbps);
    const double eff2 = std::min(1.0, mem2_best[i] / interconnect_theory_gbps);
    std::printf("%10.4g MB : mem1(stream)= %8.2f GB/s (eff=%0.4f)  |  mem2(best H2D/D2H)= %8.2f GB/s (eff=%0.4f)\n",
                sizes_mb[i], mem1_gbps[i], eff1, mem2_best[i], eff2);
  }

  auto print_mb_eff = [&](const char* tag, long long gib, double theory_gbps,
                          const std::vector<double>& series){
    std::printf("%s GiB=%lld GBps=%lld MB_eff=[",
                tag, gib, (long long) llround(theory_gbps));
    for(int i=0;i<S;i++){
      const double eff = std::min(1.0, series[i] / theory_gbps);
      std::printf("(%g,%0.4f)%s", sizes_mb[i], eff, (i+1<S ? "," : ""));
    }
    const double tail_eff = std::min(1.0, series.back() / theory_gbps);
    std::printf(",(0,%0.4f)]\n", tail_eff);
  };

  print_mb_eff("MEM1_SUMMARY", vram_gib, dram_theory_gbps, mem1_gbps);
  print_mb_eff("MEM2_SUMMARY", host_gib, interconnect_theory_gbps, mem2_best);

  return 0;
}
