/**
 * mem_bw.cu — GPU memory bandwidth microbenchmark
 *
 * What this program does
 * ----------------------
 * Measures two bandwidths across a range of payload sizes and reports each as an
 * absolute GB/s number and an efficiency (normalized to a *theoretical* peak):
 *
 *   1) mem1: device DRAM bandwidth using a STREAM-like triad kernel
 *      (two reads + one write).
 *   2) mem2: best of host→device (H2D) and device→host (D2H) copy bandwidth
 *      using pinned host memory.
 *
 * How “theoretical peaks” are computed
 * ------------------------------------
 * • DRAM (mem1): from CUDA device properties
 *     peak_GBps = 2 * memoryClockRate(Hz) * memoryBusWidth(bytes) / 1e9
 *   (factor 2 for DDR).
 * • Host↔GPU interconnect (mem2): the measured large-payload plateau is snapped
 *   to the nearest canonical PCIe peak:
 *     {7.88, 15.75, 31.50, 63.00} GB/s  // Gen3x8, Gen3x16, Gen4x16, Gen5x16
 *
 * Main features
 * -------------
 * • Fixed, LLM-oriented payload sweep (MB): 256, 64, 8, 1, 0.25, 0.0625.
 * • STREAM triad kernel for mem1; throughput computed as 3×bytes / time
 *   (two reads + one write per element).
 * • Pinned host buffers and cudaMemcpyAsync for mem2; reports the max of
 *   H2D and D2H for each size.
 * • Brief warm-ups and size-dependent repetition counts for stable averages.
 * • Normalized “efficiency” per size and compact summary lines that are easy
 *   to parse or plot offline.
 *
 * Options
 * -------
 * This program takes no command-line options (main() has no argv).
 * Defaults:
 *   • GPU device: 0 (current device).
 *   • Sizes & blocks: edit the `sizes[]` table in the source if needed.
 * Tips:
 *   • To choose a device at runtime, use the environment variable
 *     CUDA_VISIBLE_DEVICES or call cudaSetDevice() before running the benchmark.
 *
 * Build
 * -----
 * Requires CUDA Toolkit.
 *
 *   nvcc -O3 -std=c++14 mem_bw.cu -o mem_bw
 *
 * (On Windows, produce an .exe; on Linux/macOS you can keep the same command.
 *  No special libraries beyond the CUDA runtime are needed.)
 *
 * Usage
 * -----
 *   ./mem_bw
 *
 * What the output looks like
 * --------------------------
 * Header + one row per payload size, then two summary lines:
 *
 *   === Measuring Memory Bandwidths (mem1/mem2) ===
 *      256     MB : mem1(stream)=   850.12 GB/s (eff=0.9428)  |  mem2(best H2D/D2H)=   31.20 GB/s (eff=0.9906)
 *       64     MB : mem1(stream)=   812.34 GB/s (eff=0.9000)  |  mem2(best H2D/D2H)=   30.85 GB/s (eff=0.9808)
 *        8     MB : mem1(stream)=   700.11 GB/s (eff=0.7750)  |  mem2(best H2D/D2H)=   28.90 GB/s (eff=0.9175)
 *        1     MB : ...
 *     0.25     MB : ...
 *     0.0625   MB : ...
 *
 *   MEM1_SUMMARY GiB=<device_VRAM_GiB> GBps=<theory_DRAM_GBps> MB_eff=[(256,0.94),(64,0.90),(8,0.78),(1,0.65),(0.25,0.52),(0.0625,0.40),(0,0.40)]
 *   MEM2_SUMMARY GiB=<host_RAM_GiB>    GBps=<theory_PCIe_GBps> MB_eff=[(256,1.00),(64,0.98),(8,0.92),(1,0.70),(0.25,0.50),(0.0625,0.30),(0,0.30)]
 *
 * Notes on the fields
 * -------------------
 * • Per-size rows:
 *     mem1(stream)   = measured DRAM GB/s for the triad kernel at that size.
 *     mem2(best ...) = max(H2D, D2H) GB/s at that size with pinned memory.
 *     eff            = min(1.0, measured / theoretical_peak).
 * • MEM*_SUMMARY:
 *     GiB       = total memory (device VRAM for mem1, host RAM for mem2).
 *     GBps      = theoretical peak used for normalization.
 *     MB_eff    = list of (payload_MB, efficiency) points plus a tail at x=0
 *                 equal to the last point’s efficiency (handy for plotting
 *                 stepped curves).
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
static inline void ck(cudaError_t e, const char* w){
  if(e!=cudaSuccess){ std::fprintf(stderr,"CUDA error %d (%s) at %s\n",(int)e,cudaGetErrorString(e),w); std::exit(1); }
}
static inline double now_ms(){
  using clk = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}
static inline size_t MB(double x){ return (size_t) llround(x * 1024.0 * 1024.0); }

// Host RAM GiB from /proc/meminfo
static long long host_total_gib(){
  FILE* f = std::fopen("/proc/meminfo","r");
  if(!f) return 0;
  char key[64]={0}, unit[32]={0};
  long long kB=0, val=0;
  while(std::fscanf(f, "%63s %lld %31s", key, &val, unit) == 3){
    if(std::strcmp(key,"MemTotal:")==0){ kB = val; break; }
  }
  std::fclose(f);
  if(kB<=0) return 0;
  return (long long) llround((double)kB / (1024.0*1024.0)); // kB -> GiB
}

// Snap a measured plateau to canonical PCIe peaks (GB/s).
static double snap_pcie_peak(double plateau_gbps){
  const double cand[] = {7.88, 15.75, 31.50, 63.00}; // Gen3x8, Gen3x16, Gen4x16, Gen5x16
  double best = cand[0], bestd = std::fabs(plateau_gbps - cand[0]);
  for(size_t i=1;i<sizeof(cand)/sizeof(cand[0]);++i){
    double d = std::fabs(plateau_gbps - cand[i]);
    if(d < bestd){ bestd = d; best = cand[i]; }
  }
  return best;
}

// STREAM-like triad kernel (two reads + one write)
__global__ void triad(float* __restrict__ a, const float* __restrict__ b,
                      const float* __restrict__ c, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<n) a[i] = b[i] + c[i];
}

struct SizeCase { double mb; int blk; };

int main(){
  // Sizes representative of LLM tensor shards (MB)
  const SizeCase sizes[] = {
    {256.0,   64},
    { 64.0,  128},
    {  8.0,  512},
    {  1.0,  256},
    {0.25,  1024},
    {0.0625,1024}
  };
  const int S = (int)(sizeof(sizes)/sizeof(sizes[0]));
  std::vector<double> sizes_mb; sizes_mb.reserve(S);
  for(int i=0;i<S;i++) sizes_mb.push_back(sizes[i].mb);

  int dev=0; ck(cudaGetDevice(&dev),"getdev");
  cudaDeviceProp p{}; ck(cudaGetDeviceProperties(&p, dev),"getprops");

  // --- Theoretical mem1 (DRAM) ---
  // memoryClockRate is in kHz; bus width in bits; DDR factor=2.
  double mem_clk_hz   = (double)p.memoryClockRate * 1000.0;   // kHz -> Hz
  double bus_bytes    = (double)p.memoryBusWidth / 8.0;       // bits -> bytes
  double dram_theory_gbps = 0.0;
  if(mem_clk_hz > 0.0 && bus_bytes > 0.0){
    dram_theory_gbps = (2.0 * mem_clk_hz * bus_bytes) / 1e9;  // GB/s
  } else {
    // Conservative fallback (rare): assume “high 200s” GB/s for older cards.
    dram_theory_gbps = 250.0;
  }
  const long long vram_gib = (long long) llround((double)p.totalGlobalMem / (1024.0*1024.0*1024.0));
  const long long host_gib = host_total_gib();

  // --- Measure mem1 STREAM triad ---
  std::vector<double> mem1_gbps; mem1_gbps.reserve(S);
  for(int i=0;i<S;i++){
    const size_t bytes = MB(sizes[i].mb);
    const int n = (int)(bytes / sizeof(float));
    if(n<=0){ mem1_gbps.push_back(0.0); continue; }
    float *a=nullptr,*b=nullptr,*c=nullptr;
    ck(cudaMalloc(&a, bytes),"a"); ck(cudaMalloc(&b, bytes),"b"); ck(cudaMalloc(&c, bytes),"c");
    const int blk  = sizes[i].blk;
    const int grid = std::max(1, (n + blk - 1) / blk);

    // warmup
    for(int w=0; w<3; ++w) triad<<<grid,blk>>>(a,b,c,n);
    ck(cudaDeviceSynchronize(),"warm");

    // reps tuned per size
    int reps = 200;
    if(sizes[i].mb >= 256.0) reps = 120;
    else if(sizes[i].mb >= 64.0) reps = 160;
    else if(sizes[i].mb <= 0.25) reps = 800;

    double t0=now_ms();
    for(int r=0;r<reps;r++) triad<<<grid,blk>>>(a,b,c,n);
    ck(cudaDeviceSynchronize(),"sync");
    double ms=(now_ms()-t0)/reps;

    // 3*bytes per iter (2 reads + 1 write)
    double gbps = (3.0 * (double)bytes) / (ms * 1e6);
    mem1_gbps.push_back(gbps);

    cudaFree(c); cudaFree(b); cudaFree(a);
  }

  // --- Measure mem2 H2D/D2H (pinned) ---
  std::vector<double> mem2_best; mem2_best.reserve(S);
  for(int i=0;i<S;i++){
    const size_t bytes = MB(sizes[i].mb);
    float* d=nullptr; ck(cudaMalloc(&d, bytes),"d");
    void*  h=nullptr; ck(cudaHostAlloc(&h, bytes, cudaHostAllocDefault),"h");

    // warmup
    ck(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, 0),"h2d-w");
    ck(cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, 0),"d2h-w");
    ck(cudaDeviceSynchronize(),"wm");

    int reps = 400;
    if(sizes[i].mb >= 256.0) reps = 120;
    else if(sizes[i].mb >=  64.0) reps = 240;

    // H2D
    double t0=now_ms();
    for(int r=0;r<reps;r++) ck(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, 0),"h2d");
    ck(cudaDeviceSynchronize(),"h2d-s");
    double ms=(now_ms()-t0)/reps;
    double gbps_h2d = (double)bytes / (ms * 1e6);

    // D2H
    t0=now_ms();
    for(int r=0;r<reps;r++) ck(cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, 0),"d2h");
    ck(cudaDeviceSynchronize(),"d2h-s");
    ms=(now_ms()-t0)/reps;
    double gbps_d2h = (double)bytes / (ms * 1e6);

    mem2_best.push_back(std::max(gbps_h2d, gbps_d2h));

    cudaFreeHost(h); cudaFree(d);
  }

  // Infer theoretical interconnect GB/s from large-payload plateau
  double mem2_plateau = 0.0;
  for(int i=0;i<S;i++) if(sizes[i].mb >= 8.0) mem2_plateau = std::max(mem2_plateau, mem2_best[i]);
  if(mem2_plateau <= 0.0) for(double v: mem2_best) mem2_plateau = std::max(mem2_plateau, v);
  if(mem2_plateau <= 0.0) mem2_plateau = 12.0; // very conservative fallback
  const double interconnect_theory_gbps = snap_pcie_peak(mem2_plateau);

  // ---- Print ----
  std::puts("=== Measuring Memory Bandwidths (mem1/mem2) ===");
  for(int i=0;i<S;i++){
    const double eff1 = std::min(1.0, mem1_gbps[i]/dram_theory_gbps);
    const double eff2 = std::min(1.0, mem2_best[i]/interconnect_theory_gbps);
    std::printf("%10.4g MB : mem1(stream)= %8.2f GB/s (eff=%0.4f)  |  mem2(best H2D/D2H)= %8.2f GB/s (eff=%0.4f)\n",
      sizes_mb[i], mem1_gbps[i], eff1, mem2_best[i], eff2);
  }

  auto print_mb_eff = [&](const char* tag, long long gib, double theory_gbps, const std::vector<double>& series){
    std::printf("%s GiB=%lld GBps=%lld MB_eff=[",
                tag, gib, (long long) llround(theory_gbps));
    for(int i=0;i<S;i++){
      const double eff = std::min(1.0, series[i]/theory_gbps);
      std::printf("(%g,%0.4f)%s", sizes_mb[i], eff, (i+1<S?",":""));
    }
    const double tail_eff = std::min(1.0, series.back()/theory_gbps);
    std::printf(",(0,%0.4f)]\n", tail_eff);
  };

  print_mb_eff("MEM1_SUMMARY", vram_gib, dram_theory_gbps, mem1_gbps);
  print_mb_eff("MEM2_SUMMARY", host_gib, interconnect_theory_gbps, mem2_best);

  return 0;
}