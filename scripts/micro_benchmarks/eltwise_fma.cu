/**
 * @file eltwise_fma.cu
 * @brief Element-wise FP32 fused-multiply-add (FMA) micro-benchmark.
 *
 * WHAT THIS IS
 * ------------
 * A tiny, ALU-bound CUDA benchmark that measures sustained FP32 throughput
 * using a vectorized FMA kernel with a single global store per thread (to
 * keep memory traffic minimal). It runs a few preset “workload labels”
 * {16, 4, 1, 0} which correspond to ~16 GFLOPs, 4 GFLOPs, 1 GFLOP, and a
 * very small non-zero workload, then reports:
 *   • measured GFLOP/s, normalized as efficiency vs a coarse device FP32 peak.
 *
 * MAIN FEATURES
 * -------------
 * • Compute-bound kernel: unrolled groups of 4 FMAs per iteration with one
 *   final write, so results reflect ALU throughput rather than memory.
 * • Auto-sizing of work: for each target workload the program chooses
 *   (N_used, FMAs per element, #launches) to approach the requested FLOPs.
 * • Peak estimate: derives a rough FP32 TFLOPs “peak” from device properties
 *   (SM count × cores/SM × clock × 2 flops per FMA).
 * • Robust allocation: attempts to allocate N_ELEMS floats; on OOM it halves
 *   the size until success so it can still run and emit a parseable table.
 * • Stable timing: warmup launch + CUDA events for wall-time measurement.
 *
 * OPTIONS (environment variables)
 * -------------------------------
 *  N_ELEMS
 *    Base number of elements to allocate and (at most) process.
 *    • Type: int
 *    • Default: 1<<24  (≈16.7M elements; ~64 MiB for the output buffer)
 *    • Behavior: If allocation fails, the program halves this value until it
 *      succeeds (or reaches zero, in which case it still prints a zeroed table).
 *
 *  FMAS_PER_ELT
 *    Baseline number of FMAs each element performs.
 *    • Type: int
 *    • Default: 512
 *    • Notes: Internally rounded to a multiple of 4 (the unroll factor).
 *      The planner may shrink this to hit a small target workload.
 *
 *  BLOCK
 *    CUDA block size (threads per block).
 *    • Type: int
 *    • Default: 256
 *    • Range: clamped to [32, 1024]
 *
 * HOW TO BUILD
 * ------------
 * Minimum:
 *   nvcc -O3 -std=c++14 eltwise_fma.cu -o builds/eltwise_fma
 *
 * Recommended (set your architecture as appropriate, e.g. sm_90, sm_80, sm_86):
 *   mkdir -p builds
 *   nvcc -O3 -std=c++14 -arch=sm_80 eltwise_fma.cu -o builds/eltwise_fma
 *
 * HOW TO RUN
 * ----------
 * Default settings:
 *   ./builds/eltwise_fma
 *
 * With custom parameters (examples):
 *   export N_ELEMS=$((1<<25))       # ~128 MiB output
 *   export FMAS_PER_ELT=1024
 *   export BLOCK=512
 *   ./builds/eltwise_fma
 *
 * WHAT THE OUTPUT LOOKS LIKE
 * --------------------------
 * The program emits a compact, parseable table for FP32:
 *
 *   Vector table (float32):
 *     tflops=83.58
 *     gflops_efficiency=[[16,0.8921],[4,0.8743],[1,0.8237],[0,0.7420]]
 *
 * Meaning:
 *   • tflops: theoretical peak FP32 TFLOPs estimate for the current GPU.
 *   • gflops_efficiency:
 *       Pairs of [label, efficiency], where label is the target workload in
 *       GFLOPs (16, 4, 1, or a tiny non-zero for 0). Efficiency is measured
 *       GFLOP/s divided by (peak_tflops * 1000).
 *
 * IMPLEMENTATION NOTES
 * --------------------
 * • Kernel: each thread does repeated `fmaf` operations on register values:
 *     - 4 FMAs per loop iteration (unrolled), minimal dependence chain.
 *     - One final global store (`out[idx] = a + b + c + d`) to avoid memory
 *       bottlenecks.
 * • Planning: for each target FLOP budget, the code picks a combination of
 *   vector length, FMAs/element, and number of launches to approximate that
 *   budget without excessive overshoot.
 * • Peak model: cores/SM are a coarse mapping from compute capability; the
 *   estimate is sufficient for relative efficiency comparisons but not a
 *   strict hardware spec.
 *
 * EXIT / FAILURE BEHAVIOR
 * -----------------------
 * • If the output buffer cannot be allocated at any size, the program still
 *   prints a valid table with zeros for efficiencies so downstream parsers
 *   won’t break.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <limits>

#include <cuda_runtime.h>

// ----------- helpers -----------
static inline int getenv_int(const char* k, int defv){
  if(const char* s = std::getenv(k)) return std::atoi(s);
  return defv;
}
static inline void ckCuda(cudaError_t e, const char* where){
  if(e != cudaSuccess){
    std::fprintf(stderr, "CUDA error %d (%s) at %s\n", (int)e, cudaGetErrorString(e), where);
    std::exit(1);
  }
}

// Coarse FP32 peak estimate
static int fp32_cores_per_sm(int maj, int min){
  const int cc = maj*10 + min;
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
static double fp32_tflops_peak(){
  int dev = 0; ckCuda(cudaGetDevice(&dev), "cudaGetDevice");
  cudaDeviceProp p{};
  ckCuda(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");
  const double sms   = (double)p.multiProcessorCount;
  const double cores = (double)fp32_cores_per_sm(p.major, p.minor);
  const double hz    = (double)p.clockRate * 1000.0; // kHz -> Hz
  return sms * cores * hz * 2.0 / 1e12;              // FP32 FMA: 2 flops
}

// ----------- kernel -----------
__global__ void vec_fma_kernel(float* __restrict__ out, int n, int fmas_per_elt){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= n) return;

  float a = 1.000001f * (float)((idx & 1023) + 1);
  float b = 0.999991f * (float)(((idx >> 2) & 1023) + 2);
  float c = 1.000013f * (float)(((idx >> 4) &  511) + 3);
  float d = 0.999989f * (float)(((idx >> 6) &  255) + 4);

  // do FMAs in groups of 4
  const int per_iter = 4;
  int iters = fmas_per_elt / per_iter;
  #pragma unroll 64
  for(int t=0; t<iters; ++t){
    a = fmaf(a, b, c);
    b = fmaf(b, c, d);
    c = fmaf(c, d, a);
    d = fmaf(d, a, b);
  }
  out[idx] = a + b + c + d; // single global store
}

// Round FMAs/elt to a multiple of 4 (what kernel expects)
static int round_fmas(int x){
  if(x < 4) return 4;
  return (x + 3) / 4 * 4;
}

// Try to allocate output buffer; on OOM halve N until success or N==0
static int try_alloc(float** dOut, int& N){
  while(N > 0){
    size_t bytes = (size_t)N * sizeof(float);
    cudaError_t e = cudaMalloc((void**)dOut, bytes);
    if(e == cudaSuccess) return 0;
    N >>= 1;
  }
  return -1;
}

// Choose (N_used, fmas_per_elt, launches) to reach target_flops (~ GFLOPs * 1e9)
static void plan_work(double target_flops, int N_alloc, int fmas_base,
                      int& N_used, int& fmas_per_elt, int& launches)
{
  // Avoid zero; map label "0" to a small non-zero workload (0.25 GFLOPs)
  if(target_flops <= 0.0) target_flops = 0.25e9;

  N_used = N_alloc;
  fmas_per_elt = round_fmas(fmas_base);
  const auto flops_per_launch = [&](int Np, int Fp)->double{
    return 2.0 * (double)Np * (double)Fp; // FMAs -> flops
  };

  double fpl = flops_per_launch(N_used, fmas_per_elt);

  // If a single launch overshoots badly, try to shrink FMAs/elt, then N_used.
  if(fpl > target_flops){
    // First shrink FMAs/elt
    int need_fmas = (int)std::floor(target_flops / (2.0 * (double)N_used));
    need_fmas = round_fmas(need_fmas);
    if(need_fmas >= 4){
      fmas_per_elt = need_fmas;
    }else{
      // With minimum FMAs, we still overshoot: shrink N_used
      fmas_per_elt = 4;
      double n_need = std::floor(target_flops / (2.0 * (double)fmas_per_elt));
      if(n_need < 1.0) n_need = 1.0;
      if(n_need > (double)N_alloc) n_need = (double)N_alloc;
      N_used = (int)n_need;
    }
    fpl = flops_per_launch(N_used, fmas_per_elt);
  }

  // Choose number of launches to get close to target
  launches = (int)std::ceil(target_flops / std::max(1.0, fpl));
  launches = std::max(1, std::min(launches, 10000));
}

// Measure GFLOP/s and efficiency for one workload
static void run_case(float* dOut, int N_used, int fmas_per_elt, int launches,
                     int block, double peak_tflops,
                     double& gflops_out, double& eff_out)
{
  int grid = (N_used + block - 1) / block;
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

  // warmup
  vec_fma_kernel<<<grid, block>>>(dOut, N_used, fmas_per_elt);
  cudaDeviceSynchronize();

  cudaEventRecord(e0);
  for(int r=0; r<launches; ++r){
    vec_fma_kernel<<<grid, block>>>(dOut, N_used, fmas_per_elt);
  }
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms = 0.0f; cudaEventElapsedTime(&ms, e0, e1);

  const double total_flops = 2.0 * (double)N_used * (double)fmas_per_elt * (double)launches;
  const double gflops = total_flops / (ms * 1e6);
  const double eff = (peak_tflops > 0.0) ? (gflops / (peak_tflops * 1000.0)) : 0.0;

  gflops_out = gflops;
  eff_out = eff;

  cudaEventDestroy(e0); cudaEventDestroy(e1);
}

int main(){
  // Parameters (with conservative defaults)
  int    N_alloc   = getenv_int("N_ELEMS",       1<<24); // ~64 MiB output
  int    fmas_base = getenv_int("FMAS_PER_ELT",  512);
  int    block     = getenv_int("BLOCK",         256);
  block = std::max(32, std::min(1024, block));

  // Allocate output buffer (halve on OOM)
  float* dOut = nullptr;
  if(try_alloc(&dOut, N_alloc) != 0 || N_alloc <= 0){
    // Still print a valid (but zeroed) table so callers can parse
    const double peak = fp32_tflops_peak();
    std::puts("Vector table (float32):");
    std::printf("  tflops=%.2f\n", peak);
    std::printf("  gflops_efficiency=[[16,%.4f],[4,%.4f],[1,%.4f],[0,%.4f]]\n",
                0.0,0.0,0.0,0.0);
    return 0;
  }

  const double peak = fp32_tflops_peak();

  // Targets in GFLOPs (label → actual work in flops)
  struct Target { int label; double flops; };
  std::vector<Target> tgts = {
    {16, 16.0e9}, {4, 4.0e9}, {1, 1.0e9}, {0, 0.25e9} // label 0 -> small non-zero
  };

  double eff16=0, eff4=0, eff1=0, eff0=0;

  for(const auto& t : tgts){
    int N_used, fmas_per_elt, launches;
    plan_work(t.flops, N_alloc, fmas_base, N_used, fmas_per_elt, launches);

    double gflops=0, eff=0;
    run_case(dOut, N_used, fmas_per_elt, launches, block, peak, gflops, eff);

    if(t.label==16) eff16=eff;
    else if(t.label==4) eff4=eff;
    else if(t.label==1) eff1=eff;
    else eff0=eff;
  }

  // Final table only
  std::puts("Vector table (float32):");
  std::printf("  tflops=%.2f\n", peak);
  std::printf("  gflops_efficiency=[[16,%.4f],[4,%.4f],[1,%.4f],[0,%.4f]]\n",
              eff16, eff4, eff1, eff0);

  cudaFree(dOut);
  return 0;
}