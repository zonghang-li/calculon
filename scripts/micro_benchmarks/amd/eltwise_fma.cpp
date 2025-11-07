/**
 * @file eltwise_fma.cpp
 * @brief Vector-engine FMA micro-benchmark (HIP/ROCm) with FP32 measurement and FP16/FP8
 *        efficiency reported by vector-peak scaling (NOT MFMA).
 *
 * WHAT THIS BENCHMARK MEASURES
 * ----------------------------
 * • A compute-bound FP32 vector FMA kernel (one global store per thread) to minimize
 *   memory traffic and isolate ALU throughput.
 * • It reports sustained GFLOP/s for several small "workload labels" {16, 4, 1, 0} (≈GFLOPs),
 *   then normalizes those numbers against coarse theoretical VECTOR peaks for:
 *     - float32  → measured, vector engine
 *     - float16  → derived from FP32 measurement; peak = FP32_vector_peak * VEC_FP16_SCALE
 *     - float8   → derived; peak = FP32_vector_peak * VEC_FP8_SCALE
 *
 * IMPORTANT: Vector vs Matrix
 * ---------------------------
 * • This is a VECTOR-engine benchmark. It does not exercise matrix cores (MFMA).
 * • FP16/FP8 rows are NOT measured in this program; they are re-scored from the SAME
 *   FP32 vector measurement using vector-peak scaling to help compare against code that
 *   uses vector math at reduced precision.
 * • For MFMA (matrix-core) behavior and efficiency, use the separate GEMM benchmark
 *   (gemm_bench.cpp), which normalizes against MFMA peaks per dtype.
 *
 * ARCH BEHAVIOR (FP8)
 * -------------------
 * • MI210 (gfx90a, CDNA2) has no FP8; by default this program HIDES the FP8 vector table.
 * • On CDNA3+ (gfx94*, gfx95*) the FP8 table is shown by default.
 * • You can override with: VEC_REPORT_FP8=1 to force show, VEC_REPORT_FP8=0 to hide.
 *
 * OUTPUT (parseable ASCII, no unicode)
 * ------------------------------------
 * Vector table (float32):
 *   tflops=22.63
 *   gflops_efficiency=[[16,0.6954],[4,0.5972],[1,0.3949],[0,0.1259]]
 * Vector table (float16):
 *   tflops=45.26
 *   gflops_efficiency=[[16,0.3477],[4,0.2986],[1,0.1975],[0,0.0630]]
 * Vector table (float8):    // printed only if enabled by arch or VEC_REPORT_FP8=1
 *   tflops=90.52
 *   gflops_efficiency=[[16,0.1738],[4,0.1493],[1,0.0987],[0,0.0315]]
 *
 * EFFICIENCY DEFINITION
 * ---------------------
 *   efficiency = measured_GFLOP_per_s / (peak_tflops * 1000)
 * where peak_tflops is a coarse VECTOR peak computed as:
 *   FP32_vector_peak ≈ (#CUs or SMs) * (FP32 lanes per CU/SM) * (clock_Hz) * 2 / 1e12
 * and dtype vector peaks are scaled:
 *   FP16_vector_peak = FP32_vector_peak * VEC_FP16_SCALE   (default 2.0)
 *   FP8_vector_peak  = FP32_vector_peak * VEC_FP8_SCALE    (default 4.0)
 *
 * ENVIRONMENT VARIABLES
 * ---------------------
 * • N_ELEMS           : base number of elements to allocate (default: 1<<24)
 * • FMAS_PER_ELT      : base FMAs per element (rounded to multiple of 4; default: 512)
 * • BLOCK             : threads per block (default: 256; clamp to [32, 1024])
 * • VEC_FP16_SCALE    : vector FP16 peak multiplier vs FP32 (default: 2.0)
 * • VEC_FP8_SCALE     : vector FP8  peak multiplier vs FP32 (default: 4.0)
 * • VEC_REPORT_FP8    : 1 to print FP8 table, 0 to hide (default: arch_has_fp8())
 *
 * ROBUSTNESS / OOM HANDLING
 * -------------------------
 * • If allocation fails, the program halves N_ELEMS repeatedly. If it cannot allocate
 *   any buffer, it still prints valid tables with zeros so downstream parsers do not break.
 *
 * TIMING
 * ------
 * • Warmup launch followed by HIP event timing across repeated launches.
 *
 * BUILD
 * -----
 * • hipcc -O3 -std=c++14 eltwise_fma.cpp -o eltwise_fma
 * • Prefer passing --offload-arch=<gfx*> via your build system (e.g., gfx90a for MI210).
 *
 * NOTES
 * -----
 * • Single-device benchmark; if a card is multi-die (e.g., MI250), run once per visible GCD.
 * • On NVIDIA (HIP platform), a coarse SM-lane model is used to compute the vector FP32 peak.
 * • This file’s output is ASCII-only (no em-dashes) to avoid parser issues.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>

#include <hip/hip_runtime.h>

// ----------------- tiny utils -----------------
static inline int getenv_int(const char* k, int defv){
  if(const char* s = std::getenv(k)) return std::atoi(s);
  return defv;
}
static inline double getenv_double(const char* k, double defv){
  if(const char* s = std::getenv(k)) return std::atof(s);
  return defv;
}
static inline void ckHip(hipError_t e, const char* where){
  if(e != hipSuccess){
    std::fprintf(stderr, "HIP error %d (%s) at %s\n",
                 (int)e, hipGetErrorString(e), where);
    std::exit(1);
  }
}

// Arch gate for printing FP8 vector table (reflect HW capability)
static bool arch_has_fp8(){
  int dev = 0; ckHip(hipGetDevice(&dev), "hipGetDevice");
  hipDeviceProp_t p{}; ckHip(hipGetDeviceProperties(&p, dev), "hipGetDeviceProperties");
  const char* g = p.gcnArchName ? p.gcnArchName : "";
  // CDNA3+: gfx94*, gfx95*; MI210 is gfx90a => false
  return std::strncmp(g, "gfx94", 5) == 0 || std::strncmp(g, "gfx95", 5) == 0;
}

// ----------------- vector peak model -----------------
// FP32 "cores per MP" (lanes per CU/SM) for vector ALUs.
static int fp32_cores_per_mp(const hipDeviceProp_t& p){
#if defined(__HIP_PLATFORM_NVIDIA__)
  // Coarse CUDA mapping (kept for portability)
  const int cc = p.major*10 + p.minor;
  if(cc >= 90) return 128;  // Hopper/Ada fallback
  if(cc >= 86) return 128;  // Ampere GA10x
  if(cc >= 80) return  64;  // A100
  if(cc >= 75) return  64;  // Turing
  if(cc >= 70) return  64;  // Volta
  if(cc >= 61) return 128;  // Pascal GP10x
  if(cc >= 60) return 128;  // Pascal
  if(cc >= 50) return 128;  // Maxwell fallback
  return 64;
#else
  // AMD CDNA/RDNA default: 64 FP32 vector lanes per CU
  (void)p;
  return 64;
#endif
}

static double device_clock_hz(){
  int dev = 0; ckHip(hipGetDevice(&dev), "hipGetDevice");
  int clock_khz = 0;
  hipError_t a = hipDeviceGetAttribute(&clock_khz, hipDeviceAttributeClockRate, dev);
  if(a != hipSuccess || clock_khz <= 0){
    hipDeviceProp_t p{};
    if(hipGetDeviceProperties(&p, dev) == hipSuccess && p.clockRate > 0){
      clock_khz = p.clockRate; // kHz
    }
  }
  return (clock_khz > 0) ? (double)clock_khz * 1000.0 : 1.0e9; // conservative fallback
}

static double vector_fp32_peak_tflops(){
  int dev = 0; ckHip(hipGetDevice(&dev), "hipGetDevice");
  hipDeviceProp_t p{}; ckHip(hipGetDeviceProperties(&p, dev), "hipGetDeviceProperties");
  const double mps   = (double)p.multiProcessorCount;  // CUs (AMD) / SMs (NVIDIA)
  const double cores = (double)fp32_cores_per_mp(p);   // FP32 vector lanes per CU/SM
  const double hz    = device_clock_hz();              // Hz
  // One FMA per lane per cycle => 2 flops
  return mps * cores * hz * 2.0 / 1e12;
}

static double vector_dtype_peak_tflops(const char* key){
  const double base = vector_fp32_peak_tflops();
  // Vector engine scaling (NOT MFMA): defaults FP16=2x, FP8=4x; override via env
  const double s16 = getenv_double("VEC_FP16_SCALE", 2.0);
  const double s8  = getenv_double("VEC_FP8_SCALE",  4.0);
  if(std::strcmp(key, "float32") == 0) return base;
  if(std::strcmp(key, "float16") == 0) return base * s16;
  if(std::strcmp(key, "float8" ) == 0) return base * s8;
  return base;
}

// ----------------- kernel -----------------
__global__ void vec_fma_kernel(float* __restrict__ out, int n, int fmas_per_elt){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= n) return;

  float a = 1.000001f * (float)((idx & 1023) + 1);
  float b = 0.999991f * (float)(((idx >> 2) & 1023) + 2);
  float c = 1.000013f * (float)(((idx >> 4) &  511) + 3);
  float d = 0.999989f * (float)(((idx >> 6) &  255) + 4);

  const int per_iter = 4;
  int iters = fmas_per_elt / per_iter;

  #pragma unroll 64
  for(int t=0; t<iters; ++t){
    a = fmaf(a, b, c);
    b = fmaf(b, c, d);
    c = fmaf(c, d, a);
    d = fmaf(d, a, b);
  }
  out[idx] = a + b + c + d; // single global store to avoid being memory-bound
}

static int round_fmas(int x){
  if(x < 4) return 4;
  return (x + 3) / 4 * 4;
}

static int try_alloc(float** dOut, int& N){
  while(N > 0){
    size_t bytes = (size_t)N * sizeof(float);
    hipError_t e = hipMalloc((void**)dOut, bytes);
    if(e == hipSuccess) return 0;
    N >>= 1;
  }
  return -1;
}

// Plan work so each "label" roughly matches a target FLOP budget
static void plan_work(double target_flops, int N_alloc, int fmas_base,
                      int& N_used, int& fmas_per_elt, int& launches)
{
  if(target_flops <= 0.0) target_flops = 0.25e9; // label 0 -> small non-zero
  N_used = N_alloc;
  fmas_per_elt = round_fmas(fmas_base);

  auto flops_per_launch = [&](int Np, int Fp)->double{
    return 2.0 * (double)Np * (double)Fp; // FMA -> 2 flops
  };

  double fpl = flops_per_launch(N_used, fmas_per_elt);

  // If single launch overshoots, reduce FMAs/elt then N_used
  if(fpl > target_flops){
    int need_fmas = (int)std::floor(target_flops / (2.0 * (double)N_used));
    need_fmas = round_fmas(need_fmas);
    if(need_fmas >= 4){
      fmas_per_elt = need_fmas;
    }else{
      fmas_per_elt = 4;
      double n_need = std::floor(target_flops / (2.0 * (double)fmas_per_elt));
      if(n_need < 1.0) n_need = 1.0;
      if(n_need > (double)N_alloc) n_need = (double)N_alloc;
      N_used = (int)n_need;
    }
    fpl = flops_per_launch(N_used, fmas_per_elt);
  }

  launches = (int)std::ceil(target_flops / std::max(1.0, fpl));
  launches = std::max(1, std::min(launches, 10000));
}

static void run_case(float* dOut, int N_used, int fmas_per_elt, int launches, int block,
                     double& gflops_out)
{
  int grid = (N_used + block - 1) / block;

  hipEvent_t e0, e1; ckHip(hipEventCreate(&e0), "hipEventCreate e0");
  ckHip(hipEventCreate(&e1), "hipEventCreate e1");

  // warmup
  hipLaunchKernelGGL(vec_fma_kernel, dim3(grid), dim3(block), 0, 0,
                     dOut, N_used, fmas_per_elt);
  ckHip(hipDeviceSynchronize(), "hipDeviceSynchronize warmup");

  ckHip(hipEventRecord(e0, 0), "hipEventRecord e0");
  for(int r=0; r<launches; ++r){
    hipLaunchKernelGGL(vec_fma_kernel, dim3(grid), dim3(block), 0, 0,
                       dOut, N_used, fmas_per_elt);
  }
  ckHip(hipEventRecord(e1, 0), "hipEventRecord e1");
  ckHip(hipEventSynchronize(e1), "hipEventSynchronize e1");
  float ms = 0.0f; ckHip(hipEventElapsedTime(&ms, e0, e1), "hipEventElapsedTime");
  hipEventDestroy(e0); hipEventDestroy(e1);

  const double total_flops = 2.0 * (double)N_used * (double)fmas_per_elt * (double)launches;
  const double gflops = total_flops / (ms * 1e6);
  gflops_out = gflops;
}

// Convenience: compute efficiency from measured GFLOPs and peak TFLOPs
static inline double eff_from(double gflops, double peak_tflops){
  return (peak_tflops > 0.0) ? (gflops / (peak_tflops * 1000.0)) : 0.0;
}

static void print_table(const char* key, double peak_tflops,
                        double e16, double e4, double e1, double e0)
{
  std::printf("Vector table (%s):\n", key);
  std::printf("  tflops=%.2f\n", peak_tflops);
  std::printf("  gflops_efficiency=[[16,%.4f],[4,%.4f],[1,%.4f],[0,%.4f]]\n",
              e16, e4, e1, e0);
}

int main(){
  // Tunables
  int    N_alloc   = getenv_int("N_ELEMS",       1<<24); // ~64 MiB output
  int    fmas_base = getenv_int("FMAS_PER_ELT",  512);
  int    block     = getenv_int("BLOCK",         256);
  block = std::max(32, std::min(1024, block));

  // Allocate output (halve on OOM)
  float* dOut = nullptr;
  if(try_alloc(&dOut, N_alloc) != 0 || N_alloc <= 0){
    const double p32 = vector_dtype_peak_tflops("float32");
    const double p16 = vector_dtype_peak_tflops("float16");
    const double p8  = vector_dtype_peak_tflops("float8");
    const bool report_fp8 = getenv_int("VEC_REPORT_FP8", arch_has_fp8()?1:0) != 0;

    print_table("float32", p32, 0.0, 0.0, 0.0, 0.0);
    print_table("float16", p16, 0.0, 0.0, 0.0, 0.0);
    if(report_fp8) print_table("float8",  p8,  0.0, 0.0, 0.0, 0.0);
    return 0;
  }

  // Target FLOP budgets for labels
  struct Target { int label; double flops; };
  std::vector<Target> tgts = {
    {16, 16.0e9}, {4, 4.0e9}, {1, 1.0e9}, {0, 0.25e9}
  };

  // Plan and measure per workload
  double g16=0.0, g4=0.0, g1=0.0, g0=0.0;
  for(const auto& t : tgts){
    int N_used=0, fmas_per_elt=0, launches=0;
    plan_work(t.flops, N_alloc, fmas_base, N_used, fmas_per_elt, launches);
    double g=0.0;
    run_case(dOut, N_used, fmas_per_elt, launches, block, g);
    if(t.label==16) g16=g; else if(t.label==4) g4=g; else if(t.label==1) g1=g; else g0=g;
  }

  // Peaks per dtype (vector engine)
  const double p32 = vector_dtype_peak_tflops("float32");
  const double p16 = vector_dtype_peak_tflops("float16");
  const double p8  = vector_dtype_peak_tflops("float8");
  const bool report_fp8 = getenv_int("VEC_REPORT_FP8", arch_has_fp8()?1:0) != 0;

  // Efficiencies per dtype
  double e32_16 = eff_from(g16, p32), e32_4 = eff_from(g4, p32), e32_1 = eff_from(g1, p32), e32_0 = eff_from(g0, p32);
  double e16_16 = eff_from(g16, p16), e16_4 = eff_from(g4, p16), e16_1 = eff_from(g1, p16), e16_0 = eff_from(g0, p16);
  double e8_16  = eff_from(g16, p8 ), e8_4  = eff_from(g4, p8 ), e8_1  = eff_from(g1, p8 ), e8_0  = eff_from(g0, p8 );

  // Emit tables
  print_table("float32", p32, e32_16, e32_4, e32_1, e32_0);
  print_table("float16", p16, e16_16, e16_4, e16_1, e16_0);
  if(report_fp8) print_table("float8",  p8,  e8_16,  e8_4,  e8_1,  e8_0);

  hipFree(dOut);
  return 0;
}
