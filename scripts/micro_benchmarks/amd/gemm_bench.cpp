// gemm_bench.cpp  (AMD/ROCm + rocBLAS)
//
// Micro-batch + shape-aware GEMM efficiency benchmark, binned by total GFLOPs.
// Supports float32, float16, float8 (if available). Measures single and
// strided-batched GEMMs and emits a JSON summary used to derive efficiency
// factors in performance models.
//
// What it does
// ------------
// * Sweeps an outer micro-batch dimension B and a coarse grid of (M, N, K).
// * Dtype dispatch:
//     - float32  -> rocblas_sgemm / rocblas_sgemm_strided_batched
//     - float16  -> rocblas_gemm_ex / rocblas_gemm_strided_batched_ex (compute=f32)
//     - float8   -> rocblas_gemm_ex / rocblas_gemm_strided_batched_ex (compute=f32)
//                   (probed at runtime; skipped if unsupported on this platform)
// * Skips shapes that exceed a memory cap (based on B * (A+B+C) bytes).
// * Autotunes repetitions per shape to hit a target runtime.
// * Computes efficiency per shape as:
//       eff = (achieved_GFLOP/s) / (device_peak_GFLOP/s for the dtype)
//   where achieved_GFLOP/s is measured, and device peak is a configurable estimate.
// * Bins shapes by total GFLOPs, with thresholds on 2*B*M*N*K / 1e9 ∈
//   { [512,∞), [64,512), [8,64), [1,8), [0,1) }.
// * Prints one progress bar **per dtype**.
//
// Output
// ------
// Writes ./gemm_bench.json with this exact schema (one top-level object per dtype
// actually run on the platform):
//
// {
//   "float32": {
//     "tflops": <peak_tflops>,
//     "gflops_efficiency": {
//       "<B>": {                       // micro-batch size as string key
//         "512": { "M,N,K": eff, ...}, // bin >= 512 GFLOPs
//         "64":  { "M,N,K": eff, ...}, // 64..512
//         "8":   { "M,N,K": eff, ...}, // 8..64
//         "1":   { "M,N,K": eff, ...}, // 1..8
//         "0":   { "M,N,K": eff, ...}  // <1
//       },
//       ...
//     }
//   },
//   "float16": { ... },
//   "float8":  { ... }                 // omitted entirely if FP8 unsupported
// }
//
// Notes
// -----
// * Shape keys are strings "M,N,K" to keep the JSON valid and compact.
// * Efficiencies are sorted descending per (B, bin) internally.
// * On ROCm 6.3.x, FP8 is exposed as a single type (f8_r). If the runtime/arch
//   doesn’t implement FP8 GEMM, it is skipped (and "float8" is omitted).
//
// Normalization policy (MFMA vs vector)
// -------------------------------------
// * **This GEMM benchmark normalizes to matrix-core (MFMA) peak** for FP32/FP16/FP8,
//   because rocBLAS GEMMs run on MFMA units on CDNA/RDNA where available.
// * The internal baseline `fp32_tflops_peak()` returns a **vector FP32 peak**
//   (per-CU FP32 lanes × clock × 2). We then scale that vector baseline up to
//   per-dtype **MFMA peak** inside `dtype_peak_tflops()`.
// * This mirrors Megatron-LM’s behavior: GEMM eff is vs MFMA peak; separate
//   element-wise microbenches (vector ALU) should use vector FP peaks.
//
// Environment knobs (all optional)
// --------------------------------
// DTYPE=float32|fp32|f32|float16|fp16|f16|half|float8|fp8|f8
//   If set, runs only the specified dtype; otherwise runs float32, float16, float8.
//
// FAST=0|1
//   If FAST=1 and MB_DIMS is not set, test only B in {1,2,4,8,16,32}. Also sets
//   DIM_STRIDE default to 2. (See DIM_STRIDE.)
//
// DIM_STRIDE=<int>
//   Subsample stride for M/N/K lists (default: FAST?2:1). The last value is kept.
//
// MB_DIMS="b1,b2,..."
//   Explicit micro-batch sizes to test (e.g., "1,2,4,8,16,32,64,128").
//   If unset: FAST? {1,2,4,8,16,32} : {1,2,4,8,16,32,64,128}.
//
// MB_DIM_STRIDE=<int>
//   Optional subsample stride for MB_DIMS (default: DIM_STRIDE). If FAST=1 and
//   MB_DIMS is unset, the exact set {1,2,4,8,16,32} is used (no thinning).
//
// COMMON_DIMS="a,b,c,..."
//   Sets the same candidate set for M, N, and K (overridden by M_DIMS/N_DIMS/K_DIMS).
//
// M_DIMS / N_DIMS / K_DIMS="a,b,c,..."
//   Per-axis candidate lists (win over COMMON_DIMS).
//
// MEM_CAP_GB=<double>
//   Max allowed total bytes per tested shape: B*(A+B+C). Default is ~80% of
//   current free device memory. Increase to admit larger shapes.
//
// REPS_AUTO_TARGET_MS=<double>
//   Auto-tune repetitions so that each (B,M,N,K) accumulates this many ms
//   total in the timed loop (default: 35 ms).
//
// VERBOSE=0|1
//   If 0, suppresses informational messages (e.g., FP8 skip notice).
//
// Peak model & overrides
// ----------------------
// * Vector FP32 baseline (used as starting point):
//     fp32_vector_peak ≈ (#CUs) * (lanes_per_CU) * (clock_Hz) * 2 / 1e12
//   with lanes_per_CU defaults aligned to vector ALUs:
//     - CDNA (gfx90*, gfx94*, gfx95*):  64 lanes/CU
//     - RDNA (gfx10*, gfx11*):          64 lanes/CU
//   Override lanes with:  FP32_CORES_PER_MP=<int>  (e.g., 64 or 128)
// * GEMM (MFMA) dtype peaks are derived by scaling the vector baseline:
//     FP32_GEMM_peak = fp32_vector_peak * FP32_MFMA_SCALE
//     FP16_GEMM_peak = fp32_vector_peak * FP16_PEAK_SCALE
//     FP8_GEMM_peak  = fp32_vector_peak * FP8_PEAK_SCALE
//   Defaults (chosen to match CDNA behavior / Megatron-LM expectations):
//     - On CDNA:  FP32_MFMA_SCALE=2.0, FP16_PEAK_SCALE=8.0, FP8_PEAK_SCALE=16.0
//     - On RDNA/other: FP32_MFMA_SCALE=1.0, FP16_PEAK_SCALE=2.0, FP8_PEAK_SCALE=4.0
//   Runtime overrides:
//     FP32_MFMA_SCALE=<double>
//     FP16_PEAK_SCALE=<double>
//     FP8_PEAK_SCALE=<double>
//
// Progress & Logging
// ------------------
// * Prints a single-line progress bar to stderr per dtype with the current (B,M,N,K).
// * On completion, prints “Wrote JSON: ./gemm_bench.json” to stderr.
//
// Build
// -----
//   hipcc -O3 -std=c++17 gemm_bench.cpp -lrocblas -o gemm_bench
//
// Caveats
// -------
// * Peaks are coarse but chosen so that efficiencies are ≤ 1.0 in practice
//   for MFMA-backed GEMMs. If you customize the scales, eff can exceed 1.0.
// * This benchmark isolates GEMM kernels. Frameworks may use fused epilogues
//   (rocBLASLt/Tensile) or grouped scheduling with different utilization.
// * Memory cap uses raw tensor storage (A,B,C). Real models may reuse/fuse
//   buffers and have different residency/transient footprints.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

// ----- small utils -----
static inline int getenv_int(const char* k, int defv){
  if(const char* s = std::getenv(k)) return std::atoi(s);
  return defv;
}
static inline double getenv_double(const char* k, double defv){
  if(const char* s = std::getenv(k)) return std::atof(s);
  return defv;
}
static std::string getenv_str(const char* k, const char* defv=nullptr){
  const char* s = std::getenv(k);
  return s ? std::string(s) : (defv ? std::string(defv) : std::string());
}
static std::vector<int> parse_int_list(const char* envvar){
  std::vector<int> out;
  const char* s = std::getenv(envvar);
  if(!s) return out;
  int v = 0; bool in_num=false; bool neg=false;
  for(const char* p=s; ; ++p){
    char c = *p;
    if(c == '-') { neg = true; in_num = true; continue; }
    if(c >= '0' && c <= '9'){
      v = v*10 + (c - '0');
      in_num = true;
    }else{
      if(in_num){
        out.push_back(neg ? -v : v);
        v = 0; in_num=false; neg=false;
      }
      if(c == '\0') break;
    }
  }
  out.erase(std::remove_if(out.begin(), out.end(), [](int x){return x<=0;}), out.end());
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

static inline void ckHip(hipError_t e, const char* where){
  if(e != hipSuccess){
    std::fprintf(stderr, "HIP error %d (%s) at %s\n",
                 (int)e, hipGetErrorString(e), where);
    std::exit(1);
  }
}
static inline void ckRoc(rocblas_status e, const char* where){
  if(e != rocblas_status_success){
    std::fprintf(stderr, "rocBLAS error %d at %s\n", (int)e, where);
    std::exit(1);
  }
}

static inline double now_ms(){
  using clk = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

// stderr progress + current test config
static void progress_with_cfg(const char* dtype, double frac, int B, int M,int N,int K, size_t i, size_t total){
  frac = std::max(0.0, std::min(1.0, frac));
  const int barw = 40;
  int filled = (int)std::round(frac * barw);
  std::fprintf(stderr, "\r[%s] [", dtype);
  for(int j=0;j<barw;j++) std::fputc(j<filled ? '#' : '-', stderr);
  std::fprintf(stderr, "] %5.1f%%  | Testing B=%d, M=%d, N=%d, K=%d  (%zu/%zu)",
               frac*100.0, B, M, N, K, i, total);
  if(frac>=1.0) std::fprintf(stderr, "  \n");
  std::fflush(stderr);
}

// ---- Peak TFLOPS (coarse) ----
static int fp32_lanes_per_cu(const hipDeviceProp_t& p){
  if(const char* s = std::getenv("FP32_CORES_PER_MP")){
    int v = std::atoi(s);
    if(v > 0) return v;
  }
  const char* g = p.gcnArchName;
  if(g){
    // CDNA generations (e.g., gfx908/gfx90a, gfx94*, gfx95*): use 64 vector FP32 lanes/CU
    if(std::strncmp(g, "gfx90", 5) == 0 ||
       std::strncmp(g, "gfx94", 5) == 0 ||
       std::strncmp(g, "gfx95", 5) == 0)
      return 64;
    // RDNA generations: use 64 lanes/CU for FP32 vector peak
    if(std::strncmp(g, "gfx10", 5) == 0 ||
       std::strncmp(g, "gfx11", 5) == 0)
      return 64;
  }
  return 64;  // fallback
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
  return (clock_khz > 0) ? (double)clock_khz * 1000.0 : 1.0e9;
}
static double fp32_tflops_peak(){
  int dev = 0; ckHip(hipGetDevice(&dev), "hipGetDevice");
  hipDeviceProp_t p{}; ckHip(hipGetDeviceProperties(&p, dev), "hipGetDeviceProperties");
  const double cus   = (double)p.multiProcessorCount;
  const double lanes = (double)fp32_lanes_per_cu(p);
  const double hz    = device_clock_hz();
  return cus * lanes * hz * 2.0 / 1e12;   // FMA => 2 flops/cycle
}

static bool is_cdna_arch(const hipDeviceProp_t& p){
  // p.gcnArchName is a C string (char[])
  const char* g = p.gcnArchName;
  return std::strncmp(g, "gfx90", 5) == 0 || std::strncmp(g, "gfx94", 5) == 0 || std::strncmp(g, "gfx95", 5) == 0;
}
static bool is_rdna_arch(const hipDeviceProp_t& p){
  const char* g = p.gcnArchName;
  return std::strncmp(g, "gfx10", 5) == 0 || std::strncmp(g, "gfx11", 5) == 0;
}
static double dtype_peak_tflops(const char* dtype_key){
  // Start from the **vector FP32** baseline (used by eltwise microbench).
  double base_vec_fp32 = fp32_tflops_peak();

  int dev = 0; ckHip(hipGetDevice(&dev), "hipGetDevice");
  hipDeviceProp_t p{}; ckHip(hipGetDeviceProperties(&p, dev), "hipGetDeviceProperties");

  const bool cdna = is_cdna_arch(p);
  const bool rdna = is_rdna_arch(p);

  // MFMA scaling (defaults chosen to match MI210/Megatron-LM behavior):
  const double s32m_default = cdna ? 2.0  : 1.0;  // FP32 matrix-core ≈ 2× vector on CDNA (e.g., 22.6 -> ~45 TFLOPS)
  const double s16_default  = cdna ? 8.0  : (rdna ? 2.0 : 2.0);  // FP16 MFMA vs vector FP32 baseline
  const double s8_default   = cdna ? 16.0 : (rdna ? 4.0 : 4.0);

  const double s32m = getenv_double("FP32_MFMA_SCALE", s32m_default);
  const double s16  = getenv_double("FP16_PEAK_SCALE",  s16_default);
  const double s8   = getenv_double("FP8_PEAK_SCALE",   s8_default);

  if(!std::strcmp(dtype_key, "float32")) return base_vec_fp32 * s32m;  // FP32 GEMM uses MFMA peak
  if(!std::strcmp(dtype_key, "float16")) return base_vec_fp32 * s16;   // FP16 GEMM uses MFMA peak
  if(!std::strcmp(dtype_key, "float8"))  return base_vec_fp32 * s8;    // FP8 GEMM (if available)
  return base_vec_fp32 * s32m;
}

// bytes helpers (per element size)
static inline size_t bytes_A(int M,int K,int elem){ return (size_t)M * (size_t)K * (size_t)elem; }
static inline size_t bytes_B(int K,int N,int elem){ return (size_t)K * (size_t)N * (size_t)elem; }
static inline size_t bytes_C(int M,int N,int elem){ return (size_t)M * (size_t)N * (size_t)elem; }
static inline size_t total_bytes_batched(int B,int M,int N,int K,int elem){
  return (size_t)B * (bytes_A(M,K,elem) + bytes_B(K,N,elem) + bytes_C(M,N,elem));
}

// ---- Timed GEMM runners ----

// FP32 (legacy path)
static double run_sgemm_ms(rocblas_handle h, int M,int N,int K,
                           float* dA,float* dB,float* dC, int reps_target){
  const float alpha = 1.0f, beta = 0.0f;
  ckRoc(rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none,
                      M, N, K, &alpha, dA, M, dB, K, &beta, dC, M),
        "rocblas_sgemm warmup");
  ckHip(hipDeviceSynchronize(), "sync warmup");
  const int reps = std::max(1, reps_target);
  double t0 = now_ms();
  for(int i=0;i<reps;i++){
    ckRoc(rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none,
                        M, N, K, &alpha, dA, M, dB, K, &beta, dC, M),
          "rocblas_sgemm timed");
  }
  ckHip(hipDeviceSynchronize(), "sync timed");
  double t1 = now_ms();
  return (t1 - t0) / (double)reps;
}
static double run_sgemm_strided_batched_ms(rocblas_handle h, int B, int M,int N,int K,
                                           float* dA,float* dB,float* dC,
                                           long long strideA, long long strideB, long long strideC,
                                           int reps_target){
  const float alpha = 1.0f, beta = 0.0f;
  ckRoc(rocblas_sgemm_strided_batched(h,
        rocblas_operation_none, rocblas_operation_none,
        M, N, K, &alpha,
        dA, M, (rocblas_stride)strideA,
        dB, K, (rocblas_stride)strideB,
        &beta,
        dC, M, (rocblas_stride)strideC,
        B),
        "rocblas_sgemm_strided_batched warmup");
  ckHip(hipDeviceSynchronize(), "sync warmup batched");
  const int reps = std::max(1, reps_target);
  double t0 = now_ms();
  for(int i=0;i<reps;i++){
    ckRoc(rocblas_sgemm_strided_batched(h,
          rocblas_operation_none, rocblas_operation_none,
          M, N, K, &alpha,
          dA, M, (rocblas_stride)strideA,
          dB, K, (rocblas_stride)strideB,
          &beta,
          dC, M, (rocblas_stride)strideC,
          B),
          "rocblas_sgemm_strided_batched timed");
  }
  ckHip(hipDeviceSynchronize(), "sync timed batched");
  double t1 = now_ms();
  return (t1 - t0) / (double)reps;
}

// GEMM_EX (used for FP16 and FP8) — non-batched (ROCm 6.3.x signature)
static double run_gemm_ex_ms(rocblas_handle h, int M,int N,int K,
                             const void* dA,const void* dB,void* dC,
                             rocblas_datatype a_type, rocblas_datatype b_type,
                             rocblas_datatype c_type, rocblas_datatype d_type,
                             rocblas_datatype compute_type,
                             int reps_target){
  const float alpha = 1.0f, beta = 0.0f;
  const rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
  const int32_t solution_index = 0;
  const uint32_t flags = rocblas_gemm_flags_none;

  ckRoc(rocblas_gemm_ex(h,
        rocblas_operation_none, rocblas_operation_none,
        M, N, K,
        &alpha,
        dA, a_type, M,
        dB, b_type, K,
        &beta,
        dC, c_type, M,
        dC, d_type, M,
        compute_type, algo, solution_index, flags),
        "rocblas_gemm_ex warmup");
  ckHip(hipDeviceSynchronize(), "sync warmup ex");
  const int reps = std::max(1, reps_target);
  double t0 = now_ms();
  for(int i=0;i<reps;i++){
    ckRoc(rocblas_gemm_ex(h,
          rocblas_operation_none, rocblas_operation_none,
          M, N, K,
          &alpha,
          dA, a_type, M,
          dB, b_type, K,
          &beta,
          dC, c_type, M,
          dC, d_type, M,
          compute_type, algo, solution_index, flags),
          "rocblas_gemm_ex timed");
  }
  ckHip(hipDeviceSynchronize(), "sync ex timed");
  double t1 = now_ms();
  return (t1 - t0) / (double)reps;
}

// GEMM_EX (strided-batched) — ROCm 6.3.x signature
static double run_gemm_ex_strided_batched_ms(rocblas_handle h, int B, int M,int N,int K,
                                             const void* dA,const void* dB,void* dC,
                                             long long strideA, long long strideB, long long strideC,
                                             rocblas_datatype a_type, rocblas_datatype b_type,
                                             rocblas_datatype c_type, rocblas_datatype d_type,
                                             rocblas_datatype compute_type,
                                             int reps_target){
  const float alpha = 1.0f, beta = 0.0f;
  const rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
  const int32_t solution_index = 0;
  const uint32_t flags = rocblas_gemm_flags_none;

  ckRoc(rocblas_gemm_strided_batched_ex(h,
        rocblas_operation_none, rocblas_operation_none,
        M, N, K,
        &alpha,
        dA, a_type, M, (rocblas_stride)strideA,
        dB, b_type, K, (rocblas_stride)strideB,
        &beta,
        dC, c_type, M, (rocblas_stride)strideC,
        dC, d_type, M, (rocblas_stride)strideC,
        B, compute_type, algo, solution_index, flags),
        "rocblas_gemm_strided_batched_ex warmup");
  ckHip(hipDeviceSynchronize(), "sync warmup ex batched");
  const int reps = std::max(1, reps_target);
  double t0 = now_ms();
  for(int i=0;i<reps;i++){
    ckRoc(rocblas_gemm_strided_batched_ex(h,
          rocblas_operation_none, rocblas_operation_none,
          M, N, K,
          &alpha,
          dA, a_type, M, (rocblas_stride)strideA,
          dB, b_type, K, (rocblas_stride)strideB,
          &beta,
          dC, c_type, M, (rocblas_stride)strideC,
          dC, d_type, M, (rocblas_stride)strideC,
          B, compute_type, algo, solution_index, flags),
          "rocblas_gemm_strided_batched_ex timed");
  }
  ckHip(hipDeviceSynchronize(), "sync ex timed batched");
  double t1 = now_ms();
  return (t1 - t0) / (double)reps;
}

// GFLOPs binning (by 2*B*M*N*K / 1e9), thresholds: 512,64,8,1,0
static int bin_from_gflops(double gflops){
  if(gflops >= 512.0) return 512;
  if(gflops >=  64.0) return  64;
  if(gflops >=   8.0) return   8;
  if(gflops >=   1.0) return   1;
  return 0;
}

// Record per (M,N,K,eff)
struct Rec { int M,N,K; double eff; };

// Keep order [512,64,8,1,0]
static const int BIN_ORDER[5] = {512,64,8,1,0};

// ---- dtype dispatch ----
struct DTypeCfg {
  const char* key;                // "float32" | "float16" | "float8"
  int elem_bytes;                 // 4 | 2 | 1
  bool use_ex;                    // FP16/FP8 use gemm_ex; FP32 uses direct path
  rocblas_datatype a_type, b_type, c_type, d_type;
  rocblas_datatype compute_type;  // ROCm 6.3: compute_type is a rocblas_datatype (e.g., f32_r)
};

static bool string_ieq(const std::string& a, const char* b){
  if(!b) return false;
  if(a.size() != std::strlen(b)) return false;
  for(size_t i=0;i<a.size();++i){
    char ca = a[i], cb = b[i];
    if('A'<=ca && ca<='Z') ca += 'a'-'A';
    if('A'<=cb && cb<='Z') cb += 'a'-'A';
    if(ca != cb) return false;
  }
  return true;
}

// Parse DTYPE env; returns empty => run all 3.
static std::vector<std::string> select_dtypes(){
  std::string s = getenv_str("DTYPE");
  if(s.empty()){
    return {"float32", "float16", "float8"};
  }
  if(string_ieq(s,"f32")||string_ieq(s,"fp32")||string_ieq(s,"float32")||string_ieq(s,"float")){
    return {"float32"};
  }
  if(string_ieq(s,"f16")||string_ieq(s,"fp16")||string_ieq(s,"float16")||string_ieq(s,"half")){
    return {"float16"};
  }
  if(string_ieq(s,"f8")||string_ieq(s,"fp8")||string_ieq(s,"float8")){
    return {"float8"};
  }
  // Fallback to all if unknown
  return {"float32", "float16", "float8"};
}

static DTypeCfg make_dtype_cfg(const std::string& key){
  if(key=="float32"){
    return DTypeCfg{"float32", 4, false,
      rocblas_datatype_f32_r, rocblas_datatype_f32_r,
      rocblas_datatype_f32_r, rocblas_datatype_f32_r,
      rocblas_datatype_f32_r};
  }
  if(key=="float16"){
    // A/B/C/D = f16; compute = f32
    return DTypeCfg{"float16", 2, true,
      rocblas_datatype_f16_r, rocblas_datatype_f16_r,
      rocblas_datatype_f16_r, rocblas_datatype_f16_r,
      rocblas_datatype_f32_r};
  }
  if(key=="float8"){
    // ROCm 6.3.x exposes a single FP8 type: f8_r
    return DTypeCfg{"float8", 1, true,
      rocblas_datatype_f8_r, rocblas_datatype_f8_r,
      rocblas_datatype_f8_r, rocblas_datatype_f8_r,
      rocblas_datatype_f32_r};
  }
  return make_dtype_cfg(std::string("float32"));
}

int main(){
  // --------------- configuration ---------------
  const int FAST = getenv_int("FAST", 0);
  const double REPS_TARGET_MS = getenv_double("REPS_AUTO_TARGET_MS", 35.0);
  double mem_cap_gb = getenv_double("MEM_CAP_GB", 0.0); // 0 -> derive from free

  // Dims (override via env)
  std::vector<int> dims_default = {
    128, 192, 256, 384, 512, 768,
    960, 1024, 1152, 1536, 2048,
    3072, 4096, 6144, 8192, 12288, 16384
  };
  std::vector<int> M_dims = parse_int_list("M_DIMS");
  std::vector<int> N_dims = parse_int_list("N_DIMS");
  std::vector<int> K_dims = parse_int_list("K_DIMS");
  std::vector<int> common = parse_int_list("COMMON_DIMS");
  if(common.size()){
    M_dims = common; N_dims = common; K_dims = common;
  }
  if(M_dims.empty()) M_dims = dims_default;
  if(N_dims.empty()) N_dims = dims_default;
  if(K_dims.empty()) K_dims = dims_default;

  // Micro-batch sizes (outer dimension)
  std::vector<int> MB_dims = parse_int_list("MB_DIMS");
  const bool mb_env_set = !MB_dims.empty();
  if(!mb_env_set){
    if(FAST){
      MB_dims = {1,2,4,8,16,32}; // exact set when FAST=1
    }else{
      MB_dims = {1,2,4,8,16,32,64,128};
    }
  }

  // Optional grid thinning
  int dim_stride    = getenv_int("DIM_STRIDE", (FAST?2:1));
  int mb_dim_stride = getenv_int("MB_DIM_STRIDE", dim_stride);

  auto stride_view = [&](const std::vector<int>& src, int stride){
    std::vector<int> out;
    for(size_t i=0;i<src.size(); i += (size_t)std::max(1, stride)) out.push_back(src[i]);
    if(out.empty() || out.back() != src.back()) out.push_back(src.back());
    return out;
  };

  M_dims  = stride_view(M_dims,  dim_stride);
  N_dims  = stride_view(N_dims,  dim_stride);
  K_dims  = stride_view(K_dims,  dim_stride);
  if(!(FAST && !mb_env_set)){
    MB_dims = stride_view(MB_dims, mb_dim_stride);
  }

  // Derive memory cap from free memory if not set
  if(mem_cap_gb <= 0.0){
    size_t freeB=0, totalB=0;
    if(hipMemGetInfo(&freeB, &totalB) == hipSuccess){
      mem_cap_gb = (double)freeB * 0.80 / (1024.0*1024.0*1024.0);
    }else{
      mem_cap_gb = 18.0; // fallback
    }
  }
  const size_t MEM_CAP_BYTES_BASE = (size_t)std::max(0.0, mem_cap_gb) * (size_t)(1024ULL*1024ULL*1024ULL);

  // dtype selection
  std::vector<std::string> dtypes = select_dtypes();

  // Results accumulator: dtype -> (tflops, perB bins)
  struct DTypeOut { double tflops; std::map<int, std::map<int, std::vector<Rec>>> bins; };
  std::map<std::string, DTypeOut> all;

  // Create handle
  rocblas_handle handle{};
  ckRoc(rocblas_create_handle(&handle), "rocblas_create_handle");
  ckRoc(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host), "rocblas_set_pointer_mode");

  // Run each dtype in sequence (one progress bar per dtype)
  for(const auto& dkey : dtypes){
    DTypeCfg cfg = make_dtype_cfg(dkey);
    const double TFLOPS = dtype_peak_tflops(cfg.key);
    const size_t MEM_CAP_BYTES = MEM_CAP_BYTES_BASE;

    // Probe FP8 support (if requested)
    bool dtype_supported = true;
    if(cfg.use_ex && dkey=="float8"){
      // tiny sanity GEMM_EX call
      const int M=1,N=1,K=1;
      void *A=nullptr, *B=nullptr, *C=nullptr;
      ckHip(hipMalloc(&A, cfg.elem_bytes), "probe hipMalloc A");
      ckHip(hipMalloc(&B, cfg.elem_bytes), "probe hipMalloc B");
      ckHip(hipMalloc(&C, cfg.elem_bytes), "probe hipMalloc C");
      float alpha=1.0f,beta=0.0f;
      rocblas_status st = rocblas_gemm_ex(handle,
        rocblas_operation_none, rocblas_operation_none,
        M, N, K,
        &alpha,
        A, cfg.a_type, M,
        B, cfg.b_type, K,
        &beta,
        C, cfg.c_type, M,
        C, cfg.d_type, M,
        cfg.compute_type, rocblas_gemm_algo_standard, 0, rocblas_gemm_flags_none);
      ckHip(hipFree(C), "probe hipFree C");
      ckHip(hipFree(B), "probe hipFree B");
      ckHip(hipFree(A), "probe hipFree A");
      if(st == rocblas_status_not_implemented){
        dtype_supported = false;
        std::fprintf(stderr, "[float8] rocBLAS reports FP8 not implemented on this platform; skipping.\n");
      }else if(st != rocblas_status_success){
        dtype_supported = false;
        std::fprintf(stderr, "[float8] rocBLAS FP8 probe failed with status %d; skipping.\n", (int)st);
      }
    }
    if(!dtype_supported){
      all[dkey] = DTypeOut{TFLOPS, {}};
      continue;
    }

    std::map<int, std::map<int, std::vector<Rec>>> perB_bins;

    // Sweep all candidates (B,M,N,K) that fit memory
    size_t total = (size_t)MB_dims.size() * M_dims.size() * N_dims.size() * K_dims.size();
    size_t idx = 0;

    for(int B : MB_dims){
      for(int M : M_dims){
        for(int N : N_dims){
          for(int K : K_dims){
            idx++;
            if(total_bytes_batched(B,M,N,K,cfg.elem_bytes) > MEM_CAP_BYTES){
              progress_with_cfg(cfg.key, (double)idx/(double)total, B, M,N,K, idx, total);
              continue;
            }

            // Allocate
            const size_t strideA_elems = (size_t)M * (size_t)K;
            const size_t strideB_elems = (size_t)K * (size_t)N;
            const size_t strideC_elems = (size_t)M * (size_t)N;

            if(!cfg.use_ex){
              // FP32 path
              float *dA=nullptr, *dB=nullptr, *dC=nullptr;
              ckHip(hipMalloc(&dA, (size_t)B * strideA_elems * sizeof(float)), "hipMalloc dA");
              ckHip(hipMalloc(&dB, (size_t)B * strideB_elems * sizeof(float)), "hipMalloc dB");
              ckHip(hipMalloc(&dC, (size_t)B * strideC_elems * sizeof(float)), "hipMalloc dC");
              ckHip(hipMemset(dC, 0, (size_t)B * strideC_elems * sizeof(float)), "hipMemset dC");

              const double target_ms = REPS_TARGET_MS;
              double ms = 0.0;

              if(B == 1){
                double probe_ms = run_sgemm_ms(handle, M,N,K, dA,dB,dC, 1);
                int reps = std::max(1, (int)std::round(target_ms / std::max(0.1, probe_ms)));
                ms = run_sgemm_ms(handle, M,N,K, dA,dB,dC, reps);
              }else{
                const long long sA = (long long)strideA_elems;
                const long long sB = (long long)strideB_elems;
                const long long sC = (long long)strideC_elems;
                double probe_ms = run_sgemm_strided_batched_ms(handle, B, M,N,K, dA,dB,dC, sA,sB,sC, 1);
                int reps = std::max(1, (int)std::round(target_ms / std::max(0.1, probe_ms)));
                ms = run_sgemm_strided_batched_ms(handle, B, M,N,K, dA,dB,dC, sA,sB,sC, reps);
              }

              const double gflops_total = (2.0 * (double)B * (double)M * (double)N * (double)K) / 1e9;
              const double achieved_gflops_per_s = (gflops_total * 1e3) / ms;
              const double eff = (TFLOPS > 0.0) ? (achieved_gflops_per_s / (TFLOPS * 1000.0)) : 0.0;

              const int bin = bin_from_gflops(gflops_total);
              perB_bins[B][bin].push_back({M,N,K,eff});

              ckHip(hipFree(dC), "free dC");
              ckHip(hipFree(dB), "free dB");
              ckHip(hipFree(dA), "free dA");
            }else{
              // GEMM_EX path (FP16/FP8)
              void *dA=nullptr, *dB=nullptr, *dC=nullptr;
              ckHip(hipMalloc(&dA, (size_t)B * strideA_elems * (size_t)cfg.elem_bytes), "hipMalloc dA");
              ckHip(hipMalloc(&dB, (size_t)B * strideB_elems * (size_t)cfg.elem_bytes), "hipMalloc dB");
              ckHip(hipMalloc(&dC, (size_t)B * strideC_elems * (size_t)cfg.elem_bytes), "hipMalloc dC");
              ckHip(hipMemset(dC, 0, (size_t)B * strideC_elems * (size_t)cfg.elem_bytes), "hipMemset dC");

              const double target_ms = REPS_TARGET_MS;
              double ms = 0.0;

              if(B == 1){
                double probe_ms = run_gemm_ex_ms(handle, M,N,K, dA,dB,dC,
                                      cfg.a_type,cfg.b_type,cfg.c_type,cfg.d_type,cfg.compute_type, 1);
                int reps = std::max(1, (int)std::round(target_ms / std::max(0.1, probe_ms)));
                ms = run_gemm_ex_ms(handle, M,N,K, dA,dB,dC,
                                    cfg.a_type,cfg.b_type,cfg.c_type,cfg.d_type,cfg.compute_type, reps);
              }else{
                const long long sA = (long long)strideA_elems;
                const long long sB = (long long)strideB_elems;
                const long long sC = (long long)strideC_elems;
                double probe_ms = run_gemm_ex_strided_batched_ms(handle, B, M,N,K, dA,dB,dC, sA,sB,sC,
                                      cfg.a_type,cfg.b_type,cfg.c_type,cfg.d_type,cfg.compute_type, 1);
                int reps = std::max(1, (int)std::round(target_ms / std::max(0.1, probe_ms)));
                ms = run_gemm_ex_strided_batched_ms(handle, B, M,N,K, dA,dB,dC, sA,sB,sC,
                                  cfg.a_type,cfg.b_type,cfg.c_type,cfg.d_type,cfg.compute_type, reps);
              }

              const double gflops_total = (2.0 * (double)B * (double)M * (double)N * (double)K) / 1e9;
              const double achieved_gflops_per_s = (gflops_total * 1e3) / ms;
              const double eff = (TFLOPS > 0.0) ? (achieved_gflops_per_s / (TFLOPS * 1000.0)) : 0.0;

              const int bin = bin_from_gflops(gflops_total);
              perB_bins[B][bin].push_back({M,N,K,eff});

              ckHip(hipFree(dC), "free dC");
              ckHip(hipFree(dB), "free dB");
              ckHip(hipFree(dA), "free dA");
            }

            progress_with_cfg(cfg.key, (double)idx/(double)total, B, M,N,K, idx, total);
          }
        }
      }
    }

    // Sort each (B,bin) bucket by efficiency desc
    for(auto& kvB : perB_bins){
      for(auto& kvBin : kvB.second){
        auto& vec = kvBin.second;
        std::sort(vec.begin(), vec.end(), [](const Rec& a, const Rec& b){
          return a.eff > b.eff;
        });
      }
    }

    all[dkey] = DTypeOut{TFLOPS, std::move(perB_bins)};
  }

  // ---------- JSON dump ----------
  {
    std::ofstream f("./gemm_bench.json", std::ios::out | std::ios::trunc);
    if(!f){
      std::fprintf(stderr, "Failed to open gemm_bench.json for writing\n");
      return 1;
    }
    f.setf(std::ios::fixed);

    f << "{\n";
    auto emit_dtype = [&](const char* key){
      auto it = all.find(key);
      if(it == all.end()) return;
      const auto& out = it->second;
      f << "  \"" << key << "\": {\n";
      f << "    \"tflops\": " << std::setprecision(3) << out.tflops << ",\n";
      f << "    \"gflops_efficiency\": {\n";

      std::vector<int> Bs;
      Bs.reserve(out.bins.size());
      for(const auto& kv : out.bins) Bs.push_back(kv.first);
      std::sort(Bs.begin(), Bs.end());

      for(size_t bi=0; bi<Bs.size(); ++bi){
        int B = Bs[bi];
        const auto& bins = out.bins.at(B);

        f << "      \"" << B << "\": {\n";
        for(size_t oi=0; oi<5; ++oi){
          int b = BIN_ORDER[oi];
          f << "        \"" << b << "\": {";
          auto itb = bins.find(b);
          bool first = true;
          if(itb != bins.end()){
            for(const auto& r : itb->second){
              if(!first) f << ", ";
              first = false;
              f << "\"" << r.M << "," << r.N << "," << r.K << "\": " << std::setprecision(6) << r.eff;
            }
          }
          f << "}";
          if(oi+1 < 5) f << ",";
          f << "\n";
        }
        f << "      }";
        if(bi + 1 < Bs.size()) f << ",";
        f << "\n";
      }
      f << "    }\n";
      f << "  }";
    };

    // Emit in order: float32, float16, float8 (if present)
    std::vector<const char*> order = {"float32","float16","float8"};
    bool first_dtype = true;
    for(const char* key : order){
      if(all.find(key) == all.end()) continue;
      if(!first_dtype) f << ",\n";
      first_dtype = false;
      emit_dtype(key);
    }
    if(!first_dtype) f << "\n";
    f << "}\n";
    f.close();
  }

  std::fprintf(stderr, "Wrote JSON: ./gemm_bench.json\n");

  ckRoc(rocblas_destroy_handle(handle), "rocblas_destroy_handle");
  return 0;
}
