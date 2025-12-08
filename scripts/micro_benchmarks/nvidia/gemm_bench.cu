// gemm_bench.cu  (CUDA + cuBLAS)
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
//     - float32  -> cublasSgemm / cublasSgemmStridedBatched
//     - float16  -> cublasGemmEx / cublasGemmStridedBatchedEx (compute=f32)
//     - float8   -> cublasGemmEx / cublasGemmStridedBatchedEx (compute=f32)
//                   (probed at runtime; shapes that fail are skipped; if the
//                    arch truly does not support FP8, the dtype is skipped)
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
// * On recent NVIDIA GPUs, FP16/FP8 GEMMs run on tensor cores when available.
//   If the runtime/arch doesn’t implement FP8 GEMM, it is skipped
//   (and "float8" is omitted).
//
// Normalization policy (tensor cores vs vector ALUs)
// --------------------------------------------------
// * This GEMM benchmark normalizes to tensor-core (matrix-core) peak for
//   FP32/FP16/FP8, because cuBLAS GEMMs run on tensor cores where available.
// * The internal baseline `fp32_tflops_vector_peak()` returns a vector FP32
//   peak (per-SM FP32 lanes × clock × 2). We then scale that vector baseline
//   up to per-dtype tensor-core GEMM peak inside `dtype_peak_tflops()`.
// * This mirrors the ROCm/MI2xx behavior: GEMM eff is vs matrix-core peak;
//   separate element-wise microbenches (vector ALU) should use vector FP peaks.
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
//   If 0, suppresses informational messages (e.g., FP8 skip notice). This
//   implementation always prints FP8 skip messages, matching the ROCm
//   reference behavior.
//
// Peak model & overrides
// ----------------------
// * Vector FP32 baseline (used as starting point):
//     fp32_vector_peak ≈ (#SMs) * (lanes_per_SM) * (clock_Hz) * 2 / 1e12
//   with lanes_per_SM defaults aligned to vector ALUs per SM (configurable):
//     - Default: 128 FP32 lanes/SM for most recent architectures
//   Override lanes with:  FP32_CORES_PER_MP=<int>  (e.g., 64 or 128)
// * GEMM (tensor-core) dtype peaks are derived by scaling the vector baseline:
//     FP32_GEMM_peak = fp32_vector_peak * FP32_MFMA_SCALE
//     FP16_GEMM_peak = fp32_vector_peak * FP16_PEAK_SCALE
//     FP8_GEMM_peak  = fp32_vector_peak * FP8_PEAK_SCALE
//   Defaults (chosen to roughly mirror ROCm behavior):
//     - On tensor-core-capable GPUs (SM >= 80): FP32_MFMA_SCALE=2.0,
//       FP16_PEAK_SCALE=8.0, FP8_PEAK_SCALE=16.0
//     - On older GPUs: FP32_MFMA_SCALE=1.0, FP16_PEAK_SCALE=2.0,
//       FP8_PEAK_SCALE=4.0
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
//   nvcc -O3 -std=c++14 -lcublas gemm_bench.cu -o gemm_bench
//
// Caveats
// -------
// * Peaks are coarse but chosen so that efficiencies are ≤ 1.0 in practice
//   for tensor-core-backed GEMMs. If you customize the scales, eff can exceed 1.0.
// * This benchmark isolates GEMM kernels. Frameworks may use fused epilogues
//   (cuBLASLt/CUTLASS) or grouped scheduling with different utilization.
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

#include <cuda_runtime.h>
#include <cublas_v2.h>

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

static inline void ckCuda(cudaError_t e, const char* where){
  if(e != cudaSuccess){
    std::fprintf(stderr, "CUDA error %d (%s) at %s\n",
                 (int)e, cudaGetErrorString(e), where);
    std::exit(1);
  }
}
static inline void ckCublas(cublasStatus_t e, const char* where){
  if(e != CUBLAS_STATUS_SUCCESS){
    std::fprintf(stderr, "cuBLAS error %d at %s\n", (int)e, where);
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
static int fp32_cores_per_sm(const cudaDeviceProp& p){
  if(const char* s = std::getenv("FP32_CORES_PER_MP")){
    int v = std::atoi(s);
    if(v > 0) return v;
  }
  const int cc = p.major*10 + p.minor;
  if(cc >= 90) return 128; // Hopper/Ada fallback
  if(cc >= 80) return 128; // Ampere
  if(cc >= 75) return  64; // Turing
  if(cc >= 70) return  64; // Volta
  if(cc >= 61) return 128; // Pascal GP10x
  if(cc >= 60) return 128; // Pascal
  if(cc >= 50) return 128; // Maxwell
  return 64;               // very old fallback
}

static double device_clock_hz(){
  int dev = 0; ckCuda(cudaGetDevice(&dev), "cudaGetDevice");
  cudaDeviceProp p{};
  ckCuda(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");
  if(p.clockRate > 0) return (double)p.clockRate * 1000.0;
  return 1.0e9;
}

static double fp32_tflops_vector_peak(){
  int dev = 0; ckCuda(cudaGetDevice(&dev), "cudaGetDevice");
  cudaDeviceProp p{};
  ckCuda(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");
  const double sms   = (double)p.multiProcessorCount;
  const double lanes = (double)fp32_cores_per_sm(p);
  const double hz    = device_clock_hz();
  return sms * lanes * hz * 2.0 / 1e12;   // FMA => 2 flops/cycle
}

static bool is_tensor_core_arch(const cudaDeviceProp& p){
  int cc = p.major*10 + p.minor;
  return cc >= 80;
}

static double dtype_peak_tflops(const char* dtype_key){
  double base_vec_fp32 = fp32_tflops_vector_peak();

  int dev = 0; ckCuda(cudaGetDevice(&dev), "cudaGetDevice");
  cudaDeviceProp p{};
  ckCuda(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");

  const bool tensor = is_tensor_core_arch(p);

  const double s32m_default = tensor ? 2.0  : 1.0;
  const double s16_default  = tensor ? 8.0  : 2.0;
  const double s8_default   = tensor ? 16.0 : 4.0;

  const double s32m = getenv_double("FP32_MFMA_SCALE", s32m_default);
  const double s16  = getenv_double("FP16_PEAK_SCALE",  s16_default);
  const double s8   = getenv_double("FP8_PEAK_SCALE",   s8_default);

  if(!std::strcmp(dtype_key, "float32")) return base_vec_fp32 * s32m;
  if(!std::strcmp(dtype_key, "float16")) return base_vec_fp32 * s16;
  if(!std::strcmp(dtype_key, "float8"))  return base_vec_fp32 * s8;
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
static double run_sgemm_ms(cublasHandle_t h, int M,int N,int K,
                           float* dA,float* dB,float* dC, int reps_target){
  const float alpha = 1.0f, beta = 0.0f;
  ckCublas(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                       M, N, K, &alpha, dA, M, dB, K, &beta, dC, M),
           "cublasSgemm warmup");
  ckCuda(cudaDeviceSynchronize(), "sync warmup");
  const int reps = std::max(1, reps_target);
  double t0 = now_ms();
  for(int i=0;i<reps;i++){
    ckCublas(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                         M, N, K, &alpha, dA, M, dB, K, &beta, dC, M),
             "cublasSgemm timed");
  }
  ckCuda(cudaDeviceSynchronize(), "sync timed");
  double t1 = now_ms();
  return (t1 - t0) / (double)reps;
}

// STRIDED-BATCHED SGEMM (averaged over reps)
static double run_sgemm_strided_batched_ms(cublasHandle_t h, int B, int M,int N,int K,
                                           float* dA,float* dB,float* dC,
                                           long long strideA, long long strideB, long long strideC,
                                           int reps_target){
  const float alpha = 1.0f, beta = 0.0f;
  ckCublas(cublasSgemmStridedBatched(h,
           CUBLAS_OP_N, CUBLAS_OP_N,
           M, N, K, &alpha,
           dA, M, strideA,
           dB, K, strideB,
           &beta,
           dC, M, strideC,
           B),
           "cublasSgemmStridedBatched warmup");
  ckCuda(cudaDeviceSynchronize(), "sync warmup batched");
  const int reps = std::max(1, reps_target);
  double t0 = now_ms();
  for(int i=0;i<reps;i++){
    ckCublas(cublasSgemmStridedBatched(h,
             CUBLAS_OP_N, CUBLAS_OP_N,
             M, N, K, &alpha,
             dA, M, strideA,
             dB, K, strideB,
             &beta,
             dC, M, strideC,
             B),
             "cublasSgemmStridedBatched timed");
  }
  ckCuda(cudaDeviceSynchronize(), "sync timed batched");
  double t1 = now_ms();
  return (t1 - t0) / (double)reps;
}

// GEMM_EX (used for FP16 and FP8) — non-batched
// Returns:
//   >= 0 : average time in ms
//   <  0 : cuBLAS reported an error (shape/params not supported)
static double run_gemm_ex_ms(cublasHandle_t h, int M,int N,int K,
                             const void* dA,const void* dB,void* dC,
                             cudaDataType_t a_type, cudaDataType_t b_type,
                             cudaDataType_t c_type,
                             cublasComputeType_t compute_type,
                             int reps_target){
  const float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t st;

  st = cublasGemmEx(h,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    M, N, K,
                    &alpha,
                    dA, a_type, M,
                    dB, b_type, K,
                    &beta,
                    dC, c_type, M,
                    compute_type,
                    CUBLAS_GEMM_DEFAULT);
  if(st != CUBLAS_STATUS_SUCCESS){
    return -1.0;
  }
  ckCuda(cudaDeviceSynchronize(), "sync warmup ex");

  const int reps = std::max(1, reps_target);
  double t0 = now_ms();
  for(int i=0;i<reps;i++){
    st = cublasGemmEx(h,
                      CUBLAS_OP_N, CUBLAS_OP_N,
                      M, N, K,
                      &alpha,
                      dA, a_type, M,
                      dB, b_type, K,
                      &beta,
                      dC, c_type, M,
                      compute_type,
                      CUBLAS_GEMM_DEFAULT);
    if(st != CUBLAS_STATUS_SUCCESS){
      return -1.0;
    }
  }
  ckCuda(cudaDeviceSynchronize(), "sync ex timed");
  double t1 = now_ms();
  return (t1 - t0) / (double)reps;
}

// GEMM_EX (strided-batched)
// Returns:
//   >= 0 : average time in ms
//   <  0 : cuBLAS reported an error (shape/params not supported)
static double run_gemm_ex_strided_batched_ms(cublasHandle_t h, int B, int M,int N,int K,
                                             const void* dA,const void* dB,void* dC,
                                             long long strideA, long long strideB, long long strideC,
                                             cudaDataType_t a_type, cudaDataType_t b_type,
                                             cudaDataType_t c_type,
                                             cublasComputeType_t compute_type,
                                             int reps_target){
  const float alpha = 1.0f, beta = 0.0f;
  cublasStatus_t st;

  st = cublasGemmStridedBatchedEx(h,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  M, N, K,
                                  &alpha,
                                  dA, a_type, M, strideA,
                                  dB, b_type, K, strideB,
                                  &beta,
                                  dC, c_type, M, strideC,
                                  B,
                                  compute_type,
                                  CUBLAS_GEMM_DEFAULT);
  if(st != CUBLAS_STATUS_SUCCESS){
    return -1.0;
  }
  ckCuda(cudaDeviceSynchronize(), "sync warmup ex batched");

  const int reps = std::max(1, reps_target);
  double t0 = now_ms();
  for(int i=0;i<reps;i++){
    st = cublasGemmStridedBatchedEx(h,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    M, N, K,
                                    &alpha,
                                    dA, a_type, M, strideA,
                                    dB, b_type, K, strideB,
                                    &beta,
                                    dC, c_type, M, strideC,
                                    B,
                                    compute_type,
                                    CUBLAS_GEMM_DEFAULT);
    if(st != CUBLAS_STATUS_SUCCESS){
      return -1.0;
    }
  }
  ckCuda(cudaDeviceSynchronize(), "sync ex timed batched");
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
  bool use_ex;                    // FP16/FP8 use GemmEx; FP32 uses direct path
  cudaDataType_t a_type, b_type, c_type;
  cublasComputeType_t compute_type;
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
  return {"float32", "float16", "float8"};
}

static DTypeCfg make_dtype_cfg(const std::string& key){
  if(key=="float32"){
    return DTypeCfg{
      "float32", 4, false,
      CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
      CUBLAS_COMPUTE_32F
    };
  }
  if(key=="float16"){
    return DTypeCfg{
      "float16", 2, true,
      CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
      CUBLAS_COMPUTE_32F
    };
  }
  if(key=="float8"){
    return DTypeCfg{
      "float8", 1, true,
      CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3,
      CUBLAS_COMPUTE_32F
    };
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
      MB_dims = {1,2,4,8,16,32};
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
    if(cudaMemGetInfo(&freeB, &totalB) == cudaSuccess){
      mem_cap_gb = (double)freeB * 0.80 / (1024.0*1024.0*1024.0);
    }else{
      mem_cap_gb = 18.0;
    }
  }
  const size_t MEM_CAP_BYTES_BASE = (size_t)std::max(0.0, mem_cap_gb) * (size_t)(1024ULL*1024ULL*1024ULL);

  // dtype selection
  std::vector<std::string> dtypes = select_dtypes();

  struct DTypeOut { double tflops; std::map<int, std::map<int, std::vector<Rec>>> bins; };
  std::map<std::string, DTypeOut> all;

  // Create handle
  cublasHandle_t handle{};
  ckCublas(cublasCreate(&handle), "cublasCreate");
  ckCublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");

  // Run each dtype in sequence (one progress bar per dtype)
  for(const auto& dkey : dtypes){
    DTypeCfg cfg = make_dtype_cfg(dkey);
    const double TFLOPS = dtype_peak_tflops(cfg.key);
    const size_t MEM_CAP_BYTES = MEM_CAP_BYTES_BASE;

    // Global FP8 support probe (only used to detect "not implemented")
    bool dtype_supported = true;
    if(cfg.use_ex && dkey=="float8"){
      const int M=16,N=16,K=16;
      void *A=nullptr, *B=nullptr, *C=nullptr;
      ckCuda(cudaMalloc(&A, (size_t)M * (size_t)K * cfg.elem_bytes), "probe cudaMalloc A");
      ckCuda(cudaMalloc(&B, (size_t)K * (size_t)N * cfg.elem_bytes), "probe cudaMalloc B");
      ckCuda(cudaMalloc(&C, (size_t)M * (size_t)N * cfg.elem_bytes), "probe cudaMalloc C");
      float alpha=1.0f,beta=0.0f;
      cublasStatus_t st = cublasGemmEx(handle,
                                       CUBLAS_OP_N, CUBLAS_OP_N,
                                       M, N, K,
                                       &alpha,
                                       A, cfg.a_type, M,
                                       B, cfg.b_type, K,
                                       &beta,
                                       C, cfg.c_type, M,
                                       cfg.compute_type,
                                       CUBLAS_GEMM_DEFAULT);
      ckCuda(cudaFree(C), "probe cudaFree C");
      ckCuda(cudaFree(B), "probe cudaFree B");
      ckCuda(cudaFree(A), "probe cudaFree A");
      if(st == CUBLAS_STATUS_NOT_SUPPORTED || st == CUBLAS_STATUS_ARCH_MISMATCH){
        dtype_supported = false;
        std::fprintf(stderr, "[float8] cuBLAS reports FP8 not implemented on this platform; skipping.\n");
      }else if(st != CUBLAS_STATUS_SUCCESS){
        std::fprintf(stderr,
                     "[float8] cuBLAS FP8 probe got status %d; continuing with per-shape error handling.\n",
                     (int)st);
      }
    }
    if(!dtype_supported){
      all[dkey] = DTypeOut{TFLOPS, {}};
      continue;
    }

    std::map<int, std::map<int, std::vector<Rec>>> perB_bins;

    size_t total = (size_t)MB_dims.size() * M_dims.size() * N_dims.size() * K_dims.size();
    size_t idx = 0;

    for(int B : MB_dims){
      for(int M : M_dims){
        for(int N : N_dims){
          for(int K : K_dims){
            idx++;
            progress_with_cfg(cfg.key, (double)idx/(double)total, B, M,N,K, idx, total);

            if(total_bytes_batched(B,M,N,K,cfg.elem_bytes) > MEM_CAP_BYTES){
              continue;
            }

            const size_t strideA_elems = (size_t)M * (size_t)K;
            const size_t strideB_elems = (size_t)K * (size_t)N;
            const size_t strideC_elems = (size_t)M * (size_t)N;

            if(!cfg.use_ex){
              // FP32 path
              float *dA=nullptr, *dB=nullptr, *dC=nullptr;
              ckCuda(cudaMalloc(&dA, (size_t)B * strideA_elems * sizeof(float)), "cudaMalloc dA");
              ckCuda(cudaMalloc(&dB, (size_t)B * strideB_elems * sizeof(float)), "cudaMalloc dB");
              ckCuda(cudaMalloc(&dC, (size_t)B * strideC_elems * sizeof(float)), "cudaMalloc dC");
              ckCuda(cudaMemset(dC, 0, (size_t)B * strideC_elems * sizeof(float)), "cudaMemset dC");

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

              ckCuda(cudaFree(dC), "free dC");
              ckCuda(cudaFree(dB), "free dB");
              ckCuda(cudaFree(dA), "free dA");
            }else{
              // GEMM_EX path (FP16/FP8)
              void *dA=nullptr, *dB=nullptr, *dC=nullptr;
              ckCuda(cudaMalloc(&dA, (size_t)B * strideA_elems * (size_t)cfg.elem_bytes), "cudaMalloc dA");
              ckCuda(cudaMalloc(&dB, (size_t)B * strideB_elems * (size_t)cfg.elem_bytes), "cudaMalloc dB");
              ckCuda(cudaMalloc(&dC, (size_t)B * strideC_elems * (size_t)cfg.elem_bytes), "cudaMalloc dC");
              ckCuda(cudaMemset(dC, 0, (size_t)B * strideC_elems * (size_t)cfg.elem_bytes), "cudaMemset dC");

              const double target_ms = REPS_TARGET_MS;
              double ms = 0.0;

              if(B == 1){
                double probe_ms = run_gemm_ex_ms(handle, M,N,K,
                                                 dA,dB,dC,
                                                 cfg.a_type,cfg.b_type,cfg.c_type,
                                                 cfg.compute_type, 1);
                if(probe_ms < 0.0){
                  ckCuda(cudaFree(dC), "free dC");
                  ckCuda(cudaFree(dB), "free dB");
                  ckCuda(cudaFree(dA), "free dA");
                  continue;
                }
                int reps = std::max(1, (int)std::round(target_ms / std::max(0.1, probe_ms)));
                ms = run_gemm_ex_ms(handle, M,N,K,
                                    dA,dB,dC,
                                    cfg.a_type,cfg.b_type,cfg.c_type,
                                    cfg.compute_type, reps);
                if(ms < 0.0){
                  ckCuda(cudaFree(dC), "free dC");
                  ckCuda(cudaFree(dB), "free dB");
                  ckCuda(cudaFree(dA), "free dA");
                  continue;
                }
              }else{
                const long long sA = (long long)strideA_elems;
                const long long sB = (long long)strideB_elems;
                const long long sC = (long long)strideC_elems;
                double probe_ms = run_gemm_ex_strided_batched_ms(handle, B, M,N,K,
                                                                 dA,dB,dC, sA,sB,sC,
                                                                 cfg.a_type,cfg.b_type,cfg.c_type,
                                                                 cfg.compute_type, 1);
                if(probe_ms < 0.0){
                  ckCuda(cudaFree(dC), "free dC");
                  ckCuda(cudaFree(dB), "free dB");
                  ckCuda(cudaFree(dA), "free dA");
                  continue;
                }
                int reps = std::max(1, (int)std::round(target_ms / std::max(0.1, probe_ms)));
                ms = run_gemm_ex_strided_batched_ms(handle, B, M,N,K,
                                                    dA,dB,dC, sA,sB,sC,
                                                    cfg.a_type,cfg.b_type,cfg.c_type,
                                                    cfg.compute_type, reps);
                if(ms < 0.0){
                  ckCuda(cudaFree(dC), "free dC");
                  ckCuda(cudaFree(dB), "free dB");
                  ckCuda(cudaFree(dA), "free dA");
                  continue;
                }
              }

              const double gflops_total = (2.0 * (double)B * (double)M * (double)N * (double)K) / 1e9;
              const double achieved_gflops_per_s = (gflops_total * 1e3) / ms;
              const double eff = (TFLOPS > 0.0) ? (achieved_gflops_per_s / (TFLOPS * 1000.0)) : 0.0;

              const int bin = bin_from_gflops(gflops_total);
              perB_bins[B][bin].push_back({M,N,K,eff});

              ckCuda(cudaFree(dC), "free dC");
              ckCuda(cudaFree(dB), "free dB");
              ckCuda(cudaFree(dA), "free dA");
            }
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

  cublasDestroy(handle);
  return 0;
}
