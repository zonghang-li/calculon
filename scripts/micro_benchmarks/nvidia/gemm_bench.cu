// gemm_bench.cu
//
// Micro-batch + shape-aware GEMM efficiency benchmark (FP32), binned by
// total GFLOPs. Measures both single and strided-batched cuBLAS GEMMs and
// emits a JSON summary used to derive efficiency factors in performance models.
//
// What it does
// ------------
// * Sweeps an outer micro-batch dimension B and a coarse grid of (M, N, K).
// * Uses:
//     - cublasSgemm                when B == 1
//     - cublasSgemmStridedBatched  when B > 1
// * Skips shapes that exceed a memory cap (based on B * (A+B+C) bytes).
// * Autotunes the number of repetitions per shape to hit a target runtime.
// * Computes efficiency per shape as:
//       eff = (achieved_GFLOP/s) / (device_peak_GFLOP/s)
//   where achieved_GFLOP/s is measured and device peak is estimated from
//   device properties (coarse).
// * Bins shapes by total GFLOPs, with thresholds on 2*B*M*N*K / 1e9 ∈
//   { [512,∞), [64,512), [8,64), [1,8), [0,1) }.
//
// Output
// ------
// Writes ./gemm_bench.json with the exact schema:
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
//       "<B2>": { ... },
//       ...
//     }
//   }
// }
//
// Notes:
// * Shape keys are strings "M,N,K" to keep the JSON valid and compact.
// * Efficiencies are sorted descending per (B, bin) internally (for any
//   consumer that chooses to inspect before parsing).
//
// Environment knobs (all optional)
// --------------------------------
// FAST=0|1
//   If FAST=1 and MB_DIMS is not set, test only B in {1,2,4,8,16,32}.
//   Also sets DIM_STRIDE default to 2. (See DIM_STRIDE.)
// DIM_STRIDE=<int>
//   Subsample stride for M/N/K lists (default: FAST?2:1). The last value is
//   always kept to include the largest shape.
// MB_DIMS="b1,b2,..."
//   Explicit micro-batch sizes to test (e.g., MB_DIMS="1,2,4,8,16,32,64,128").
//   If unset: FAST? {1,2,4,8,16,32} : {1,2,4,8,16,32,64,128}.
// MB_DIM_STRIDE=<int>
//   Optional subsample stride for MB_DIMS (default: DIM_STRIDE). If FAST=1 and
//   MB_DIMS is unset, the exact set {1,2,4,8,16,32} is used (no thinning).
// COMMON_DIMS="a,b,c,..."
//   Sets the same candidate set for M, N, and K (overridden by M_DIMS/N_DIMS/K_DIMS).
// M_DIMS / N_DIMS / K_DIMS="a,b,c,..."
//   Per-axis candidate lists (win over COMMON_DIMS).
// MEM_CAP_GB=<double>
//   Max allowed total bytes per tested shape: B*(A+B+C). Default is ~80% of
//   current free device memory. Increase to admit larger shapes.
// REPS_AUTO_TARGET_MS=<double>
//   Auto-tune repetitions so that each (B,M,N,K) accumulates this many ms
//   total in the timed loop (default: 35 ms).
//
// Progress & Logging
// ------------------
// * Prints a single-line progress bar to stderr with the current (B,M,N,K).
// * On completion, prints “Wrote JSON: ./gemm_bench.json” to stderr.
//
// Build
// -----
//   nvcc -O3 -std=c++14 -lcublas gemm_bench.cu -o gemm_bench
//
// Caveats
// -------
// * Peak TFLOPS is a coarse estimate; efficiency values are relative.
// * This benchmark isolates GEMM kernel behavior (single and strided-batched).
//   Frameworks may use fused/epilogue kernels (e.g., cuBLASLt/CUTLASS) or
//   grouped GEMM scheduling that can achieve different utilization.
// * Memory cap uses raw tensor storage (A,B,C). Real models may reuse or fuse
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
#include <cuda_fp16.h>

// ----- small utils -----
static inline int getenv_int(const char* k, int defv){
  if(const char* s = std::getenv(k)) return std::atoi(s);
  return defv;
}
static inline double getenv_double(const char* k, double defv){
  if(const char* s = std::getenv(k)) return std::atof(s);
  return defv;
}
static std::string getenv_str(const char* k, const char* defv){
    if(const char* s = std::getenv(k)) return std::string(s);
    return std::string(defv);
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
static void progress_with_cfg(double frac, int B, int M,int N,int K, size_t i, size_t total){
  frac = std::max(0.0, std::min(1.0, frac));
  const int barw = 40;
  int filled = (int)std::round(frac * barw);
  std::fprintf(stderr, "\r[");
  for(int j=0;j<barw;j++) std::fputc(j<filled ? '#' : '-', stderr);
  std::fprintf(stderr, "] %5.1f%%  | Testing B=%d, M=%d, N=%d, K=%d  (%zu/%zu)",
               frac*100.0, B, M, N, K, i, total);
  if(frac>=1.0) std::fprintf(stderr, "  \n");
  std::fflush(stderr);
}

// Estimate FP32 TFLOPS peak (coarse)
static int fp32_cores_per_sm(const cudaDeviceProp& p){
  const int cc = p.major*10 + p.minor;
  if(cc >= 90) return 128; // Hopper/Ada fallback
  if(cc >= 86) return 128; // Ampere GA10x
  if(cc >= 80) return  64; // A100
  if(cc >= 75) return  64; // Turing
  if(cc >= 70) return  64; // Volta
  if(cc >= 61) return 128; // Pascal GP10x
  if(cc >= 60) return 128; // Pascal
  if(cc >= 50) return 128; // Maxwell fallback
  return 64;               // very old fallback
}
static double fp32_tflops_peak(){
  int dev = 0;
  ckCuda(cudaGetDevice(&dev), "cudaGetDevice");
  cudaDeviceProp p{};
  ckCuda(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");
  const double sms   = (double)p.multiProcessorCount;
  const double cores = (double)fp32_cores_per_sm(p);
  const double hz    = (double)p.clockRate * 1000.0; // kHz -> Hz
  return sms * cores * hz * 2.0 / 1e12; // FP32 FMA => *2 flops/cycle
}

template <typename T> struct GemmTraits;

template <> struct GemmTraits<float> {
    static const char* name() { return "float32"; }
    static float one() { return 1.0f; }
    static float zero() { return 0.0f; }
    
    static void gemm(cublasHandle_t handle, 
                     cublasOperation_t transa, cublasOperation_t transb,
                     int m, int n, int k,
                     const float* alpha, const float* A, int lda,
                     const float* B, int ldb,
                     const float* beta, float* C, int ldc) {
        ckCublas(cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc), "cublasSgemm");
    }

    static void gemmStridedBatched(cublasHandle_t handle,
                                   cublasOperation_t transa, cublasOperation_t transb,
                                   int m, int n, int k,
                                   const float* alpha, const float* A, int lda, long long strideA,
                                   const float* B, int ldb, long long strideB,
                                   const float* beta, float* C, int ldc, long long strideC,
                                   int batchCount) {
        ckCublas(cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount), "cublasSgemmStridedBatched");
    }
};

template <> struct GemmTraits<__half> {
    static const char* name() { return "float16"; }
    static __half one() { return __float2half(1.0f); }
    static __half zero() { return __float2half(0.0f); }

    static void gemm(cublasHandle_t handle, 
                     cublasOperation_t transa, cublasOperation_t transb,
                     int m, int n, int k,
                     const __half* alpha, const __half* A, int lda,
                     const __half* B, int ldb,
                     const __half* beta, __half* C, int ldc) {
        ckCublas(cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc), "cublasHgemm");
    }

    static void gemmStridedBatched(cublasHandle_t handle,
                                   cublasOperation_t transa, cublasOperation_t transb,
                                   int m, int n, int k,
                                   const __half* alpha, const __half* A, int lda, long long strideA,
                                   const __half* B, int ldb, long long strideB,
                                   const __half* beta, __half* C, int ldc, long long strideC,
                                   int batchCount) {
        ckCublas(cublasHgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount), "cublasHgemmStridedBatched");
    }
};

// bytes helpers
template <typename T>
static inline size_t bytes_A(int M,int K){ return (size_t)M * (size_t)K * sizeof(T); }
template <typename T>
static inline size_t bytes_B(int K,int N){ return (size_t)K * (size_t)N * sizeof(T); }
template <typename T>
static inline size_t bytes_C(int M,int N){ return (size_t)M * (size_t)N * sizeof(T); }
template <typename T>
static inline size_t total_bytes_batched(int B,int M,int N,int K){
  return (size_t)B * (bytes_A<T>(M,K) + bytes_B<T>(K,N) + bytes_C<T>(M,N));
}

// single SGEMM (averaged over reps)
template <typename T>
static double run_gemm_ms(cublasHandle_t h, int M,int N,int K,
                          T* dA, T* dB, T* dC, int reps_target){
  const T alpha = GemmTraits<T>::one();
  const T beta  = GemmTraits<T>::zero();
  
  GemmTraits<T>::gemm(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dA, M, dB, K, &beta, dC, M);
  ckCuda(cudaDeviceSynchronize(), "sync warmup");
  
  const int reps = std::max(1, reps_target);
  double t0 = now_ms();
  for(int i=0;i<reps;i++){
     GemmTraits<T>::gemm(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dA, M, dB, K, &beta, dC, M);
  }
  ckCuda(cudaDeviceSynchronize(), "sync timed");
  double t1 = now_ms();
  return (t1 - t0) / (double)reps;
}

// STRIDED-BATCHED SGEMM (averaged over reps)
template <typename T>
static double run_gemm_strided_batched_ms(cublasHandle_t h, int B, int M,int N,int K,
                                          T* dA, T* dB, T* dC,
                                          long long strideA, long long strideB, long long strideC,
                                          int reps_target){
  const T alpha = GemmTraits<T>::one();
  const T beta  = GemmTraits<T>::zero();

  GemmTraits<T>::gemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dA, M, strideA, dB, K, strideB, &beta, dC, M, strideC, B);
  ckCuda(cudaDeviceSynchronize(), "sync warmup batched");
  
  const int reps = std::max(1, reps_target);
  double t0 = now_ms();
  for(int i=0;i<reps;i++){
    GemmTraits<T>::gemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dA, M, strideA, dB, K, strideB, &beta, dC, M, strideC, B);
  }
  ckCuda(cudaDeviceSynchronize(), "sync timed batched");
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

template <typename T>
int bench_run(int FAST, double REPS_TARGET_MS, double mem_cap_gb, 
              std::vector<int> M_dims, std::vector<int> N_dims, std::vector<int> K_dims, std::vector<int> MB_dims,
              double TFLOPS_FP32) 
{
    const char* dtype_name = GemmTraits<T>::name();
    size_t MEM_CAP_BYTES = (size_t)std::max(0.0, mem_cap_gb) * (size_t)(1024ULL*1024ULL*1024ULL);

    // 对于 FP16，我们暂时使用 FP32 的峰值作为基准进行归一化（或者你可以乘以一个系数）
    // 为了 Calculon 模型一致性，通常使用 FP32 Peak 作为“Unit 1.0” 或者明确知道 Hardware 的 FP16 是 FP32 的几倍
    // 这里简单处理：如果 T是half，峰值理论上是 TFLOPS_FP32 * 8 (Ampere TensorCore) 左右
    // 但为了让 gemm_bench.json 的格式兼容，我们保持 TFLOPS 字段写入的是 FP32 峰值，
    // 而 efficiency = Measured / FP32_Peak。这样如果在 FP16 跑出 800% 的效率，用户知道是用 TensorCore 了。
    double BASE_TFLOPS = TFLOPS_FP32;

    std::fprintf(stdout, "=== Measuring %s GEMM with micro-batch and shape grid (GFLOPs-binned) ===\n", dtype_name);
    std::fflush(stdout);

    cublasHandle_t handle{};
    ckCublas(cublasCreate(&handle), "cublasCreate");
    // 启用 Tensor Core 模式 (对 FP16 至关重要)
    ckCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");

    std::map<int, std::map<int, std::vector<Rec>>> perB_bins;
    size_t total = (size_t)MB_dims.size() * M_dims.size() * N_dims.size() * K_dims.size();
    size_t idx = 0;

    for(int B : MB_dims){
        for(int M : M_dims){
            for(int N : N_dims){
                for(int K : K_dims){
                    idx++;
                    if(total_bytes_batched<T>(B,M,N,K) > MEM_CAP_BYTES){
                        progress_with_cfg((double)idx/(double)total, B, M,N,K, idx, total);
                        continue;
                    }

                    const size_t strideA_elems = (size_t)M * (size_t)K;
                    const size_t strideB_elems = (size_t)K * (size_t)N;
                    const size_t strideC_elems = (size_t)M * (size_t)N;
                    
                    T *dA=nullptr, *dB=nullptr, *dC=nullptr;
                    ckCuda(cudaMalloc(&dA, (size_t)B * strideA_elems * sizeof(T)), "cudaMalloc dA");
                    ckCuda(cudaMalloc(&dB, (size_t)B * strideB_elems * sizeof(T)), "cudaMalloc dB");
                    ckCuda(cudaMalloc(&dC, (size_t)B * strideC_elems * sizeof(T)), "cudaMalloc dC");
                    ckCuda(cudaMemset(dC, 0, (size_t)B * strideC_elems * sizeof(T)), "cudaMemset dC");

                    double ms = 0.0;
                    if(B == 1){
                        double probe_ms = run_gemm_ms<T>(handle, M,N,K, dA,dB,dC, 1);
                        int reps = std::max(1, (int)std::round(REPS_TARGET_MS / std::max(0.1, probe_ms)));
                        ms = run_gemm_ms<T>(handle, M,N,K, dA,dB,dC, reps);
                    }else{
                        const long long sA = (long long)strideA_elems;
                        const long long sB = (long long)strideB_elems;
                        const long long sC = (long long)strideC_elems;
                        double probe_ms = run_gemm_strided_batched_ms<T>(handle, B, M,N,K, dA,dB,dC, sA,sB,sC, 1);
                        int reps = std::max(1, (int)std::round(REPS_TARGET_MS / std::max(0.1, probe_ms)));
                        ms = run_gemm_strided_batched_ms<T>(handle, B, M,N,K, dA,dB,dC, sA,sB,sC, reps);
                    }

                    double gflops_total = (2.0 * (double)B * (double)M * (double)N * (double)K) / 1e9;
                    double achieved_gflops_per_s = (gflops_total * 1e3) / ms;
                    double eff = (BASE_TFLOPS > 0.0) ? (achieved_gflops_per_s / (BASE_TFLOPS * 1000.0)) : 0.0;

                    int bin = bin_from_gflops(gflops_total);
                    perB_bins[B][bin].push_back({M,N,K,eff});

                    ckCuda(cudaFree(dC), "free dC");
                    ckCuda(cudaFree(dB), "free dB");
                    ckCuda(cudaFree(dA), "free dA");

                    progress_with_cfg((double)idx/(double)total, B, M,N,K, idx, total);
                }
            }
        }
    }
    progress_with_cfg(1.0, 0,0,0,0, total, total);

    for(auto& kvB : perB_bins){
        for(auto& kvBin : kvB.second){
            auto& vec = kvBin.second;
            std::sort(vec.begin(), vec.end(), [](const Rec& a, const Rec& b){ return a.eff > b.eff; });
        }
    }

    // JSON Dump
    {
        std::ofstream f("./gemm_bench.json", std::ios::out | std::ios::trunc);
        if(!f) { std::fprintf(stderr, "Failed to open gemm_bench.json\n"); return 1; }
        f.setf(std::ios::fixed);
        f << std::setprecision(3);

        f << "{\n"
          << "  \"" << dtype_name << "\": {\n"  // 使用动态的 Key (float32 或 float16)
          << "    \"tflops\": " << BASE_TFLOPS << ",\n"
          << "    \"gflops_efficiency\": {\n";

        std::vector<int> Bs;
        for(const auto& kv : perB_bins) Bs.push_back(kv.first);
        std::sort(Bs.begin(), Bs.end());

        for(size_t bi=0; bi<Bs.size(); ++bi){
            int B = Bs[bi];
            const auto& bins = perB_bins[B];
            f << "      \"" << B << "\": {\n";
            for(size_t oi=0; oi<5; ++oi){
                int b = BIN_ORDER[oi];
                f << "        \"" << b << "\": {";
                auto it = bins.find(b);
                bool first = true;
                if(it != bins.end()){
                    for(const auto& r : it->second){
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
        f << "    }\n"
          << "  }\n"
          << "}\n";
        f.close();
    }
    cublasDestroy(handle);
    std::fprintf(stdout, "Wrote JSON: ./gemm_bench.json (Type: %s)\n", dtype_name);
    return 0;
}

int main(){
  // ---------------- configuration ----------------
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

  // For MB_dims: if FAST=1 and user did NOT set MB_DIMS, keep EXACT set {1,2,4,8,16,32} (no thinning).
  if(!(FAST && !mb_env_set)){
    MB_dims = stride_view(MB_dims, mb_dim_stride);
  }

  // Derive memory cap from free memory if not set
  if(mem_cap_gb <= 0.0){
    size_t freeB=0, totalB=0;
    if(cudaMemGetInfo(&freeB, &totalB) == cudaSuccess){
      mem_cap_gb = (double)freeB * 0.80 / (1024.0*1024.0*1024.0);
    }else{
      mem_cap_gb = 18.0; // fallback
    }
  }
  const size_t MEM_CAP_BYTES = (size_t)std::max(0.0, mem_cap_gb) * (size_t)(1024ULL*1024ULL*1024ULL);

  // TFLOPS peak for efficiency (coarse)
  const double TFLOPS_FP32 = fp32_tflops_peak();

  // Mode Selection based on env param
  std::string prec = getenv_str("PRECISION", "float32");
  
  if(prec == "float16" || prec == "fp16" || prec == "half") {
      return bench_run<__half>(FAST, REPS_TARGET_MS, mem_cap_gb, M_dims, N_dims, K_dims, MB_dims, TFLOPS_FP32);
  } else {
      return bench_run<float>(FAST, REPS_TARGET_MS, mem_cap_gb, M_dims, N_dims, K_dims, MB_dims, TFLOPS_FP32);
  }
}
