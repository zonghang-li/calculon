/**
 * net_bw.cu — single-node network/comms micro-benchmark (CUDA + NCCL optional)
 *
 * What this program measures
 * --------------------------
 * Quickly characterizes “Megatron-like” communication on a single node across 2/4/8 GPUs:
 *   • PP (pipeline-parallel) steady-state 1F1B traffic between adjacent ranks
 *     using ncclSend/ncclRecv (or cudaMemcpyPeer when built with NO_NCCL).
 *   • AR (data/tensor-parallel) NCCL AllReduce (ring) bus bandwidth.
 *   • A small-message one-way P2P latency via 4 KiB cudaMemcpyPeer.
 *
 * Megatron-LM alignment
 * ---------------------
 * • By default, lets NCCL auto-select algorithm/protocol/rings — mirroring Megatron-LM.
 * • You can pin a deterministic config for A/B by setting: PIN_NCCL=1
 *   (Ring algo, Simple proto, 1 ring, CollNet disabled).
 * • Logging is quiet by default (NCCL_DEBUG=ERROR) unless you override it.
 *
 * Output (file)
 * -------------
 * Writes JSON to: ./net_bw.json
 *
 * Schema (abridged):
 * {
 *   "networks": [
 *     {
 *       "bandwidth": <theory_GBps>,         // single-link “theory” GB/s used for normalization
 *       "pp_efficiency": [[MiB, eff], ...], // size-binned PP efficiencies (including [0, eff0] tiny-msg anchor)
 *       "ar_efficiency": [[MiB, eff], ...], // size-binned AR efficiencies (including [0, eff0] tiny-msg anchor)
 *       "size": <2|4|8>,                     // run size (participating GPUs)
 *       "latency": <seconds>,                // small one-way P2P latency (4 KiB), seconds
 *       "ops": { "p2p":[1.0,null], "reduce_scatter":[1.0,-1], "all_gather":[1.0,-1], "all_reduce":[2.0,-1] },
 *       "must_be_filled": <bool>,            // hint for model builders
 *       "processor_usage": <0.03|0.04|0.05>  // light heuristic per size
 *     },
 *     ...
 *   ]
 * }
 *
 * Theory bandwidth (normalization)
 * --------------------------------
 * The “bandwidth” field is a single-link theory GB/s used to normalize efficiencies:
 *   1) If P2P_THEORY_GBPS > 0 → use it.
 *   2) Else if PCIE_GEN & PCIE_WIDTH set → per-lane payload GB/s × width.
 *   3) Else if NVLINK_GBPS & NVLINK_LINKS set → product.
 *   4) Else infer a conservative bin from an 8 MiB unidirectional plateau copy.
 *
 * Efficiency binning vs message size
 * ----------------------------------
 * • LLM payload sizes are a CSV list (MiB). For each size we measure:
 *     - PP: forward throughput across the boundary pair in a timed 1F1B loop.
 *     - AR: ring AllReduce “BusBW” (moved bytes per iter / time).
 * • Each efficiency entry is normalized by the “bandwidth” above.
 * • A tiny-message anchor [0, eff] is appended:
 *     - PP: from 4 KiB one-way cudaMemcpyPeer latency.
 *     - AR: from ~1 KiB AllReduce latency (ring model for moved bytes).
 *
 * Runs & topology
 * ---------------
 * • Runs attempt sizes 2, 4, and 8 when enough visible GPUs exist.
 * • Device sets and the boundary pair are configurable (see env below).
 * • CUDA P2P access is enabled across all pairs when possible.
 *
 * Progress & logging (matches gemm_bench.cu style)
 * ------------------------------------------------
 * • A single-line progress bar is printed to stderr:
 *     [########------------------------] 37.5% | PP  size=4  payload=32 MiB  (i/total)
 * • On completion, prints:  Wrote JSON: ./net_bw.json
 *
 * Key environment variables
 * -------------------------
 *  Device selection & boundary (per run size)
 *    SET2, SET4, SET8        Comma-separated device IDs forming the ordered set (default [0..N-1]).
 *    BOUND2, BOUND4, BOUND8  Two IDs “L,R” that must be adjacent in the set; this pair is the boundary.
 *                            Defaults: size=2 → "0,1"; size=4 → "1,2"; size=8 → "3,4".
 *
 *  Payloads & loop counts
 *    LLM_PAYLOAD_MB          CSV MiB list for PP/AR (default: 128,96,64,32,16,8,4,2,1).
 *    PP_WARMUP               Warm-up steps for PP 1F1B (default: 20).
 *    PP_STEPS                Timed PP steps (default: 400).
 *    AR_ITERS                Timed AllReduce iterations (default: 200).
 *
 *  Normalization / theory bandwidth
 *    P2P_THEORY_GBPS         Force the theory GB/s (overrides all inference).
 *    PCIE_GEN                3|4|5 (payload GB/s per lane ≈ 0.985/1.969/3.938).
 *    PCIE_WIDTH              PCIe lane count (e.g., 16).
 *    NVLINK_GBPS             Per-link payload GB/s (e.g., 50).
 *    NVLINK_LINKS            Number of NVLink links between boundary GPUs.
 *
 *  Output shaping
 *    MUST_FILL_POLICY        SINGLE|ALL|NONE (default SINGLE).
 *                            SINGLE → only the fastest theory BW (ties → smaller size) sets must_be_filled=true.
 *
 *  NCCL control (optional)
 *    PIN_NCCL=1              Pin Ring + Simple + 1 ring + CollNet off (deterministic A/B).
 *    NCCL_DEBUG=*            If you set NCCL_DEBUG yourself, your setting is respected.
 *
 * Build
 * -----
 *   With NCCL (recommended):
 *     nvcc -O3 -std=c++14 net_bw.cu -lnccl -o builds/net_bw
 *
 *   Without NCCL (PP uses cudaMemcpyPeer, AR skipped):
 *     nvcc -O3 -std=c++14 -DNO_NCCL net_bw.cu -o builds/net_bw
 *
 * Example runs
 * ------------
 *   # Megatron-like (auto NCCL), default payload grid
 *   ./builds/net_bw
 *
 *   # Pin deterministic NCCL selections for comparison
 *   PIN_NCCL=1 ./builds/net_bw
 *
 *   # Custom payloads (MiB)
 *   LLM_PAYLOAD_MB=128,64,32,16,8,4,2,1 ./builds/net_bw
 *
 * Notes & caveats
 * ---------------
 * • This focuses on single-node tiers (2/4/8) and steady-state traffic.
 * • Efficiencies are relative to the inferred/forced “theory” link GB/s.
 * • Real training may overlap compute/comm differently from this micro-bench.
 */


#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <thread>
#include <atomic>
#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>
#include <iomanip>

#ifndef _WIN32
  #include <unistd.h>
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <sys/types.h>
#else
  #include <process.h>
  #include <direct.h>
#endif

#ifndef NO_NCCL
  #include <nccl.h>
#endif

// ----------------------------- Utils -----------------------------
static inline const char* getenv_cstr(const char* k, const char* defv){
  const char* s = std::getenv(k);
  return s ? s : defv;
}
static inline int getenv_int(const char* k, int defv){
  const char* s = std::getenv(k);
  return s ? std::atoi(s) : defv;
}
static inline double getenv_double(const char* k, double defv){
  const char* s = std::getenv(k);
  return s ? std::atof(s) : defv;
}
static inline void ckCuda(cudaError_t e, const char* where){
  if(e != cudaSuccess){
    std::fprintf(stderr, "CUDA error %d (%s) at %s\n",
                 (int)e, cudaGetErrorString(e), where);
#ifdef _WIN32
    std::exit(1);
#else
    _exit(1);
#endif
  }
}
#ifndef NO_NCCL
static inline void ckNccl(ncclResult_t e, const char* where){
  if(e != ncclSuccess){
    std::fprintf(stderr, "NCCL error %d (%s) at %s\n",
                 (int)e, ncclGetErrorString(e), where);
#ifdef _WIN32
    std::exit(1);
#else
    _exit(1);
#endif
  }
}
#endif
static inline double now_ms(){
  using clk = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}
static inline size_t MB(double x){ return (size_t)std::llround(x * 1024.0 * 1024.0); }

// ----- Progress bar (MATCHES gemm_bench.cu style) -----
static void progress_with_cfg(double frac, const char* phase, int run_size, int payload_mib,
                              size_t i, size_t total){
  frac = std::max(0.0, std::min(1.0, frac));
  const int barw = 40;
  int filled = (int)std::round(frac * barw);
  std::fprintf(stderr, "\r[");
  for(int j=0;j<barw;j++) std::fputc(j<filled ? '#' : '-', stderr);
  std::fprintf(stderr, "] %5.1f%%  | %s  size=%d  payload=%d MiB  (%zu/%zu)",
               frac*100.0, phase, run_size, payload_mib, i, total);
  if(frac>=1.0) std::fprintf(stderr, "  \n");
  std::fflush(stderr);
}

// Parse MiB CSV
static std::vector<double> parse_mib_list_from_env(const char* key){
  const char* s = std::getenv(key);
  if(!s || !*s){
    return {128,96,64,32,16,8,4,2,1};
  }
  std::vector<double> out;
  std::istringstream iss; iss.str(std::string(s));
  std::string tok;
  while(std::getline(iss, tok, ',')){
    if(tok.empty()) continue;
    size_t start = tok.find_first_not_of(" \t");
    size_t end   = tok.find_last_not_of(" \t");
    if(start==std::string::npos) continue;
    double v = std::atof(tok.substr(start, end-start+1).c_str());
    if(v > 0.0) out.push_back(v);
  }
  if(out.empty()) out = {128,96,64,32,16,8,4,2,1};
  return out;
}

// Infer “theory” GB/s for a single P2P link
static double infer_theory_gbps(double measured_plateau_gbps){
  double forced = getenv_double("P2P_THEORY_GBPS", -1.0);
  if(forced > 0.0) return forced;

  int gen = getenv_int("PCIE_GEN", 0);
  int w   = getenv_int("PCIE_WIDTH", 0);
  if(gen>0 && w>0){
    double per_lane = (gen==3?0.985:(gen==4?1.969:(gen==5?3.938:0.985)));
    return per_lane * std::max(1, w);
  }
  double nvl = getenv_double("NVLINK_GBPS", -1.0);
  int links  = getenv_int("NVLINK_LINKS", 0);
  if(nvl>0.0 && links>0) return nvl * links;

  if(measured_plateau_gbps >= 95.0) return 100.0;
  if(measured_plateau_gbps >= 45.0) return 50.0;
  if(measured_plateau_gbps >= 28.0) return 31.5;   // PCIe Gen4 x8-ish
  if(measured_plateau_gbps >= 13.0) return 15.75;  // PCIe Gen3/4 x16-ish older gen
  if(measured_plateau_gbps >= 7.0)  return 7.88;   // PCIe Gen3/4 x8-ish
  return std::max(7.0, measured_plateau_gbps);
}

// Enable CUDA P2P if possible
static void enable_p2p(int a, int b){
  int canAB=0, canBA=0;
  ckCuda(cudaDeviceCanAccessPeer(&canAB, a, b), "canAB");
  ckCuda(cudaDeviceCanAccessPeer(&canBA, b, a), "canBA");
  if(canAB){ cudaSetDevice(a); cudaDeviceEnablePeerAccess(b,0); cudaGetLastError(); }
  if(canBA){ cudaSetDevice(b); cudaDeviceEnablePeerAccess(a,0); cudaGetLastError(); }
}

// 4 KiB one-way latency via cudaMemcpyPeerAsync
static double latency_us_pair_cudacpy(int dst, int src){
  const size_t bytes = 4 * 1024;
  const int reps = 4000;
  cudaSetDevice(dst);
  void* dDst=nullptr; ckCuda(cudaMalloc(&dDst, bytes), "lat dst");
  cudaSetDevice(src);
  void* dSrc=nullptr; ckCuda(cudaMalloc(&dSrc, bytes), "lat src");
  cudaSetDevice(dst);
  cudaStream_t s; ckCuda(cudaStreamCreate(&s), "lat s");
  for(int i=0;i<50;i++) ckCuda(cudaMemcpyPeerAsync(dDst,dst,dSrc,src,bytes,s),"lat warm");
  ckCuda(cudaStreamSynchronize(s), "lat warm sync");
  double t0 = now_ms();
  for(int i=0;i<reps;i++) ckCuda(cudaMemcpyPeerAsync(dDst,dst,dSrc,src,bytes,s),"lat copy");
  ckCuda(cudaStreamSynchronize(s), "lat sync");
  double t1 = now_ms();
  cudaStreamDestroy(s);
  cudaFree(dDst); cudaSetDevice(src); cudaFree(dSrc);
  return (t1 - t0) * 1000.0 / reps; // us
}

// ----------------------------- Run definition -----------------------------
struct RunDef {
  int size;                 // number of GPUs in this run
  std::vector<int> devs;    // device ids in order (pipeline order)
  int boundary_left_idx;    // index 'i' so boundary pair is (i, i+1) in 'devs'
};
static std::vector<int> parse_devlist(const char* s){
  std::vector<int> v;
  std::string str = s ? s : "";
  if(str.empty()) return v;
  std::istringstream iss; iss.str(str);
  std::string tok;
  while(std::getline(iss, tok, ',')){
    if(!tok.empty()) v.push_back(std::atoi(tok.c_str()));
  }
  return v;
}
static RunDef make_run_from_env(int want_size, const char* env_set, const char* env_bound_default){
  RunDef r{};
  r.size = want_size;
  std::vector<int> def; for(int i=0;i<want_size;i++) def.push_back(i);
  std::vector<int> set = parse_devlist(std::getenv(env_set));
  if((int)set.size() != want_size) set = def;
  r.devs = set;
  std::vector<int> bound = parse_devlist(getenv_cstr((want_size==2?"BOUND2":(want_size==4?"BOUND4":"BOUND8")), env_bound_default));
  int bi = 0;
  if(bound.size()==2){
    for(int i=0;i<want_size-1;i++){
      if(r.devs[i]==bound[0] && r.devs[i+1]==bound[1]){ bi = i; break; }
    }
  }
  r.boundary_left_idx = bi;
  return r;
}

// ----------------------------- PP: 1F1B steady-state -----------------------------
#ifndef NO_NCCL
struct PPRank {
  int rank, world, dev;
  ncclComm_t comm;
  cudaStream_t stream;
  float *fwd_send, *fwd_recv, *bwd_send, *bwd_recv;
  size_t elems;
  float ms_per_step;
};

static void run_pp_1f1b_meas(const RunDef& rd, size_t bytes, int warmup_steps, int timed_steps,
                             int boundary_left_rank,
                             double& out_forward_gbps,
                             double& out_small_lat_s){
  const int W = rd.size;
  const int left  = boundary_left_rank;
  const int right = boundary_left_rank + 1;

  ncclUniqueId uid; ckNccl(ncclGetUniqueId(&uid), "pp getuid");

  std::vector<PPRank> rr(W);
  for(int r=0;r<W;r++){
    rr[r] = PPRank{
      r, W, rd.devs[r],
      nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr,
      std::max<size_t>(1, bytes/sizeof(float)),
      0.f
    };
  }

  auto boot = [&](int r){
    ckCuda(cudaSetDevice(rr[r].dev),"pp set dev");
    ckCuda(cudaStreamCreate(&rr[r].stream),"pp stream");
    ckCuda(cudaMalloc(&rr[r].fwd_send, rr[r].elems*sizeof(float)),"pp malloc fs");
    ckCuda(cudaMalloc(&rr[r].fwd_recv, rr[r].elems*sizeof(float)),"pp malloc fr");
    ckCuda(cudaMalloc(&rr[r].bwd_send, rr[r].elems*sizeof(float)),"pp malloc bs");
    ckCuda(cudaMalloc(&rr[r].bwd_recv, rr[r].elems*sizeof(float)),"pp malloc br");
    ckNccl(ncclCommInitRank(&rr[r].comm, rr[r].world, uid, rr[r].rank), "pp comm");
  };
  auto cleanup = [&](int r){
    ncclCommDestroy(rr[r].comm);
    cudaFree(rr[r].fwd_send);
    cudaFree(rr[r].fwd_recv);
    cudaFree(rr[r].bwd_send);
    cudaFree(rr[r].bwd_recv);
    cudaStreamDestroy(rr[r].stream);
  };
  { std::vector<std::thread> t; for(int r=0;r<W;r++) t.emplace_back(boot, r); for(auto& th: t) th.join(); }

  struct SharedStart { std::atomic<int> ready{0}; std::atomic<bool> go{false}; } ss;

  auto worker = [&](int r){
    ckCuda(cudaSetDevice(rr[r].dev),"pp wrk set");
    for(int i=0;i<warmup_steps;i++){
      ckNccl(ncclGroupStart(), "pp grp warm start");
      if(r < W-1) ckNccl(ncclSend(rr[r].fwd_send, (ssize_t)rr[r].elems, ncclFloat32, r+1, rr[r].comm, rr[r].stream), "pp fwd send warm");
      if(r > 0   ) ckNccl(ncclRecv(rr[r].fwd_recv, (ssize_t)rr[r].elems, ncclFloat32, r-1, rr[r].comm, rr[r].stream), "pp fwd recv warm");
      if(r > 0   ) ckNccl(ncclSend(rr[r].bwd_send, (ssize_t)rr[r].elems, ncclFloat32, r-1, rr[r].comm, rr[r].stream), "pp bwd send warm");
      if(r < W-1) ckNccl(ncclRecv(rr[r].bwd_recv, (ssize_t)rr[r].elems, ncclFloat32, r+1, rr[r].comm, rr[r].stream), "pp bwd recv warm");
      ckNccl(ncclGroupEnd(), "pp grp warm end");
    }
    ckCuda(cudaStreamSynchronize(rr[r].stream),"pp warm sync");

    ss.ready.fetch_add(1);
    while(!ss.go.load(std::memory_order_acquire)) {}

    cudaEvent_t e0=nullptr, e1=nullptr;
    if(r == left){ ckCuda(cudaEventCreate(&e0),"pp e0"); ckCuda(cudaEventCreate(&e1),"pp e1"); }

    if(r == left) ckCuda(cudaEventRecord(e0, rr[r].stream), "pp rec e0");
    for(int it=0; it<timed_steps; ++it){
      ckNccl(ncclGroupStart(), "pp grp start");
      if(r < W-1) ckNccl(ncclSend(rr[r].fwd_send, (ssize_t)rr[r].elems, ncclFloat32, r+1, rr[r].comm, rr[r].stream), "pp fwd send");
      if(r > 0   ) ckNccl(ncclRecv(rr[r].fwd_recv, (ssize_t)rr[r].elems, ncclFloat32, r-1, rr[r].comm, rr[r].stream), "pp fwd recv");
      if(r > 0   ) ckNccl(ncclSend(rr[r].bwd_send, (ssize_t)rr[r].elems, ncclFloat32, r-1, rr[r].comm, rr[r].stream), "pp bwd send");
      if(r < W-1) ckNccl(ncclRecv(rr[r].bwd_recv, (ssize_t)rr[r].elems, ncclFloat32, r+1, rr[r].comm, rr[r].stream), "pp bwd recv");
      ckNccl(ncclGroupEnd(), "pp grp end");
    }
    if(r == left) ckCuda(cudaEventRecord(e1, rr[r].stream), "pp rec e1");

    ckCuda(cudaStreamSynchronize(rr[r].stream),"pp timed sync");

    if(r == left){
      float ms=0.f; ckCuda(cudaEventElapsedTime(&ms, e0, e1), "pp elapsed");
      rr[r].ms_per_step = ms / std::max(1, timed_steps);
      cudaEventDestroy(e0); cudaEventDestroy(e1);
    }else{
      rr[r].ms_per_step = 0.f;
    }
  };

  std::vector<std::thread> workers;
  for(int r=0;r<W;r++) workers.emplace_back(worker, r);
  while(ss.ready.load() < W) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  ss.go.store(true, std::memory_order_release);
  for(auto& th: workers) th.join();

  float ms_step = rr[left].ms_per_step;
  if(ms_step <= 0.f) ms_step = 1e3f;
  double gb = (double)bytes / 1e9;
  out_forward_gbps = gb / ( (double)ms_step / 1000.0 );

  out_small_lat_s = latency_us_pair_cudacpy(rd.devs[right], rd.devs[left]) / 1e6;

  { std::vector<std::thread> t; for(int r=0;r<W;r++) t.emplace_back(cleanup, r); for(auto& th: t) th.join(); }
}
#else
static void run_pp_1f1b_meas(const RunDef& rd, size_t bytes, int, int, int boundary_left_rank,
                             double& out_forward_gbps, double& out_small_lat_s){
  const int left  = rd.devs[boundary_left_rank];
  const int right = rd.devs[boundary_left_rank+1];

  cudaSetDevice(right);
  void* dDst=nullptr; ckCuda(cudaMalloc(&dDst, bytes),"pp dst");
  cudaSetDevice(left);
  void* dSrc=nullptr; ckCuda(cudaMalloc(&dSrc, bytes),"pp src");
  cudaSetDevice(right);
  cudaStream_t s; ckCuda(cudaStreamCreate(&s),"pp stream");

  for(int i=0;i<20;i++) ckCuda(cudaMemcpyPeerAsync(dDst,right,dSrc,left,bytes,s),"pp warm");
  ckCuda(cudaStreamSynchronize(s),"pp warm sync");

  int reps = 50;
  double t0=now_ms();
  for(int i=0;i<reps;i++) ckCuda(cudaMemcpyPeerAsync(dDst,right,dSrc,left,bytes,s),"pp copy");
  ckCuda(cudaStreamSynchronize(s),"pp sync");
  double t1=now_ms();

  double ms = (t1 - t0) / std::max(1,reps);
  out_forward_gbps = bytes / (ms * 1e6);
  out_small_lat_s = latency_us_pair_cudacpy(right, left) / 1e6;

  cudaStreamDestroy(s);
  cudaFree(dDst); cudaSetDevice(left); cudaFree(dSrc);
}
#endif

// ----------------------------- AR: NCCL AllReduce (ring) -----------------------------
#ifndef NO_NCCL
struct ARRank {
  int rank, world, dev;
  ncclComm_t comm;
  cudaStream_t stream;
  float* buf;
  size_t elems;
  float ms_per_iter;
  float small_ms;
};

static void run_ar_collective(const RunDef& rd, size_t bytes, int iters,
                              double& out_busbw_gbps, double& out_small_lat_s){
  const int W = rd.size;
  size_t elems = std::max<size_t>(1, bytes/sizeof(float));
  size_t used_bytes = elems*sizeof(float);

  ncclUniqueId uid; ckNccl(ncclGetUniqueId(&uid), "ar getuid");

  std::vector<ARRank> rr(W);
  for(int r=0;r<W;r++){
    rr[r] = ARRank{
      r, W, rd.devs[r],
      nullptr, nullptr,
      nullptr, elems, 0.f, 0.f
    };
  }

  auto boot = [&](int r){
    ckCuda(cudaSetDevice(rr[r].dev), "ar set");
    ckCuda(cudaStreamCreate(&rr[r].stream), "ar stream");
    ckCuda(cudaMalloc(&rr[r].buf, rr[r].elems*sizeof(float)), "ar malloc");
    ckNccl(ncclCommInitRank(&rr[r].comm, rr[r].world, uid, rr[r].rank), "ar comm");
  };
  auto cleanup = [&](int r){
    ncclCommDestroy(rr[r].comm);
    cudaFree(rr[r].buf);
    cudaStreamDestroy(rr[r].stream);
  };
  { std::vector<std::thread> t; for(int r=0;r<W;r++) t.emplace_back(boot, r); for(auto& th: t) th.join(); }

  struct SharedStart { std::atomic<int> ready{0}; std::atomic<bool> go{false}; } ss;

  auto worker = [&](int r){
    ckCuda(cudaSetDevice(rr[r].dev),"ar wrk set");
    for(int i=0;i<5;i++){
      ckNccl(ncclAllReduce(rr[r].buf, rr[r].buf, rr[r].elems, ncclFloat32, ncclSum, rr[r].comm, rr[r].stream), "ar warm");
    }
    ckCuda(cudaStreamSynchronize(rr[r].stream),"ar warm sync");

    ss.ready.fetch_add(1);
    while(!ss.go.load(std::memory_order_acquire)) {}

    cudaEvent_t e0,e1; ckCuda(cudaEventCreate(&e0),"ar e0"); ckCuda(cudaEventCreate(&e1),"ar e1");
    ckCuda(cudaEventRecord(e0, rr[r].stream),"ar rec0");
    for(int it=0; it<iters; ++it){
      ckNccl(ncclAllReduce(rr[r].buf, rr[r].buf, rr[r].elems, ncclFloat32, ncclSum, rr[r].comm, rr[r].stream), "ar iter");
    }
    ckCuda(cudaEventRecord(e1, rr[r].stream),"ar rec1");
    ckCuda(cudaEventSynchronize(e1),"ar sync e1");
    float ms=0.f; ckCuda(cudaEventElapsedTime(&ms, e0, e1),"ar elapsed");
    rr[r].ms_per_iter = ms / std::max(1,iters);
    cudaEventDestroy(e0); cudaEventDestroy(e1);

    size_t small_elems = std::max<size_t>(1, 1024/sizeof(float));
    cudaEvent_t l0,l1; ckCuda(cudaEventCreate(&l0),"l0"); ckCuda(cudaEventCreate(&l1),"l1");
    ckCuda(cudaEventRecord(l0, rr[r].stream),"l0rec");
    ckNccl(ncclAllReduce(rr[r].buf, rr[r].buf, small_elems, ncclFloat32, ncclSum, rr[r].comm, rr[r].stream), "lat ar");
    ckCuda(cudaEventRecord(l1, rr[r].stream),"l1rec");
    ckCuda(cudaEventSynchronize(l1),"lat sync");
    float lms=0.f; ckCuda(cudaEventElapsedTime(&lms,l0,l1),"lat el");
    rr[r].small_ms = lms;
    cudaEventDestroy(l0); cudaEventDestroy(l1);
  };

  std::vector<std::thread> workers; workers.reserve(W);
  for(int r=0;r<W;r++) workers.emplace_back(worker, r);
  while(ss.ready.load() < W) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  ss.go.store(true, std::memory_order_release);
  for(auto& th: workers) th.join();

  float ms_iter = 0.f, sml_ms = 0.f;
  for(int r=0;r<W;r++){ ms_iter = std::max(ms_iter, rr[r].ms_per_iter);
                        sml_ms  = std::max(sml_ms,  rr[r].small_ms); }

  double moved = 2.0 * (W - 1.0) / (double)W * (double)used_bytes;
  out_busbw_gbps  = moved / (ms_iter/1000.0) / 1e9;
  out_small_lat_s = sml_ms / 1000.0;

  { std::vector<std::thread> t; for(int r=0;r<W;r++) t.emplace_back(cleanup, r); for(auto& th: t) th.join(); }
}
#else
static void run_ar_collective(const RunDef&, size_t, int, double& bw, double& l){
  bw = 0.0; l = 5e-5;
}
#endif

// ----------------------------- File helpers -----------------------------
static void set_env_if_empty(const char* k, const char* v){
#ifdef _WIN32
  const char* cur = std::getenv(k);
  if(!cur || !*cur) _putenv_s(k, v);
#else
  const char* cur = std::getenv(k);
  if(!cur || !*cur) setenv(k, v, 1);
#endif
}

static bool ensure_dir(const std::string& path){
#ifdef _WIN32
  if(_mkdir(path.c_str()) == 0) return true;
  if(errno == EEXIST) return true;
  return false;
#else
  if(mkdir(path.c_str(), 0755) == 0) return true;
  if(errno == EEXIST) return true;
  return false;
#endif
}

static std::string timestamp_yyyyMMdd_HHmmss(){
  std::time_t t = std::time(nullptr);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm);
  return std::string(buf);
}

static long get_pid(){
#ifdef _WIN32
  return (long)_getpid();
#else
  return (long)getpid();
#endif
}

// ----------------------------- Main -----------------------------
int main(){
  // Quiet logging unless user explicitly set NCCL_DEBUG*
  set_env_if_empty("NCCL_DEBUG", "ERROR");
#ifdef _WIN32
  set_env_if_empty("NCCL_DEBUG_FILE", "NUL");
#else
  set_env_if_empty("NCCL_DEBUG_FILE", "/dev/null");
#endif
  set_env_if_empty("NCCL_DEBUG_SUBSYS", "INIT");

  // Optional: pin NCCL like old benchmark if PIN_NCCL=1
  if(getenv_int("PIN_NCCL", 0) != 0){
#ifdef _WIN32
    _putenv_s("NCCL_ALGO", "Ring");
    _putenv_s("NCCL_PROTO", "Simple");
    _putenv_s("NCCL_MIN_NRINGS", "1");
    _putenv_s("NCCL_MAX_NRINGS", "1");
    _putenv_s("NCCL_COLLNET_ENABLE", "0");
#else
    setenv("NCCL_ALGO", "Ring", 1);
    setenv("NCCL_PROTO", "Simple", 1);
    setenv("NCCL_MIN_NRINGS", "1", 1);
    setenv("NCCL_MAX_NRINGS", "1", 1);
    setenv("NCCL_COLLNET_ENABLE", "0", 1);
#endif
  }

  int ndev=0; ckCuda(cudaGetDeviceCount(&ndev), "getDeviceCount");
  if(ndev < 2){ std::printf("{\"networks\": []}\n"); return 0; }

  RunDef R2  = make_run_from_env(2, "SET2", "0,1");
  RunDef R4  = make_run_from_env(4, "SET4", "1,2");
  RunDef R8  = make_run_from_env(8, "SET8", "3,4");

  for(int i=0;i<ndev;i++) for(int j=0;j<ndev;j++) if(i!=j) enable_p2p(i,j);

  const std::vector<double> PAYLOADS_MIB = parse_mib_list_from_env("LLM_PAYLOAD_MB");
  const int    PP_WARM   = getenv_int("PP_WARMUP", 20);
  const int    PP_STEPS  = getenv_int("PP_STEPS", 400);
  const int    AR_ITERS  = getenv_int("AR_ITERS", 200);

  const int run_count = 1 + (ndev>=4 ? 1:0) + (ndev>=8 ? 1:0);
  const size_t total_work = (size_t)run_count * PAYLOADS_MIB.size() * 2;
  size_t idx = 0;

  struct OutItem{
    int size;
    double theory_bw;
    std::vector<std::pair<int,double>> pp_eff_pairs; // [MiB, eff]
    std::vector<std::pair<int,double>> ar_eff_pairs; // [MiB, eff]
    double lat_s;
    bool must;
    double cpu;
  };

  auto run_one = [&](const RunDef& rd)->OutItem{
    const int bi = rd.boundary_left_idx;
    const int devL = rd.devs[bi];
    const int devR = rd.devs[bi+1];

    const size_t PROBE = MB(8.0);
    auto unidir_gbps = [&](int dst, int src)->double{
      cudaSetDevice(dst);
      void* dDst=nullptr; ckCuda(cudaMalloc(&dDst, PROBE), "uni dst");
      cudaSetDevice(src);
      void* dSrc=nullptr; ckCuda(cudaMalloc(&dSrc, PROBE), "uni src");
      cudaSetDevice(dst);
      cudaStream_t s; ckCuda(cudaStreamCreate(&s), "uni stream");
      for(int w=0;w<8;w++) ckCuda(cudaMemcpyPeerAsync(dDst,dst,dSrc,src,PROBE,s),"uni warm");
      ckCuda(cudaStreamSynchronize(s), "uni warm sync");
      const int reps = 40;
      double t0=now_ms();
      for(int i=0;i<reps;i++) ckCuda(cudaMemcpyPeerAsync(dDst,dst,dSrc,src,PROBE,s),"uni copy");
      ckCuda(cudaStreamSynchronize(s), "uni sync");
      double t1=now_ms();
      cudaStreamDestroy(s);
      cudaFree(dDst); cudaSetDevice(src); cudaFree(dSrc);
      double ms = (t1 - t0) / reps;
      return PROBE / (ms * 1e6); // GB/s
    };
    double plateau = std::max(unidir_gbps(devR,devL), unidir_gbps(devL,devR));
    double theory  = infer_theory_gbps(plateau);

    std::vector<std::pair<int,double>> pp_pairs;
    std::vector<std::pair<int,double>> ar_pairs;

    double small_pp_lat_s = 0.0;
    double small_ar_lat_s = 0.0;
    bool   small_pp_got   = false;
    bool   small_ar_got   = false;

    for(double mMiB_d : PAYLOADS_MIB){
      const int    mMiB  = (int)std::llround(mMiB_d);
      const size_t BYTES = MB(std::max(1.0, (double)mMiB));

      double pp_fwd_gbps=0.0, pp_small_lat_s=0.0;
      run_pp_1f1b_meas(rd, BYTES, PP_WARM, PP_STEPS, rd.boundary_left_idx, pp_fwd_gbps, pp_small_lat_s);
      double pp_eff = (theory>0.0 ? std::min(0.99, std::max(0.0, pp_fwd_gbps / theory)) : 0.0);
      pp_pairs.emplace_back(mMiB, pp_eff);
      small_pp_lat_s = pp_small_lat_s; small_pp_got = true;

      progress_with_cfg((double)(++idx)/(double)total_work, "PP", rd.size, mMiB, idx, total_work);

      double busbw=0.0, ar_small_lat_s=0.0;
      run_ar_collective(rd, BYTES, AR_ITERS, busbw, ar_small_lat_s);
      double ar_eff = (theory>0.0 ? std::min(0.99, std::max(0.0, busbw / theory)) : 0.0);
      ar_pairs.emplace_back(mMiB, ar_eff);
      small_ar_lat_s = ar_small_lat_s; small_ar_got = true;

      progress_with_cfg((double)(++idx)/(double)total_work, "AR", rd.size, mMiB, idx, total_work);
    }

    if(small_pp_got){
      const double small_bytes = 4.0 * 1024.0;
      const double small_bw_gbps = (small_bytes / small_pp_lat_s) / 1e9;
      double eff0 = (theory>0.0 ? std::min(0.99, std::max(0.0, small_bw_gbps / theory)) : 0.0);
      pp_pairs.emplace_back(0, eff0);
    }else{
      pp_pairs.emplace_back(0, 0.0);
    }

    if(small_ar_got){
      const int    W = rd.size;
      const size_t small_elems = std::max<size_t>(1, 1024/sizeof(float));
      const double used_small_bytes = (double)small_elems * sizeof(float);
      const double moved_small = 2.0 * (W - 1.0) / (double)W * used_small_bytes;
      const double small_bw_gbps = (moved_small / small_ar_lat_s) / 1e9;
      double eff0 = (theory>0.0 ? std::min(0.99, std::max(0.0, small_bw_gbps / theory)) : 0.0);
      ar_pairs.emplace_back(0, eff0);
    }else{
      ar_pairs.emplace_back(0, 0.0);
    }

    double lat_s = (small_pp_got ? small_pp_lat_s : 5e-6);
    double cpu = (rd.size==2?0.03:(rd.size==4?0.04:0.05));

    OutItem oi;
    oi.size=rd.size; oi.theory_bw=theory;
    oi.pp_eff_pairs=std::move(pp_pairs);
    oi.ar_eff_pairs=std::move(ar_pairs);
    oi.lat_s=lat_s; oi.must=false; oi.cpu=cpu;
    return oi;
  };

  const char* MF = getenv_cstr("MUST_FILL_POLICY", "SINGLE"); // SINGLE|ALL|NONE
  OutItem o2 = run_one(R2);
  OutItem o4 = (ndev>=4 ? run_one(R4) : OutItem{4,0,{}, {},5e-6,false,0.04});
  OutItem o8 = (ndev>=8 ? run_one(R8) : OutItem{8,0,{}, {},5e-6,false,0.05});

  if(std::strcmp(MF,"ALL")==0){ o2.must=o4.must=o8.must=true; }
  else if(std::strcmp(MF,"NONE")==0){ o2.must=o4.must=o8.must=false; }
  else{
    OutItem* best = &o2;
    if(o4.theory_bw > best->theory_bw || (o4.theory_bw==best->theory_bw && o4.size<best->size)) best = &o4;
    if(o8.theory_bw > best->theory_bw || (o8.theory_bw==best->theory_bw && o8.size<best->size)) best = &o8;
    o2.must = (&o2==best); o4.must = (&o4==best); o8.must = (&o8==best);
  }

  // ---------- Build JSON string once ----------
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss << "{\"networks\": [\n";

  auto print_pairs = [&](const std::vector<std::pair<int,double>>& v){
    oss << "[";
    for(size_t k=0;k<v.size();++k){
      oss << "[" << v[k].first << ", " << std::setprecision(4) << v[k].second << "]";
      if(k+1<v.size()) oss << ", ";
    }
    oss << "]";
    oss << std::setprecision(6); // reset precision for later numbers if needed
  };

  auto dump_item = [&](const OutItem& it, bool trailing_comma){
    oss << "  {\"bandwidth\": " << std::setprecision(2) << it.theory_bw
        << ", \"pp_efficiency\": ";
    print_pairs(it.pp_eff_pairs);
    oss << ", \"ar_efficiency\": ";
    print_pairs(it.ar_eff_pairs);
    oss << ", \"size\": " << it.size
        << ", \"latency\": " << std::scientific << it.lat_s << std::fixed
        << ", \"ops\": { \"p2p\":[1.0,null], \"reduce_scatter\":[1.0,-1], \"all_gather\":[1.0,-1], \"all_reduce\":[2.0,-1] }, "
        << "\"must_be_filled\": " << (it.must ? "true":"false")
        << ", \"processor_usage\": " << std::setprecision(2) << it.cpu << "}";
    if(trailing_comma) oss << ",";
    oss << "\n";
  };

  std::vector<OutItem> items;
  items.push_back(o2);
  if(ndev>=4) items.push_back(o4);
  if(ndev>=8) items.push_back(o8);

  for(size_t i=0;i<items.size();++i){
    dump_item(items[i], i+1<items.size());
  }
  oss << "]}";
  const std::string json = oss.str();

  // ---------- Print JSON to stdout ----------
  std::fwrite(json.data(), 1, json.size(), stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);

  // ---------- Save JSON to ./net_bw.json ----------
  std::string out_path = "./net_bw.json";

  std::ofstream f(out_path, std::ios::out | std::ios::trunc | std::ios::binary);
  if(f){
    f.write(json.data(), (std::streamsize)json.size());
    f.close();
  }else{
    std::fprintf(stderr, "Failed to write JSON\n");
  }

  return 0;
}
