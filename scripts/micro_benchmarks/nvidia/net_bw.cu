/**
 * net_bw.cu — single-node network/comms micro-benchmark (CUDA + optional NCCL)
 *
 * WHAT THIS PROGRAM MEASURES
 * --------------------------
 * Quickly characterizes “Megatron-like” communication on one CUDA node across 2/4/8 GPUs:
 *   • PP (pipeline-parallel) steady-state 1F1B traffic across a *boundary pair* of adjacent ranks.
 *     - Default: NCCL point-to-point (ncclSend/ncclRecv) if NCCL is enabled.
 *     - Fallback: cudaMemcpyPeer for one-way copies (NO_CCL or PP_BACKEND=p2p).
 *   • AR (data/tensor-parallel) AllReduce (ring) *bus bandwidth* via NCCL (if enabled).
 *   • A tiny-message one-way P2P latency using 4 KiB cudaMemcpyPeer.
 *
 * THEORY BANDWIDTH (AUTO)
 * -----------------------
 * The benchmark auto-detects an effective per-direction theory bandwidth by:
 *   1) Optional explicit override via P2P_THEORY_GBPS, or
 *   2) A short peer memcpy probe between the boundary pair and snapping to:
 *      {200.0, 100.0, 63.0, 31.5, 15.75, 7.88} GB/s
 *
 * OUTPUT (FILE)
 * -------------
 * JSON written to: ./net_bw.json
 *
 * Schema (abridged):
 * {
 *   "networks": [
 *     {
 *       "bandwidth": <theory_GBps>,         // per-direction normalization GB/s
 *       "pp_efficiency": [[MiB, eff], ...], // PP eff vs message size, plus tiny-msg anchor [0,eff0]
 *       "ar_efficiency": [[MiB, eff], ...], // AR eff vs message size, plus tiny-msg anchor [0,eff0]
 *       "size": <2|4|8>,                    // participating GPU count for this run
 *       "latency": <seconds>,               // 4 KiB one-way P2P latency in seconds
 *       "ops": { "p2p":[1.0,null], "reduce_scatter":[1.0,-1], "all_gather":[1.0,-1], "all_reduce":[2.0,-1] },
 *       "must_be_filled": <bool>,           // hint for performance-model fill policy
 *       "processor_usage": <0.03|0.04|0.05> // tiny heuristic per size
 *     }
 *   ]
 * }
 *
 * ENVIRONMENT (subset)
 * --------------------
 *  Device selection & boundary (per run size)
 *    SET2, SET4, SET8       Comma-separated device IDs forming the ordered set (default [0..N-1]).
 *    BOUND2, BOUND4, BOUND8 Two IDs “L,R” which must be adjacent in the set; this pair is the boundary.
 *    RUN_SIZES              Comma list of sizes to run (subset of {2,4,8}), default "4,8".
 *
 *  Payloads & loop counts
 *    LLM_PAYLOAD_MB         CSV MiB list for PP/AR (default: 128,96,64,32,16,8,4,2,1).
 *    PP_WARMUP              Warm-up steps for PP 1F1B (default: 20).
 *    PP_STEPS               Timed PP steps (default: 400).
 *    AR_ITERS               Timed AllReduce iterations (default: 200).
 *
 *  Normalization / theory bandwidth
 *    P2P_THEORY_GBPS        Hard override of theory GB/s (per direction).
 *
 *  PP backend
 *    PP_BACKEND             "nccl" or "p2p". Default "nccl" (if NCCL enabled), else "p2p".
 *
 *  Output control
 *    DISPLAY_OUTPUT=0|1     Default 1. If 0, suppress printing JSON to stdout.
 *
 * BUILD
 * -----
 *   With NCCL (recommended):
 *     nvcc -O3 -std=c++14 net_bw.cu -lnccl -o builds/net_bw
 *
 *   Customize your NCCL path:
 *     nvcc -O3 -std=c++14 net_bw.cu \
 *       -I/home/zonghang.li/.conda/envs/megatron-lm/lib/python3.12/site-packages/nvidia/nccl/include \
 *       -L/home/zonghang.li/.conda/envs/megatron-lm/lib/python3.12/site-packages/nvidia/nccl/lib \
 *       -lnccl \
 *       -o builds/net_bw
 *     export LD_LIBRARY_PATH=/home/zonghang.li/.conda/envs/megatron-lm/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
 *
 *   Without NCCL (PP uses cudaMemcpyPeer, AR skipped/fallback):
 *     nvcc -O3 -std=c++14 -DNO_CCL net_bw.cu -o builds/net_bw
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
#include <chrono>
#include <thread>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <numeric>

#ifndef NO_CCL
#include <nccl.h>
#endif

// ------------ env helpers ------------
static inline int getenv_int(const char* k, int defv){
  const char* s = std::getenv(k);
  return s ? std::atoi(s) : defv;
}
static inline double getenv_double(const char* k, double defv){
  const char* s = std::getenv(k);
  return s ? std::atof(s) : defv;
}
static inline const char* getenv_cstr(const char* k, const char* defv){
  const char* s = std::getenv(k);
  return s ? s : defv;
}

// ------------ time helpers -----------
static inline double now_ms(){
  using clk = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(
      clk::now().time_since_epoch()).count();
}

// ------------ error helpers ----------
static inline void ckCuda(cudaError_t e, const char* where){
  if(e != cudaSuccess){
    std::fprintf(stderr, "CUDA error %d (%s) at %s\n",
                 (int)e, cudaGetErrorString(e), where);
    std::exit(1);
  }
}

#ifndef NO_CCL
static inline void ckNccl(ncclResult_t e, const char* where){
  if(e != ncclSuccess){
    std::fprintf(stderr, "NCCL error %d (%s) at %s\n",
                 (int)e, ncclGetErrorString(e), where);
    std::exit(1);
  }
}
#endif

// ------------ misc helpers -----------
static inline size_t MB(double x){
  return (size_t)std::llround(x * 1024.0 * 1024.0);
}

static void progress_bar(double frac, const char* phase,
                         int run_size, int payload_mib,
                         size_t i, size_t total)
{
  frac = std::max(0.0, std::min(1.0, frac));
  const int barw = 40;
  int filled = (int)std::round(frac * barw);
  std::fprintf(stderr, "\r[");
  for(int j=0;j<barw;j++) std::fputc(j<filled?'#':'-', stderr);
  std::fprintf(stderr, "] %5.1f%%  | %s  size=%d  payload=%d MiB  (%zu/%zu)",
               100.0*frac, phase, run_size, payload_mib, i, total);
  if(frac>=1.0) std::fprintf(stderr, "  \n");
  std::fflush(stderr);
}

// ------------ payload list -----------
static std::vector<double> parse_mib_list_from_env(const char* key){
  const char* s = std::getenv(key);
  if(!s || !*s) return {128,96,64,32,16,8,4,2,1};
  std::vector<double> out;
  std::istringstream iss(s);
  std::string tok;
  while(std::getline(iss, tok, ',')){
    size_t a = tok.find_first_not_of(" \t");
    size_t b = tok.find_last_not_of(" \t");
    if(a==std::string::npos) continue;
    double v = std::atof(tok.substr(a, b-a+1).c_str());
    if(v>0.0) out.push_back(v);
  }
  if(out.empty()) out = {128,96,64,32,16,8,4,2,1};
  return out;
}

// ------------ run sizes --------------
static void parse_run_sizes(bool& want2, bool& want4, bool& want8){
  const char* s = getenv_cstr("RUN_SIZES","4,8");
  want2 = want4 = want8 = false;
  std::istringstream iss(s);
  std::string tok;
  while(std::getline(iss,tok,',')){
    int v = std::atoi(tok.c_str());
    if(v==2)      want2 = true;
    else if(v==4) want4 = true;
    else if(v==8) want8 = true;
  }
  if(!(want2||want4||want8)){
    want4 = true;
    want8 = true;
  }
}

// ------------ P2P enable + latency ---
static void enable_p2p(int a, int b){
  int canAB=0, canBA=0;
  ckCuda(cudaDeviceCanAccessPeer(&canAB,a,b), "canAB");
  ckCuda(cudaDeviceCanAccessPeer(&canBA,b,a), "canBA");
  if(canAB){
    ckCuda(cudaSetDevice(a), "set A");
    cudaDeviceEnablePeerAccess(b,0);
    cudaGetLastError();
  }
  if(canBA){
    ckCuda(cudaSetDevice(b), "set B");
    cudaDeviceEnablePeerAccess(a,0);
    cudaGetLastError();
  }
}

static double latency_us_pair_cuda(int dst, int src){
  const size_t bytes = 4*1024;
  const int reps = 4000;
  ckCuda(cudaSetDevice(dst), "lat dst dev");
  void* dDst = nullptr;
  ckCuda(cudaMalloc(&dDst, bytes), "lat dst");
  ckCuda(cudaSetDevice(src), "lat src dev");
  void* dSrc = nullptr;
  ckCuda(cudaMalloc(&dSrc, bytes), "lat src");
  ckCuda(cudaSetDevice(dst), "lat stream dev");
  cudaStream_t s;
  ckCuda(cudaStreamCreate(&s), "lat s");
  for(int i=0;i<50;i++)
    ckCuda(cudaMemcpyPeerAsync(dDst,dst,dSrc,src,bytes,s), "lat warm");
  ckCuda(cudaStreamSynchronize(s), "lat sync warm");

  double t0 = now_ms();
  for(int i=0;i<reps;i++)
    ckCuda(cudaMemcpyPeerAsync(dDst,dst,dSrc,src,bytes,s), "lat copy");
  ckCuda(cudaStreamSynchronize(s), "lat sync");
  double t1 = now_ms();

  cudaStreamDestroy(s);
  cudaFree(dDst);
  ckCuda(cudaSetDevice(src), "lat free src dev");
  cudaFree(dSrc);

  return (t1-t0)*1000.0 / reps;
}

// ------------ run description --------
struct RunDef{
  int size;
  std::vector<int> devs;
  int boundary_left_idx;
};

static std::vector<int> parse_devlist(const char* s){
  std::vector<int> v;
  if(!s || !*s) return v;
  std::istringstream iss(s);
  std::string tok;
  while(std::getline(iss,tok,',')){
    if(!tok.empty()) v.push_back(std::atoi(tok.c_str()));
  }
  return v;
}

static RunDef make_run_from_env(int want_size,
                                const char* env_set,
                                const char* env_bound_default)
{
  RunDef r{};
  r.size = want_size;
  std::vector<int> def(want_size);
  std::iota(def.begin(), def.end(), 0);

  std::vector<int> set = parse_devlist(std::getenv(env_set));
  if((int)set.size() != want_size) set = def;
  r.devs = set;

  const char* bkey =
      (want_size==2 ? "BOUND2" :
       (want_size==4 ? "BOUND4" : "BOUND8"));
  std::vector<int> bound = parse_devlist(
      getenv_cstr(bkey, env_bound_default));

  int bi = 0;
  if(bound.size()==2){
    for(int i=0;i<want_size-1;i++){
      if(r.devs[i]==bound[0] && r.devs[i+1]==bound[1]){
        bi = i;
        break;
      }
    }
  }
  r.boundary_left_idx = bi;
  return r;
}

// ------------ simple P2P probe -------
static double peer_copy_gbps_size(int dst, int src, size_t BYTES,
                                  int reps=40){
  ckCuda(cudaSetDevice(dst), "uni dst dev");
  void* dDst = nullptr;
  ckCuda(cudaMalloc(&dDst,BYTES),"uni dst");
  ckCuda(cudaSetDevice(src), "uni src dev");
  void* dSrc = nullptr;
  ckCuda(cudaMalloc(&dSrc,BYTES),"uni src");
  ckCuda(cudaSetDevice(dst), "uni stream dev");
  cudaStream_t s;
  ckCuda(cudaStreamCreate(&s),"uni stream");

  for(int w=0;w<6;w++)
    ckCuda(cudaMemcpyPeerAsync(dDst,dst,dSrc,src,BYTES,s),"uni warm");
  ckCuda(cudaStreamSynchronize(s),"uni warm sync");

  double t0 = now_ms();
  for(int i=0;i<reps;i++)
    ckCuda(cudaMemcpyPeerAsync(dDst,dst,dSrc,src,BYTES,s),"uni copy");
  ckCuda(cudaStreamSynchronize(s),"uni sync");
  double t1 = now_ms();

  cudaStreamDestroy(s);
  cudaFree(dDst);
  ckCuda(cudaSetDevice(src), "uni free src dev");
  cudaFree(dSrc);

  double ms = (t1-t0)/std::max(1,reps);
  return BYTES / (ms*1e9);
}

// ------------ theory bandwidth -------
static double detect_theory_bw(int devL, int devR){
  double forced = getenv_double("P2P_THEORY_GBPS", -1.0);
  if(forced > 0.0) return forced;

  const size_t BYTES = MB(getenv_double("DETECT_BYTES_MB", 128.0));
  const int    REPS  = getenv_int("DETECT_REPS", 30);

  double lr = peer_copy_gbps_size(devR,devL,BYTES,REPS);
  double rl = peer_copy_gbps_size(devL,devR,BYTES,REPS);
  double meas = std::max(lr, rl);

  const double cand[] = {200.0, 100.0, 63.0, 31.5, 15.75, 7.88};
  double best = cand[0];
  double bestd = std::fabs(meas - cand[0]);
  for(int i=1;i<6;i++){
    double d = std::fabs(meas - cand[i]);
    if(d < bestd){
      bestd = d;
      best = cand[i];
    }
  }
  if(meas < 5.0) best = 7.88;
  return best;
}

// ------------ PP: p2p backend --------
static void run_pp_p2p(const RunDef& rd,
                       size_t bytes,
                       int warmup_steps,
                       int timed_steps,
                       int boundary_left_rank,
                       double& out_forward_gbps,
                       double& out_small_lat_s)
{
  const int left  = rd.devs[boundary_left_rank];
  const int right = rd.devs[boundary_left_rank+1];

  ckCuda(cudaSetDevice(right),"pp dst dev");
  void* dDst=nullptr; ckCuda(cudaMalloc(&dDst,bytes),"pp dst");
  ckCuda(cudaSetDevice(left),"pp src dev");
  void* dSrc=nullptr; ckCuda(cudaMalloc(&dSrc,bytes),"pp src");
  ckCuda(cudaSetDevice(right),"pp stream dev");
  cudaStream_t s; ckCuda(cudaStreamCreate(&s),"pp stream");

  for(int i=0;i<std::max(1,warmup_steps/4);i++)
    ckCuda(cudaMemcpyPeerAsync(dDst,right,dSrc,left,bytes,s),"pp warm");
  ckCuda(cudaStreamSynchronize(s),"pp warm sync");

  cudaEvent_t e0,e1;
  ckCuda(cudaEventCreate(&e0),"pp e0");
  ckCuda(cudaEventCreate(&e1),"pp e1");
  ckCuda(cudaEventRecord(e0,s),"pp rec e0");
  for(int i=0;i<timed_steps;i++)
    ckCuda(cudaMemcpyPeerAsync(dDst,right,dSrc,left,bytes,s),"pp copy");
  ckCuda(cudaEventRecord(e1,s),"pp rec e1");
  ckCuda(cudaEventSynchronize(e1),"pp sync");
  float ms=0.f;
  ckCuda(cudaEventElapsedTime(&ms,e0,e1),"pp dt");
  cudaEventDestroy(e0); cudaEventDestroy(e1);

  out_forward_gbps = ((double)bytes/1e9) /
                     ((ms/std::max(1,timed_steps))/1000.0);
  out_small_lat_s  = latency_us_pair_cuda(right, left)/1e6;

  cudaStreamDestroy(s);
  cudaFree(dDst);
  ckCuda(cudaSetDevice(left),"pp free src dev");
  cudaFree(dSrc);
}

// ------------ PP: NCCL backend -------
#ifndef NO_CCL
struct PPRank {
  int dev;
  ncclComm_t comm;
  cudaStream_t stream;
  float *fwd_send,*fwd_recv,*bwd_send,*bwd_recv;
  size_t elems;
};

static void run_pp_nccl(const RunDef& rd,
                        size_t bytes,
                        int warmup_steps,
                        int timed_steps,
                        int boundary_left_rank,
                        double& out_forward_gbps,
                        double& out_small_lat_s)
{
  const int W    = rd.size;
  const int left = boundary_left_rank;
  const int right = left+1;

  ncclUniqueId uid;
  ckNccl(ncclGetUniqueId(&uid),"pp getuid");

  std::vector<PPRank> rr(W);
  for(int r=0;r<W;r++){
    rr[r].dev   = rd.devs[r];
    rr[r].elems = std::max<size_t>(1, bytes / sizeof(float));
  }

  ncclGroupStart();
  for(int r=0;r<W;r++){
    ckCuda(cudaSetDevice(rr[r].dev),"pp set dev");
    ckCuda(cudaStreamCreate(&rr[r].stream),"pp stream");
    ckCuda(cudaMalloc(&rr[r].fwd_send,rr[r].elems*sizeof(float)),"pp fs");
    ckCuda(cudaMalloc(&rr[r].fwd_recv,rr[r].elems*sizeof(float)),"pp fr");
    ckCuda(cudaMalloc(&rr[r].bwd_send,rr[r].elems*sizeof(float)),"pp bs");
    ckCuda(cudaMalloc(&rr[r].bwd_recv,rr[r].elems*sizeof(float)),"pp br");
    ckNccl(ncclCommInitRank(&rr[r].comm,W,uid,r),"pp comm");
  }
  ncclGroupEnd();

  for(int i=0;i<warmup_steps;i++){
    ncclGroupStart();
    for(int r=0;r<W;r++){
      ckCuda(cudaSetDevice(rr[r].dev),"pp warm set");
      if(r<W-1)
        ckNccl(ncclSend(rr[r].fwd_send,rr[r].elems,ncclFloat32,
                        r+1,rr[r].comm,rr[r].stream),"pp fwd send warm");
      if(r>0)
        ckNccl(ncclRecv(rr[r].fwd_recv,rr[r].elems,ncclFloat32,
                        r-1,rr[r].comm,rr[r].stream),"pp fwd recv warm");
      if(r>0)
        ckNccl(ncclSend(rr[r].bwd_send,rr[r].elems,ncclFloat32,
                        r-1,rr[r].comm,rr[r].stream),"pp bwd send warm");
      if(r<W-1)
        ckNccl(ncclRecv(rr[r].bwd_recv,rr[r].elems,ncclFloat32,
                        r+1,rr[r].comm,rr[r].stream),"pp bwd recv warm");
    }
    ncclGroupEnd();
  }
  for(int r=0;r<W;r++)
    ckCuda(cudaStreamSynchronize(rr[r].stream),"pp warm sync");

  cudaEvent_t e0=nullptr,e1=nullptr;
  ckCuda(cudaSetDevice(rr[left].dev),"pp evt dev");
  ckCuda(cudaEventCreate(&e0),"pp e0");
  ckCuda(cudaEventCreate(&e1),"pp e1");
  ckCuda(cudaEventRecord(e0,rr[left].stream),"pp rec e0");

  for(int it=0; it<timed_steps; ++it){
    ncclGroupStart();
    for(int r=0;r<W;r++){
      ckCuda(cudaSetDevice(rr[r].dev),"pp timed set");
      if(r<W-1)
        ckNccl(ncclSend(rr[r].fwd_send,rr[r].elems,ncclFloat32,
                        r+1,rr[r].comm,rr[r].stream),"pp fwd send");
      if(r>0)
        ckNccl(ncclRecv(rr[r].fwd_recv,rr[r].elems,ncclFloat32,
                        r-1,rr[r].comm,rr[r].stream),"pp fwd recv");
      if(r>0)
        ckNccl(ncclSend(rr[r].bwd_send,rr[r].elems,ncclFloat32,
                        r-1,rr[r].comm,rr[r].stream),"pp bwd send");
      if(r<W-1)
        ckNccl(ncclRecv(rr[r].bwd_recv,rr[r].elems,ncclFloat32,
                        r+1,rr[r].comm,rr[r].stream),"pp bwd recv");
    }
    ncclGroupEnd();
  }

  ckCuda(cudaEventRecord(e1,rr[left].stream),"pp rec e1");
  for(int r=0;r<W;r++)
    ckCuda(cudaStreamSynchronize(rr[r].stream),"pp sync");
  float ms=0.f;
  ckCuda(cudaEventElapsedTime(&ms,e0,e1),"pp dt");
  cudaEventDestroy(e0); cudaEventDestroy(e1);

  out_forward_gbps = ((double)bytes/1e9) /
                     ((ms/std::max(1,timed_steps))/1000.0);
  out_small_lat_s  = latency_us_pair_cuda(rr[right].dev, rr[left].dev)/1e6;

  for(int r=0;r<W;r++){
    ncclCommDestroy(rr[r].comm);
    cudaFree(rr[r].fwd_send);
    cudaFree(rr[r].fwd_recv);
    cudaFree(rr[r].bwd_send);
    cudaFree(rr[r].bwd_recv);
    cudaStreamDestroy(rr[r].stream);
  }
}
#endif

static void run_pp(const RunDef& rd,
                   size_t bytes,
                   int warmup_steps,
                   int timed_steps,
                   int boundary_left_rank,
                   double& out_forward_gbps,
                   double& out_small_lat_s)
{
  std::string backend = getenv_cstr("PP_BACKEND","auto");
#ifdef NO_CCL
  backend = "p2p";
#else
  if(backend=="auto") backend = "nccl";
#endif

#ifndef NO_CCL
  if(backend=="nccl"){
    run_pp_nccl(rd,bytes,warmup_steps,timed_steps,
                boundary_left_rank,out_forward_gbps,out_small_lat_s);
    return;
  }
#endif
  run_pp_p2p(rd,bytes,warmup_steps,timed_steps,
             boundary_left_rank,out_forward_gbps,out_small_lat_s);
}

// ------------ AR via NCCL ------------
#ifndef NO_CCL
struct ARRank{
  int dev;
  ncclComm_t comm;
  cudaStream_t stream;
  float* buf;
  size_t elems;
};

static void run_ar(const RunDef& rd,
                   size_t bytes,
                   int iters,
                   double& out_busbw_gbps,
                   double& out_small_lat_s)
{
  const int W = rd.size;
  size_t elems = std::max<size_t>(1, bytes/sizeof(float));

  ncclUniqueId uid;
  ckNccl(ncclGetUniqueId(&uid),"ar getuid");

  std::vector<ARRank> rr(W);
  for(int r=0;r<W;r++){
    rr[r].dev   = rd.devs[r];
    rr[r].elems = elems;
  }

  ncclGroupStart();
  for(int r=0;r<W;r++){
    ckCuda(cudaSetDevice(rr[r].dev),"ar set dev");
    ckCuda(cudaStreamCreate(&rr[r].stream),"ar stream");
    ckCuda(cudaMalloc(&rr[r].buf, rr[r].elems*sizeof(float)),"ar malloc");
    ckNccl(ncclCommInitRank(&rr[r].comm,W,uid,r),"ar comm");
  }
  ncclGroupEnd();

  // warmup
  for(int i=0;i<5;i++){
    ncclGroupStart();
    for(int r=0;r<W;r++){
      ckCuda(cudaSetDevice(rr[r].dev),"ar warm set");
      ckNccl(ncclAllReduce(rr[r].buf, rr[r].buf, rr[r].elems,
                           ncclFloat32, ncclSum,
                           rr[r].comm, rr[r].stream),"ar warm");
    }
    ncclGroupEnd();
  }
  for(int r=0;r<W;r++)
    ckCuda(cudaStreamSynchronize(rr[r].stream),"ar warm sync");

  cudaEvent_t e0,e1;
  ckCuda(cudaSetDevice(rr[0].dev),"ar evt dev");
  ckCuda(cudaEventCreate(&e0),"ar e0");
  ckCuda(cudaEventCreate(&e1),"ar e1");
  ckCuda(cudaEventRecord(e0,rr[0].stream),"ar rec e0");

  for(int it=0; it<iters; ++it){
    ncclGroupStart();
    for(int r=0;r<W;r++){
      ckCuda(cudaSetDevice(rr[r].dev),"ar timed set");
      ckNccl(ncclAllReduce(rr[r].buf, rr[r].buf, rr[r].elems,
                           ncclFloat32, ncclSum,
                           rr[r].comm, rr[r].stream),"ar iter");
    }
    ncclGroupEnd();
  }

  ckCuda(cudaEventRecord(e1,rr[0].stream),"ar rec e1");
  for(int r=0;r<W;r++)
    ckCuda(cudaStreamSynchronize(rr[r].stream),"ar sync");
  float ms=0.f;
  ckCuda(cudaEventElapsedTime(&ms,e0,e1),"ar dt");
  cudaEventDestroy(e0); cudaEventDestroy(e1);

  const double used_bytes=(double)(elems*sizeof(float));
  const double moved = 2.0*(W-1.0)/(double)W * used_bytes;
  const double ms_per_iter = ms/std::max(1,iters);
  out_busbw_gbps = moved/(ms_per_iter/1000.0)/1e9;

  // tiny AR latency (~1 KiB)
  size_t small_elems = std::max<size_t>(1,1024/sizeof(float));
  cudaEvent_t l0,l1;
  ckCuda(cudaSetDevice(rr[0].dev),"ar tiny dev");
  ckCuda(cudaEventCreate(&l0),"l0");
  ckCuda(cudaEventCreate(&l1),"l1");
  ckCuda(cudaEventRecord(l0,rr[0].stream),"l0rec");
  ncclGroupStart();
  for(int r=0;r<W;r++){
    ckCuda(cudaSetDevice(rr[r].dev),"ar tiny set");
    ckNccl(ncclAllReduce(rr[r].buf, rr[r].buf, small_elems,
                         ncclFloat32, ncclSum,
                         rr[r].comm, rr[r].stream),"lat ar");
  }
  ncclGroupEnd();
  ckCuda(cudaEventRecord(l1,rr[0].stream),"l1rec");
  for(int r=0;r<W;r++)
    ckCuda(cudaStreamSynchronize(rr[r].stream),"ar tiny sync");
  float lms=0.f;
  ckCuda(cudaEventElapsedTime(&lms,l0,l1),"lat el");
  cudaEventDestroy(l0); cudaEventDestroy(l1);
  out_small_lat_s = lms/1000.0;

  for(int r=0;r<W;r++){
    ncclCommDestroy(rr[r].comm);
    cudaFree(rr[r].buf);
    cudaStreamDestroy(rr[r].stream);
  }
}
#else
static void run_ar(const RunDef& rd,
                   size_t bytes,
                   int iters,
                   double& out_busbw_gbps,
                   double& out_small_lat_s)
{
  (void)iters;
  // Fallback: treat AR as P2P bus BW between boundary pair
  const int devL = rd.devs[rd.boundary_left_idx];
  const int devR = rd.devs[rd.boundary_left_idx+1];
  double g1 = peer_copy_gbps_size(devR,devL,bytes);
  double g2 = peer_copy_gbps_size(devL,devR,bytes);
  out_busbw_gbps = std::max(g1,g2);
  out_small_lat_s = latency_us_pair_cuda(devR,devL)/1e6;
}
#endif

// ------------ env util ---------------
static void set_env_if_empty(const char* k,const char* v){
#if defined(_WIN32)
  const char* cur = std::getenv(k);
  if(!cur || !*cur) _putenv_s(k,v);
#else
  const char* cur = std::getenv(k);
  if(!cur || !*cur) setenv(k,v,1);
#endif
}

static std::string compat_env_name(const char* suffix){
  std::string s;
  s.reserve(16);
  s += "NCCL_";
  s += suffix;
  return s;
}

// ------------ main -------------------
int main(){
  // Pin deterministic NCCL config by default
  if(getenv_int("PIN_COLL",1)!=0){
#ifndef NO_CCL
    auto set_compat=[&](const char* k,const char* v){
#if defined(_WIN32)
      _putenv_s(compat_env_name(k).c_str(), v);
#else
      setenv(compat_env_name(k).c_str(), v, 1);
#endif
    };
    set_compat("ALGO","Ring");
    set_compat("PROTO","Simple");
    set_compat("MIN_NRINGS","1");
    set_compat("MAX_NRINGS","1");
    set_compat("BLOCKING_WAIT","1");
    set_compat("ASYNC_ERROR_HANDLING","1");
    set_compat("COLLNET_ENABLE","0");
#endif
  }

  set_env_if_empty("CONTEND_AR","1");
  set_env_if_empty("CONTEND_PP","1");

  int ndev = 0;
  ckCuda(cudaGetDeviceCount(&ndev),"cudaGetDeviceCount");
  if(ndev < 2){
    std::printf("{\"networks\": []}\n");
    return 0;
  }

  RunDef R2 = make_run_from_env(2,"SET2","0,1");
  RunDef R4 = make_run_from_env(4,"SET4","1,2");
  RunDef R8 = make_run_from_env(8,"SET8","3,4");

  // enable P2P between all visible pairs
  for(int i=0;i<ndev;i++)
    for(int j=0;j<ndev;j++)
      if(i!=j) enable_p2p(i,j);

  const std::vector<double> PAYLOADS_MIB =
      parse_mib_list_from_env("LLM_PAYLOAD_MB");
  const int PP_WARM  = getenv_int("PP_WARMUP",20);
  const int PP_STEPS = getenv_int("PP_STEPS",400);
  const int AR_ITERS = getenv_int("AR_ITERS",200);

  bool want2=false, want4=false, want8=false;
  parse_run_sizes(want2,want4,want8);
  int run_count=0;
  if(want2) run_count++;
  if(want4 && ndev>=4) run_count++;
  if(want8 && ndev>=8) run_count++;

  const size_t total_work =
      (size_t)run_count * PAYLOADS_MIB.size() * 2; // PP+AR
  size_t idx = 0;

  struct OutItem{
    int size;
    double theory_bw;
    std::vector<std::pair<int,double>> pp_eff_pairs;
    std::vector<std::pair<int,double>> ar_eff_pairs;
    double lat_s;
    bool must;
    double cpu;
  };

  auto run_one = [&](const RunDef& rd)->OutItem{
    const int bi   = rd.boundary_left_idx;
    const int devL = rd.devs[bi];
    const int devR = rd.devs[bi+1];

    double theory = detect_theory_bw(devL, devR);

    std::vector<std::pair<int,double>> pp_pairs, ar_pairs;
    double small_pp_lat_s=0.0, small_ar_lat_s=0.0;
    bool small_pp_got=false, small_ar_got=false;

    for(double mMiBd : PAYLOADS_MIB){
      const int mMiB = (int)std::llround(mMiBd);
      const size_t BYTES = MB(std::max(1.0,(double)mMiB));

      // PP
      double pp_gbps=0.0, pp_lat=0.0;
      run_pp(rd,BYTES,PP_WARM,PP_STEPS,
             rd.boundary_left_idx,
             pp_gbps,pp_lat);
      double pp_eff = (theory>0.0)
        ? std::min(0.99, std::max(0.0, pp_gbps/theory))
        : 0.0;
      pp_pairs.emplace_back(mMiB, pp_eff);
      small_pp_lat_s = pp_lat;
      small_pp_got   = true;

      idx += 1;
      progress_bar((double)idx/(double)total_work,
                   "PP",rd.size,mMiB,idx,total_work);

      // AR
      double busbw=0.0, ar_lat=0.0;
      run_ar(rd,BYTES,AR_ITERS,busbw,ar_lat);
      double ar_eff = (theory>0.0)
        ? std::min(0.99, std::max(0.0, busbw/theory))
        : 0.0;
      ar_pairs.emplace_back(mMiB, ar_eff);
      small_ar_lat_s = ar_lat;
      small_ar_got   = true;

      idx += 1;
      progress_bar((double)idx/(double)total_work,
                   "AR",rd.size,mMiB,idx,total_work);
    }

    // tiny-message anchors (label 0)
    if(small_pp_got){
      const double small_bytes = 4.0*1024.0;
      const double small_bw =
          (small_bytes / small_pp_lat_s) / 1e9;
      pp_pairs.emplace_back(
          0,
          (theory>0.0)
            ? std::min(0.99, std::max(0.0, small_bw/theory))
            : 0.0);
    }else{
      pp_pairs.emplace_back(0,0.0);
    }

    if(small_ar_got){
      const int W = rd.size;
      const size_t small_elems =
          std::max<size_t>(1,1024/sizeof(float));
      const double used_small_bytes =
          (double)small_elems*sizeof(float);
      const double moved_small =
          2.0*(W-1.0)/(double)W * used_small_bytes;
      const double small_bw =
          (moved_small / small_ar_lat_s) / 1e9;
      ar_pairs.emplace_back(
          0,
          (theory>0.0)
            ? std::min(0.99, std::max(0.0, small_bw/theory))
            : 0.0);
    }else{
      ar_pairs.emplace_back(0,0.0);
    }

    double lat_s = (small_pp_got ? small_pp_lat_s : 5e-6);
    double cpu   = (rd.size==2?0.03:(rd.size==4?0.04:0.05));

    OutItem oi{
      rd.size,
      theory,
      std::move(pp_pairs),
      std::move(ar_pairs),
      lat_s,
      false,
      cpu
    };
    return oi;
  };

  const char* MF = getenv_cstr("MUST_FILL_POLICY","SINGLE");

  OutItem o2{2,0,{}, {},5e-6,false,0.03};
  OutItem o4{4,0,{}, {},5e-6,false,0.04};
  OutItem o8{8,0,{}, {},5e-6,false,0.05};

  if(want2)             o2 = run_one(R2);
  if(want4 && ndev>=4)  o4 = run_one(R4);
  if(want8 && ndev>=8)  o8 = run_one(R8);

  if(std::strcmp(MF,"ALL")==0){
    o2.must = o4.must = o8.must = true;
  }else if(std::strcmp(MF,"NONE")==0){
    o2.must = o4.must = o8.must = false;
  }else{
    OutItem* best = &o2;
    if(o4.theory_bw > best->theory_bw ||
       (o4.theory_bw==best->theory_bw && o4.size<best->size))
      best = &o4;
    if(o8.theory_bw > best->theory_bw ||
       (o8.theory_bw==best->theory_bw && o8.size<best->size))
      best = &o8;
    o2.must = (&o2 == best);
    o4.must = (&o4 == best);
    o8.must = (&o8 == best);
  }

  // JSON
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss << "{\"networks\": [\n";

  auto print_pairs = [&](const std::vector<std::pair<int,double>>& v){
    oss << "[";
    for(size_t k=0;k<v.size();++k){
      oss << "[" << v[k].first << ", "
          << std::setprecision(4) << v[k].second << "]";
      if(k+1<v.size()) oss << ", ";
    }
    oss << "]";
    oss << std::setprecision(6);
  };

  auto dump = [&](const OutItem& it,bool comma){
    oss << "  {\"bandwidth\": " << std::setprecision(2)
        << it.theory_bw
        << ", \"pp_efficiency\": ";
    print_pairs(it.pp_eff_pairs);
    oss << ", \"ar_efficiency\": ";
    print_pairs(it.ar_eff_pairs);
    oss << ", \"size\": " << it.size
        << ", \"latency\": " << std::scientific
        << it.lat_s << std::fixed
        << ", \"ops\": { \"p2p\":[1.0,null], "
           "\"reduce_scatter\":[1.0,-1], "
           "\"all_gather\":[1.0,-1], "
           "\"all_reduce\":[2.0,-1] }, "
        << "\"must_be_filled\": "
        << (it.must ? "true":"false")
        << ", \"processor_usage\": "
        << std::setprecision(2) << it.cpu << "}";
    if(comma) oss << ",";
    oss << "\n";
  };

  std::vector<OutItem> items;
  if(want2)             items.push_back(o2);
  if(want4 && ndev>=4)  items.push_back(o4);
  if(want8 && ndev>=8)  items.push_back(o8);

  for(size_t i=0;i<items.size();++i)
    dump(items[i], i+1<items.size());
  oss << "]}";

  std::string json = oss.str();
  if (getenv_int("DISPLAY_OUTPUT", 1) != 0) {
    std::fwrite(json.data(), 1, json.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
  }

  std::ofstream f("./net_bw.json",
                  std::ios::out | std::ios::trunc | std::ios::binary);
  if (f) {
    f.write(json.data(), (std::streamsize)json.size());
    f.close();
  }
  return 0;
}
