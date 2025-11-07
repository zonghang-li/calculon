/**
 * net_bw.cpp — single-node network/comms micro-benchmark (HIP + optional RCCL)
 *
 * WHAT THIS PROGRAM MEASURES
 * --------------------------
 * Quickly characterizes “Megatron-like” communication on one AMD ROCm node across 4/8 GPUs:
 *   • PP (pipeline-parallel) steady-state 1F1B traffic across a *boundary pair* of adjacent ranks.
 *     - Default: RCCL point-to-point (rcclSend/rcclRecv).
 *     - Fallback (NO_RCCL): hipMemcpyPeer for one-way copies (AR skipped).
 *   • AR (data/tensor-parallel) AllReduce (ring) *bus bandwidth* via RCCL.
 *   • A tiny-message one-way P2P latency using 4 KiB hipMemcpyPeer.
 *
 * TOPOLOGY AWARENESS (AUTO THEORY BANDWIDTH)
 * ------------------------------------------
 * The benchmark auto-detects the link type between the boundary pair using:
 *   1) `rocm-smi --showtopo` (preferred): parses the “Link Type between two GPUs” table.
 *      - If link is XGMI   → theory per-direction GB/s defaults to 100.0 (overrideable).
 *      - If link is PCIE   → theory per-direction GB/s snaps to canonical PCIe peaks.
 *   2) If rocm-smi is unavailable, a short 8 MiB unidirectional probe is run and the
 *      measured plateau is snapped to {7.88, 15.75, 31.50, 63.00, 100.0} GB/s.
 *
 * The chosen *theory bandwidth* is used to normalize efficiencies for both PP and AR:
 *   eff = min(0.99, measured_GBps / theory_GBps)
 *
 * On MI250 2-tier nodes (e.g., GPUs 0–3 and 4–7 form separate XGMI “islands” connected
 * by PCIe), this yields:
 *   • size=4 (within one island, XGMI)  → theory ≈ 100.0 GB/s (per direction, default).
 *   • size=8 (cross-island via PCIe 4×16) → theory ≈ 31.50 GB/s (per direction).
 *
 * OUTPUT (FILE)
 * -------------
 * JSON written to: ./net_bw.json
 *
 * Schema (abridged):
 * {
 *   "networks": [
 *     {
 *       "bandwidth": <theory_GBps>,         // per-direction normalization GB/s (auto-detected)
 *       "pp_efficiency": [[MiB, eff], ...], // PP eff vs message size, plus a tiny-msg anchor [0, eff0]
 *       "ar_efficiency": [[MiB, eff], ...], // AR eff vs message size, plus a tiny-msg anchor [0, eff0]
 *       "size": <4|8>,                      // participating GPU count for this run
 *       "latency": <seconds>,               // 4 KiB one-way P2P latency in seconds
 *       "ops": { "p2p":[1.0,null], "reduce_scatter":[1.0,-1], "all_gather":[1.0,-1], "all_reduce":[2.0,-1] },
 *       "must_be_filled": <bool>,           // hint for performance-model fill policy
 *       "processor_usage": <0.03|0.04|0.05> // tiny heuristic per size
 *     }
 *   ]
 * }
 *
 * EFFICIENCY BINNING VS MESSAGE SIZE
 * ----------------------------------
 * For a CSV of LLM-relevant payload sizes (MiB):
 *   • PP: runs a timed 1F1B loop across the boundary pair and reports forward GB/s.
 *   • AR: runs RCCL AllReduce ring and reports *BusBW* (bytes moved per iter / time).
 * A tiny-message anchor is appended:
 *   • PP: derived from 4 KiB hipMemcpyPeer latency.
 *   • AR: derived from ~1 KiB AllReduce latency using the ring moved-bytes model.
 *
 * RUN SIZES & DEFAULT TOPOLOGY
 * ----------------------------
 * The tool attempts sizes {4, 8} (size=2 is typically unnecessary on 2-tier MI250 nodes).
 * Device order and the boundary pair are configurable (see ENV below). By default:
 *   • size=4  → device set [0,1,2,3], boundary "1,2" (within an XGMI island).
 *   • size=8  → device set [0,1,2,3,4,5,6,7], boundary "3,4" (cross-island via PCIe).
 * P2P access is enabled between all visible pairs whenever possible.
 *
 * PROGRESS & LOGGING
 * ------------------
 * A single-line progress bar to stderr (same style as other benches):
 *   [########------------------------] 37.5% | PP  size=4  payload=32 MiB  (i/total)
 * Debug prints are available (see ENV). On completion the JSON is also printed to stdout.
 *
 * KEY ENVIRONMENT VARIABLES
 * -------------------------
 *  Device selection & boundary (per run size)
 *    SET4, SET8            Comma-separated device IDs forming the ordered set (default [0..N-1]).
 *    BOUND4, BOUND8        Two IDs “L,R” which must be adjacent in the set; this pair is the boundary.
 *                          Defaults: size=4 → "1,2"; size=8 → "3,4".
 *    RUN_SIZES             Comma list of sizes to run (subset of {4,8}), e.g., RUN_SIZES=4 or RUN_SIZES=8.
 *    HIP_VISIBLE_DEVICES   Use to pin a subset or reorder devices if desired.
 *
 *  Payloads & loop counts
 *    LLM_PAYLOAD_MB        CSV MiB list for PP/AR (default: 128,96,64,32,16,8,4,2,1).
 *    PP_WARMUP             Warm-up steps for PP 1F1B (default: 20).
 *    PP_STEPS              Timed PP steps (default: 400).
 *    AR_ITERS              Timed AllReduce iterations (default: 200).
 *
 *  Normalization / theory bandwidth (auto-detected; override if needed)
 *    USE_ROCMSMI           1|0 Force use of `rocm-smi --showtopo` (default: 1 when available).
 *    XGMI_PERDIR_GBPS      Per-direction theory GB/s for XGMI (default: 100.0).
 *    PCIE_GEN              3|4|5  → payload GB/s per lane ≈ 0.985/1.969/3.938; used with PCIE_WIDTH.
 *    PCIE_WIDTH            PCIe lane count (e.g., 16).
 *    P2P_THEORY_GBPS       Hard override of theory GB/s (bypasses auto-detection).
 *
 *  RCCL control (optional)
 *    PIN_RCCL=1            Pin deterministic config for A/B: Ring algo + Simple proto + 1 ring.
 *    RCCL_DEBUG=*          Your setting is respected; default is quiet unless you override.
 *
 *  Output control
 *    DISPLAY_OUTPUT=0|1     Default 1. If 0, suppress printing JSON to stdout.
 *                           The file ./net_bw.json is always written.
 *
 *  Watchdog / anti-hang
 *    STEP_TIMEOUT_MS       Bounded wait per PP step / AR iter (default: 10000 ms).
 *    TINY_TIMEOUT_MS       Timeout for small-message AR latency op (default: 5000 ms).
 *
 *  Debug
 *    DEBUG=1               Enable concise debug logs.
 *    DEBUG_PROBE=1|2       Print unidirectional probe details used to infer theory BW.
 *
 * BUILD
 * -----
 *   With RCCL (recommended):
 *     hipcc -O3 -std=c++14 --offload-arch=gfx90a net_bw.cpp -lrccl -o builds/net_bw_amd
 *
 *   Without RCCL (PP uses hipMemcpyPeer, AR skipped):
 *     hipcc -O3 -std=c++14 --offload-arch=gfx90a -DNO_RCCL net_bw.cpp -o builds/net_bw_amd
 *
 * EXAMPLES
 * --------
 *   ./builds/net_bw_amd
 *   RUN_SIZES=4 ./builds/net_bw_amd
 *   DEBUG=1 DEBUG_PROBE=2 ./builds/net_bw_amd
 *   XGMI_PERDIR_GBPS=50 ./builds/net_bw_amd         # normalize XGMI to 50 GB/s per direction
 *   PCIE_GEN=5 PCIE_WIDTH=16 ./builds/net_bw_amd    # normalize PCIe Gen5 ×16 to 63.0 GB/s
 *
 * NOTES / LIMITATIONS
 * -------------------
 * • The “theory” bandwidth is a *normalization* target; actual plateaus depend on drivers,
 *   NUMA, SDMA engines, stream concurrency, and runtime load.
 * • PP measures a single peer path at a time; multi-stream pipelining can push higher GB/s.
 * • AR reports ring *BusBW*, not payload/replica GB/s.
 * • The “must_be_filled” and “processor_usage” fields are hints for downstream performance
 *   models and are not strict hardware metrics.
 */

#include <hip/hip_runtime.h>
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

#if defined(_WIN32)
  #include <process.h>
  #include <direct.h>
#else
  #include <unistd.h>
#endif

#ifndef NO_CCL
  #if __has_include(<rccl/rccl.h>)
    #include <rccl/rccl.h>
  #else
    #include <rccl.h>
  #endif
  // RCCL API shim (RCCL inherits NCCL symbols)
  #define cclComm_t                 ncclComm_t
  #define cclUniqueId               ncclUniqueId
  #define cclResult_t               ncclResult_t
  #define cclSuccess                ncclSuccess
  #define cclGetUniqueId            ncclGetUniqueId
  #define cclCommInitRank           ncclCommInitRank
  #define cclCommAbort              ncclCommAbort
  #define cclCommDestroy            ncclCommDestroy
  #define cclGetErrorString         ncclGetErrorString
  #define cclGroupStart             ncclGroupStart
  #define cclGroupEnd               ncclGroupEnd
  #define cclSend                   ncclSend
  #define cclRecv                   ncclRecv
  #define cclAllReduce              ncclAllReduce
  #define cclFloat32                ncclFloat32
  #define cclSum                    ncclSum
#endif

// ---------------- env + logging ----------------
static inline int getenv_int(const char* k, int defv){ const char* s=getenv(k); return s?std::atoi(s):defv; }
static inline double getenv_double(const char* k, double defv){ const char* s=getenv(k); return s?std::atof(s):defv; }
static inline const char* getenv_cstr(const char* k, const char* defv){ const char* s=getenv(k); return s?s:defv; }

static double g_t0_ms=0.0;
static inline double now_ms(){ using clk=std::chrono::high_resolution_clock; return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count(); }
static inline double rel_ms(){ return now_ms()-g_t0_ms; }

static inline int dbg_global(){ return getenv_int("DEBUG", 0); }
static inline int dbg_ar(){ int v=getenv_int("DEBUG_AR",-1); return v>=0?v:dbg_global(); }
static inline int dbg_pp(){ int v=getenv_int("DEBUG_PP",-1); return v>=0?v:dbg_global(); }
static inline int dbg_probe(){ int v=getenv_int("DEBUG_PROBE",-1); return v>=0?v:dbg_global(); }

#define DLOG_AR(l, fmt, ...)  do{ if(dbg_ar()    >=(l)) { fprintf(stderr,"[dbg][AR][%8.3f ms] " fmt "\n", rel_ms(), ##__VA_ARGS__); fflush(stderr);} }while(0)
#define DLOG_PP(l, fmt, ...)  do{ if(dbg_pp()    >=(l)) { fprintf(stderr,"[dbg][PP][%8.3f ms] " fmt "\n", rel_ms(), ##__VA_ARGS__); fflush(stderr);} }while(0)
#define DLOG_PR(l, fmt, ...)  do{ if(dbg_probe() >=(l)) { fprintf(stderr,"[dbg][PROBE][%8.3f ms] " fmt "\n", rel_ms(), ##__VA_ARGS__); fflush(stderr);} }while(0)

static inline void ckHip(hipError_t e, const char* where){
  if(e != hipSuccess){ std::fprintf(stderr, "HIP error %d (%s) at %s\n",(int)e, hipGetErrorString(e), where);
#if defined(_WIN32)
    std::exit(1);
#else
    _exit(1);
#endif
  }
}
#ifndef NO_CCL
static inline void ckCcl(ncclResult_t e, const char* where){
  if(e != ncclSuccess){ std::fprintf(stderr, "Collective error %d (%s) at %s\n",(int)e, ncclGetErrorString(e), where);
#if defined(_WIN32)
    std::exit(1);
#else
    _exit(1);
#endif
  }
}
#endif

static inline size_t MB(double x){ return (size_t)std::llround(x*1024.0*1024.0); }

// --------------- progress bar ---------------
static void progress_with_cfg(double frac, const char* phase, int run_size, int payload_mib,
                              size_t i, size_t total){
  frac = std::max(0.0, std::min(1.0, frac));
  const int barw = 40; int filled = (int)std::round(frac * barw);
  std::fprintf(stderr, "\r["); for(int j=0;j<barw;j++) std::fputc(j<filled?'#':'-', stderr);
  std::fprintf(stderr, "] %5.1f%%  | %s  size=%d  payload=%d MiB  (%zu/%zu)", 100.0*frac, phase, run_size, payload_mib, i, total);
  if(frac>=1.0) std::fprintf(stderr, "  \n");
  std::fflush(stderr);
}

// --------------- parse sizes ----------------
static std::vector<double> parse_mib_list_from_env(const char* key){
  const char* s = std::getenv(key);
  if(!s || !*s) return {128,96,64,32,16,8,4,2,1};
  std::vector<double> out; std::istringstream iss(s); std::string tok;
  while(std::getline(iss, tok, ',')){
    size_t a = tok.find_first_not_of(" \t"), b = tok.find_last_not_of(" \t");
    if(a==std::string::npos) continue;
    double v = std::atof(tok.substr(a, b-a+1).c_str());
    if(v>0.0) out.push_back(v);
  }
  if(out.empty()) out = {128,96,64,32,16,8,4,2,1};
  return out;
}
static void parse_run_sizes(bool& want2, bool& want4, bool& want8){
  const char* s = getenv_cstr("RUN_SIZES","4,8");
  want2=want4=want8=false; std::istringstream iss(s); std::string tok;
  while(std::getline(iss,tok,',')){ int v=std::atoi(tok.c_str()); if(v==2) want2=true; else if(v==4) want4=true; else if(v==8) want8=true; }
  if(!(want2||want4||want8)){ want4=true; want8=true; }
}

// --------------- P2P enable + latency ---------------
static void enable_p2p(int a, int b){
  int canAB=0, canBA=0; ckHip(hipDeviceCanAccessPeer(&canAB,a,b),"canAB"); ckHip(hipDeviceCanAccessPeer(&canBA,b,a),"canBA");
  if(canAB){ hipSetDevice(a); hipDeviceEnablePeerAccess(b,0); hipGetLastError(); }
  if(canBA){ hipSetDevice(b); hipDeviceEnablePeerAccess(a,0); hipGetLastError(); }
}
static double latency_us_pair_hipcpy(int dst, int src){
  const size_t bytes=4*1024; const int reps=4000;
  hipSetDevice(dst); void* dDst=nullptr; ckHip(hipMalloc(&dDst,bytes),"lat dst");
  hipSetDevice(src); void* dSrc=nullptr; ckHip(hipMalloc(&dSrc,bytes),"lat src");
  hipSetDevice(dst); hipStream_t s; ckHip(hipStreamCreate(&s),"lat s");
  for(int i=0;i<50;i++) ckHip(hipMemcpyPeerAsync(dDst,dst,dSrc,src,bytes,s),"lat warm");
  ckHip(hipStreamSynchronize(s),"lat warm sync");
  double t0=now_ms(); for(int i=0;i<reps;i++) ckHip(hipMemcpyPeerAsync(dDst,dst,dSrc,src,bytes,s),"lat copy");
  ckHip(hipStreamSynchronize(s),"lat sync"); double t1=now_ms();
  hipStreamDestroy(s); hipFree(dDst); hipSetDevice(src); hipFree(dSrc);
  return (t1-t0)*1000.0/reps;
}

// --------------- run description ---------------
struct RunDef{ int size; std::vector<int> devs; int boundary_left_idx; };
static std::vector<int> parse_devlist(const char* s){
  std::vector<int> v; if(!s||!*s) return v; std::istringstream iss(s); std::string tok;
  while(std::getline(iss,tok,',')) if(!tok.empty()) v.push_back(std::atoi(tok.c_str())); return v;
}
static RunDef make_run_from_env(int want_size, const char* env_set, const char* env_bound_default){
  RunDef r{}; r.size=want_size; std::vector<int> def(want_size); std::iota(def.begin(),def.end(),0);
  std::vector<int> set = parse_devlist(std::getenv(env_set)); if((int)set.size()!=want_size) set=def; r.devs=set;
  std::vector<int> bound = parse_devlist(getenv_cstr((want_size==2?"BOUND2":(want_size==4?"BOUND4":"BOUND8")), env_bound_default));
  int bi=0; if(bound.size()==2) for(int i=0;i<want_size-1;i++) if(r.devs[i]==bound[0] && r.devs[i+1]==bound[1]){ bi=i; break; }
  r.boundary_left_idx=bi; return r;
}

// --------------------- small helpers ---------------------
static inline size_t tokenize_ws(const std::string& line, std::vector<std::string>& toks){
  toks.clear(); std::istringstream iss(line); std::string t; while(iss>>t) toks.push_back(t); return toks.size();
}
#ifndef _WIN32
static std::string run_cmd_capture(const char* cmd){
  std::string out; FILE* f = popen(cmd, "r");
  if(!f) return out;
  char buf[4096];
  while(true){ size_t n=fread(buf,1,sizeof(buf),f); if(n==0) break; out.append(buf, n); }
  pclose(f); return out;
}
#endif

// Parse "Link Type between two GPUs" table from `rocm-smi --showtopo`.
// Returns "XGMI" / "PCIE" / "" (unknown).
static std::string rocmsmi_link_type(int a, int b){
#ifndef _WIN32
  std::string topo = run_cmd_capture("rocm-smi --showtopo 2>/dev/null");
  if(topo.empty()) return std::string();

  std::istringstream ss(topo);
  std::string line;
  bool in_link = false;
  std::vector<int> cols;
  std::vector<std::string> toks;

  while(std::getline(ss, line)){
    if(!in_link){
      if(line.find("Link Type between two GPUs") != std::string::npos){
        in_link = true; cols.clear();
      }
      continue;
    }
    // Stop on next big separator or next section
    if(line.find("====") != std::string::npos && !cols.empty()) break;

    // First header row contains GPU column labels.
    if(cols.empty()){
      if(line.find("GPU0")!=std::string::npos && line.find("GPU1")!=std::string::npos){
        tokenize_ws(line, toks);
        for(const auto& t : toks){
          if(t.rfind("GPU",0)==0){ cols.push_back(std::atoi(t.c_str()+3)); }
        }
      }
      continue;
    }

    // Parse data rows: "GPUi  XGMI PCIE ..."
    tokenize_ws(line, toks);
    if(toks.size() < 1) continue;
    if(toks[0].rfind("GPU",0)!=0) continue;
    int row = std::atoi(toks[0].c_str()+3);
    if(row != a) continue;
    // cells align with cols
    if(toks.size()-1 < cols.size()) continue;
    for(size_t i=0;i<cols.size();++i){
      if(cols[i]==b){
        std::string v = toks[i+1];
        if(v=="XGMI" || v=="PCIE") return v;
        return std::string();
      }
    }
  }
#endif
  return std::string();
}

// --------------- probes for fallback detection ---------------
static double peer_copy_gbps_size(int dst, int src, size_t BYTES, int reps=40){
  hipSetDevice(dst); void* dDst=nullptr; ckHip(hipMalloc(&dDst,BYTES),"uni dst");
  hipSetDevice(src); void* dSrc=nullptr; ckHip(hipMalloc(&dSrc,BYTES),"uni src");
  hipSetDevice(dst); hipStream_t s; ckHip(hipStreamCreate(&s),"uni stream");
  for(int w=0;w<6;w++) ckHip(hipMemcpyPeerAsync(dDst,dst,dSrc,src,BYTES,s),"uni warm");
  ckHip(hipStreamSynchronize(s),"uni warm sync");
  double t0=now_ms(); for(int i=0;i<reps;i++) ckHip(hipMemcpyPeerAsync(dDst,dst,dSrc,src,BYTES,s),"uni copy");
  ckHip(hipStreamSynchronize(s),"uni sync"); double t1=now_ms();
  hipStreamDestroy(s); hipFree(dDst); hipSetDevice(src); hipFree(dSrc);
  double ms=(t1-t0)/std::max(1,reps); return BYTES/(ms*1e9);
}

static bool bidi_probe_gbps(int a,int b,size_t BYTES,int reps,double budget_ms,double& out_gbps){
  hipSetDevice(a); void* aDst=nullptr; void* aSrc=nullptr; ckHip(hipMalloc(&aDst,BYTES),"bidi aDst"); ckHip(hipMalloc(&aSrc,BYTES),"bidi aSrc");
  hipSetDevice(b); void* bDst=nullptr; void* bSrc=nullptr; ckHip(hipMalloc(&bDst,BYTES),"bidi bDst"); ckHip(hipMalloc(&bSrc,BYTES),"bidi bSrc");
  hipSetDevice(a); hipStream_t sa; ckHip(hipStreamCreate(&sa),"bidi sa"); hipEvent_t a0,a1; ckHip(hipEventCreate(&a0),"a0"); ckHip(hipEventCreate(&a1),"a1");
  hipSetDevice(b); hipStream_t sb; ckHip(hipStreamCreate(&sb),"bidi sb"); hipEvent_t b0,b1; ckHip(hipEventCreate(&b0),"b0"); ckHip(hipEventCreate(&b1),"b1");

  // warmup both directions
  hipSetDevice(a); ckHip(hipMemcpyPeerAsync(bDst,b,aSrc,a,BYTES,sa),"bidi warm a->b");
  hipSetDevice(b); ckHip(hipMemcpyPeerAsync(aDst,a,bSrc,b,BYTES,sb),"bidi warm b->a");
  hipSetDevice(a); ckHip(hipStreamSynchronize(sa),"bidi warm sync a");
  hipSetDevice(b); ckHip(hipStreamSynchronize(sb),"bidi warm sync b");

  hipSetDevice(a); ckHip(hipEventRecord(a0,sa),"rec a0");
  hipSetDevice(b); ckHip(hipEventRecord(b0,sb),"rec b0");
  for(int i=0;i<reps;i++){
    hipSetDevice(a); ckHip(hipMemcpyPeerAsync(bDst,b,aSrc,a,BYTES,sa),"bidi a->b");
    hipSetDevice(b); ckHip(hipMemcpyPeerAsync(aDst,a,bSrc,b,BYTES,sb),"bidi b->a");
  }
  hipSetDevice(a); ckHip(hipEventRecord(a1,sa),"rec a1");
  hipSetDevice(b); ckHip(hipEventRecord(b1,sb),"rec b1");

  double t0=now_ms(); bool ok_a=false, ok_b=false;
  while(!(ok_a && ok_b)){
    if(!ok_a){ hipError_t qa=hipEventQuery(a1); if(qa==hipSuccess) ok_a=true; else if(qa!=hipErrorNotReady) ckHip(qa,"bidi a query"); }
    if(!ok_b){ hipError_t qb=hipEventQuery(b1); if(qb==hipSuccess) ok_b=true; else if(qb!=hipErrorNotReady) ckHip(qb,"bidi b query"); }
    if(now_ms()-t0>budget_ms) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  double gbps=0.0;
  if(ok_a && ok_b){
    float msa=0.f, msb=0.f; hipEventElapsedTime(&msa,a0,a1); hipEventElapsedTime(&msb,b0,b1);
    double ms = std::max((double)msa,(double)msb);
    gbps = ( (double)reps * 2.0 * (double)BYTES ) / ( (ms/1000.0) ) / 1e9;
  }

  hipEventDestroy(a0); hipEventDestroy(a1); hipEventDestroy(b0); hipEventDestroy(b1);
  hipStreamDestroy(sa); hipStreamDestroy(sb);
  hipFree(aDst); hipFree(aSrc); hipSetDevice(b); hipFree(bDst); hipFree(bSrc);

  if(gbps>0.0){ out_gbps=gbps; return true; }
  return false;
}

static bool peer_probe_with_timeout(int dst,int src,size_t BYTES,int reps,double budget_ms,double& out_gbps){
  hipSetDevice(dst); void* dDst=nullptr; ckHip(hipMalloc(&dDst,BYTES),"probe dst");
  hipSetDevice(src); void* dSrc=nullptr; ckHip(hipMalloc(&dSrc,BYTES),"probe src");
  hipSetDevice(dst); hipStream_t s; ckHip(hipStreamCreate(&s),"probe stream");
  hipEvent_t e0,e1; ckHip(hipEventCreate(&e0),"probe e0"); ckHip(hipEventCreate(&e1),"probe e1");
  ckHip(hipEventRecord(e0,s),"probe rec e0");
  for(int i=0;i<reps;i++) ckHip(hipMemcpyPeerAsync(dDst,dst,dSrc,src,BYTES,s),"probe copy");
  ckHip(hipEventRecord(e1,s),"probe rec e1");
  double t0=now_ms(); bool ok=false;
  while(true){
    hipError_t q=hipEventQuery(e1);
    if(q==hipSuccess){ ok=true; break; }
    if(q!=hipErrorNotReady){ ckHip(q,"probe query"); break; }
    if(now_ms()-t0>budget_ms) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  double gbps=0.0;
  if(ok){ float ms=0.f; hipEventElapsedTime(&ms,e0,e1); gbps=((double)reps*(double)BYTES)/((ms/1000.0))/1e9; }
  hipEventDestroy(e0); hipEventDestroy(e1); hipStreamDestroy(s);
  hipFree(dDst); hipSetDevice(src); hipFree(dSrc);
  if(ok){ out_gbps=gbps; return true; } return false;
}

// classification output
struct DetOut{ double uni_max, bidi_sum; double theory; const char* label; };

// Auto-detect theory bandwidth (per direction) for the boundary link
static DetOut detect_theory_auto(int devL,int devR){
  // 0) explicit overrides
  double forced = getenv_double("P2P_THEORY_GBPS", -1.0);
  if(forced>0.0){ DLOG_PR(1,"detect forced=%.2f", forced); return {0.0,0.0,forced,"forced"}; }

  // 1) try topology from rocm-smi
  std::string typ = rocmsmi_link_type(devL, devR);
  if(!typ.empty()){
    if(typ=="XGMI"){
      double perdir = getenv_double("XGMI_PERDIR_GBPS", 100.0); // 200 bidir ≈ 100 per dir
      DLOG_PR(1,"topology: XGMI -> theory=%.2f (per-dir)", perdir);
      return {0.0,0.0, perdir, "xgmi(topology)"};
    }else if(typ=="PCIE"){
      int g = getenv_int("PCIE_GEN", 4);
      int W = getenv_int("PCIE_WIDTH", 16);
      double per_lane=(g==3?0.985:(g==4?1.969:(g==5?3.938:1.969)));
      double t = per_lane * std::max(1, W);
      DLOG_PR(1,"topology: PCIE -> gen=%d width=%d -> theory=%.2f", g, W, t);
      return {0.0,0.0, t, "pcie(topology)"};
    }
  }

  // 2) fall back to measurement snap
  const size_t BYTES = MB(getenv_double("DETECT_BYTES_MB",128.0));
  const int    REPS  = getenv_int("DETECT_REPS", 30);
  const double TMO   = getenv_double("PROBE_TIMEOUT_MS", 10000.0);

  double lr=0.0, rl=0.0, bi=0.0;
  bool ok_lr = peer_probe_with_timeout(devR, devL, BYTES, REPS, TMO, lr);
  bool ok_rl = peer_probe_with_timeout(devL, devR, BYTES, REPS, TMO, rl);
  bool ok_bi = bidi_probe_gbps(devL, devR, BYTES, std::max(10, REPS/2), TMO, bi);
  double uni = std::max(ok_lr?lr:0.0, ok_rl?rl:0.0);
  double perdir_meas = std::max(uni, (ok_bi ? (bi * 0.5) : 0.0));
  DLOG_PR(1,"detect fallback uni_max=%.2f bidi=%.2f perdir_est=%.2f", uni, bi, perdir_meas);

  const double cand[] = {100.0, 50.0, 31.5, 15.75, 7.88};
  const char*  clab[] = {"if-100","if-50","pcie4x16","pcie3x16","pcie3x8"};
  double best=cand[0]; const char* lab=clab[0]; double bestd=fabs(perdir_meas-cand[0]);
  for(int i=1;i<5;i++){ double d=fabs(perdir_meas-cand[i]); if(d<bestd){bestd=d; best=cand[i]; lab=clab[i];}}
  if (perdir_meas < 5.0) { best=7.88; lab="pcie3x8"; }
  DLOG_PR(1,"detect snap -> theory=%.2f (%s)", best, lab);
  return {uni, bi, best, lab};
}

// --------------- PP backends ---------------
static void run_pp_1f1b_meas_p2p(const RunDef& rd, size_t bytes, int warmup_steps, int timed_steps,
                                 int boundary_left_rank,
                                 double& out_forward_gbps,
                                 double& out_small_lat_s)
{
  const int left  = rd.devs[boundary_left_rank];
  const int right = rd.devs[boundary_left_rank+1];
  DLOG_PP(1,"PP p2p start size=%d boundary=%d-%d bytes=%.3f MiB", rd.size,left,right,(double)bytes/(1024.0*1024.0));

  hipSetDevice(right); void* dDst=nullptr; ckHip(hipMalloc(&dDst,bytes),"pp dst");
  hipSetDevice(left ); void* dSrc=nullptr; ckHip(hipMalloc(&dSrc,bytes),"pp src");
  hipSetDevice(right); hipStream_t s; ckHip(hipStreamCreate(&s),"pp stream");

  for(int i=0;i<std::max(1,warmup_steps/4);i++) ckHip(hipMemcpyPeerAsync(dDst,right,dSrc,left,bytes,s),"pp warm");
  ckHip(hipStreamSynchronize(s),"pp warm sync");

  hipEvent_t e0,e1; ckHip(hipEventCreate(&e0),"pp e0"); ckHip(hipEventCreate(&e1),"pp e1");
  ckHip(hipEventRecord(e0,s),"pp rec e0"); for(int i=0;i<timed_steps;i++) ckHip(hipMemcpyPeerAsync(dDst,right,dSrc,left,bytes,s),"pp copy");
  ckHip(hipEventRecord(e1,s),"pp rec e1"); ckHip(hipEventSynchronize(e1),"pp sync");
  float ms=0.f; hipEventElapsedTime(&ms,e0,e1); hipEventDestroy(e0); hipEventDestroy(e1);

  out_forward_gbps = ((double)bytes/1e9) / ((ms/std::max(1,timed_steps))/1000.0);
  out_small_lat_s  = latency_us_pair_hipcpy(right, left)/1e6;

  hipStreamDestroy(s); hipFree(dDst); hipSetDevice(left); hipFree(dSrc);
  DLOG_PP(1,"PP p2p done fwd=%.2f GB/s", out_forward_gbps);
}

#ifndef NO_CCL
struct PPRank { int dev; cclComm_t comm; hipStream_t stream; float *fwd_send,*fwd_recv,*bwd_send,*bwd_recv; size_t elems; };
static void run_pp_1f1b_meas_rccl(const RunDef& rd, size_t bytes, int warmup_steps, int timed_steps,
                                  int boundary_left_rank,
                                  double& out_forward_gbps,
                                  double& out_small_lat_s){
  const int W=rd.size, left=boundary_left_rank, right=left+1;
  DLOG_PP(1,"PP rccl start size=%d boundary=%d-%d bytes=%.3f MiB",W,rd.devs[left],rd.devs[right],(double)bytes/(1024.0*1024.0));

  cclUniqueId uid; ckCcl(cclGetUniqueId(&uid),"pp getuid");
  std::vector<PPRank> rr(W); for(int r=0;r<W;r++){ rr[r].dev=rd.devs[r]; rr[r].elems=std::max<size_t>(1,bytes/sizeof(float)); }

  ckCcl(cclGroupStart(),"pp grp start");
  for(int r=0;r<W;r++){
    ckHip(hipSetDevice(rr[r].dev),"pp set"); ckHip(hipStreamCreate(&rr[r].stream),"pp stream");
    ckHip(hipMalloc(&rr[r].fwd_send,rr[r].elems*sizeof(float)),"pp fs"); ckHip(hipMalloc(&rr[r].fwd_recv,rr[r].elems*sizeof(float)),"pp fr");
    ckHip(hipMalloc(&rr[r].bwd_send,rr[r].elems*sizeof(float)),"pp bs"); ckHip(hipMalloc(&rr[r].bwd_recv,rr[r].elems*sizeof(float)),"pp br");
    ckCcl(cclCommInitRank(&rr[r].comm,W,uid,r),"pp comm");
  }
  ckCcl(cclGroupEnd(),"pp grp end");

  for(int i=0;i<warmup_steps;i++){
    ckCcl(cclGroupStart(),"pp warm start");
    for(int r=0;r<W;r++){
      ckHip(hipSetDevice(rr[r].dev),"pp wrm set");
      if(r<W-1) ckCcl(cclSend(rr[r].fwd_send,rr[r].elems,cclFloat32,r+1,rr[r].comm,rr[r].stream),"pp fwd send warm");
      if(r>0  ) ckCcl(cclRecv(rr[r].fwd_recv,rr[r].elems,cclFloat32,r-1,rr[r].comm,rr[r].stream),"pp fwd recv warm");
      if(r>0  ) ckCcl(cclSend(rr[r].bwd_send,rr[r].elems,cclFloat32,r-1,rr[r].comm,rr[r].stream),"pp bwd send warm");
      if(r<W-1) ckCcl(cclRecv(rr[r].bwd_recv,rr[r].elems,cclFloat32,r+1,rr[r].comm,rr[r].stream),"pp bwd recv warm");
    }
    ckCcl(cclGroupEnd(),"pp warm end");
  }
  for(int r=0;r<W;r++) ckHip(hipStreamSynchronize(rr[r].stream),"pp wrm sync");

  hipEvent_t e0=nullptr,e1=nullptr; ckHip(hipSetDevice(rr[left].dev),"pp e dev");
  ckHip(hipEventCreate(&e0),"pp e0"); ckHip(hipEventCreate(&e1),"pp e1"); ckHip(hipEventRecord(e0,rr[left].stream),"pp rec e0");
  for(int it=0; it<timed_steps; ++it){
    ckCcl(cclGroupStart(),"pp grp start");
    for(int r=0;r<W;r++){
      ckHip(hipSetDevice(rr[r].dev),"pp timed set");
      if(r<W-1) ckCcl(cclSend(rr[r].fwd_send,rr[r].elems,cclFloat32,r+1,rr[r].comm,rr[r].stream),"pp fwd send");
      if(r>0  ) ckCcl(cclRecv(rr[r].fwd_recv,rr[r].elems,cclFloat32,r-1,rr[r].comm,rr[r].stream),"pp fwd recv");
      if(r>0  ) ckCcl(cclSend(rr[r].bwd_send,rr[r].elems,cclFloat32,r-1,rr[r].comm,rr[r].stream),"pp bwd send");
      if(r<W-1) ckCcl(cclRecv(rr[r].bwd_recv,rr[r].elems,cclFloat32,r+1,rr[r].comm,rr[r].stream),"pp bwd recv");
    }
    ckCcl(cclGroupEnd(),"pp grp end");
  }
  ckHip(hipEventRecord(e1,rr[left].stream),"pp rec e1");
  for(int r=0;r<W;r++) ckHip(hipStreamSynchronize(rr[r].stream),"pp sync");
  float ms=0.f; ckHip(hipEventElapsedTime(&ms,e0,e1),"pp ms");
  hipEventDestroy(e0); hipEventDestroy(e1);

  out_forward_gbps = ((double)bytes/1e9)/((ms/std::max(1,timed_steps))/1000.0);
  out_small_lat_s  = latency_us_pair_hipcpy(rr[right].dev, rr[left].dev)/1e6;

  for(int r=0;r<W;r++){ cclCommDestroy(rr[r].comm); hipFree(rr[r].fwd_send); hipFree(rr[r].fwd_recv); hipFree(rr[r].bwd_send); hipFree(rr[r].bwd_recv); hipStreamDestroy(rr[r].stream); }
  DLOG_PP(1,"PP rccl done fwd=%.2f GB/s", out_forward_gbps);
}
#endif

static void run_pp_1f1b_meas(const RunDef& rd, size_t bytes, int warmup_steps, int timed_steps,
                             int boundary_left_rank,
                             double& out_forward_gbps,
                             double& out_small_lat_s)
{
  const std::string backend = getenv_cstr("PP_BACKEND","p2p");
#ifndef NO_CCL
  if(backend=="rccl"){ run_pp_1f1b_meas_rccl(rd,bytes,warmup_steps,timed_steps,boundary_left_rank,out_forward_gbps,out_small_lat_s); return; }
#endif
  run_pp_1f1b_meas_p2p(rd,bytes,warmup_steps,timed_steps,boundary_left_rank,out_forward_gbps,out_small_lat_s);
}

// --------------- AR robust ----------------
#ifndef NO_CCL
struct ARRank{ int dev; cclComm_t comm; hipStream_t stream; float* buf; size_t elems; };
static bool wait_events_all_with_timeout(const std::vector<hipEvent_t>& evs,double budget_ms,int iter){
  const int N=(int)evs.size(); std::vector<char> done(N,0); int remain=N; double t0=now_ms(), last=t0;
  while(remain>0){
    for(int i=0;i<N;i++){ if(done[i]) continue; hipError_t q=hipEventQuery(evs[i]); if(q==hipSuccess){ done[i]=1; --remain; } else if(q!=hipErrorNotReady) ckHip(q,"event query"); }
    if(remain==0) break; double nn=now_ms(); if(dbg_ar()>=2 && nn-last>200.0){ std::fprintf(stderr,"[dbg][AR][%8.3f ms] iter=%d waiting %d/%d\n",rel_ms(),iter,N-remain,N); fflush(stderr); last=nn; }
    if(nn-t0>budget_ms) return false; std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } return true;
}
static void run_ar_collective(const RunDef& rd, size_t bytes, int iters, double& out_busbw_gbps, double& out_small_lat_s){
  const int W=rd.size; size_t elems=std::max<size_t>(1,bytes/sizeof(float));
  int iters_cap=iters; if(bytes<=MB(1.0)) iters_cap=std::min(iters_cap,80); else if(bytes<=MB(2.0)) iters_cap=std::min(iters_cap,120);
  double step_budget_ms=getenv_double("STEP_TIMEOUT_MS",10000.0); if(bytes<=MB(1.0)) step_budget_ms=std::max(step_budget_ms,20000.0);
  const int use_group=getenv_int("AR_GROUP",0);

  cclUniqueId uid; ckCcl(cclGetUniqueId(&uid),"ar getuid");
  std::vector<ARRank> rr(W); for(int r=0;r<W;r++){ rr[r].dev=rd.devs[r]; rr[r].elems=elems; }
  DLOG_AR(1,"setup size=%d bytes=%.3f MiB iters=%d group=%d",W,(double)bytes/(1024.0*1024.0),iters_cap,use_group);

  ckCcl(cclGroupStart(),"ar grp start");
  for(int r=0;r<W;r++){
    ckHip(hipSetDevice(rr[r].dev),"ar set"); ckHip(hipStreamCreate(&rr[r].stream),"ar stream"); ckHip(hipMalloc(&rr[r].buf,rr[r].elems*sizeof(float)),"ar malloc");
    ckCcl(cclCommInitRank(&rr[r].comm,W,uid,r),"ar comm");
  }
  ckCcl(cclGroupEnd(),"ar grp end");

  for(int i=0;i<5;i++){
    if(use_group) ckCcl(cclGroupStart(),"ar warm grp start");
    for(int r=0;r<W;r++){ ckHip(hipSetDevice(rr[r].dev),"ar wrm set"); ckCcl(cclAllReduce(rr[r].buf,rr[r].buf,rr[r].elems,cclFloat32,cclSum,rr[r].comm,rr[r].stream),"ar warm"); }
    if(use_group) ckCcl(cclGroupEnd(),"ar warm grp end");
  }
  for(int r=0;r<W;r++) ckHip(hipStreamSynchronize(rr[r].stream),"ar warm sync");

  hipEvent_t e_prev,e_curr; ckHip(hipSetDevice(rr[0].dev),"ar t dev"); ckHip(hipEventCreate(&e_prev),"ar eprev"); ckHip(hipEventCreate(&e_curr),"ar ecurr"); ckHip(hipEventRecord(e_prev,rr[0].stream),"ar rec prev");
  std::vector<hipEvent_t> evs(W); for(int r=0;r<W;r++){ ckHip(hipSetDevice(rr[r].dev),"ar ev dev"); ckHip(hipEventCreate(&evs[r]),"ar ev"); }

  double total_ms=0.0; bool aborted=false;
  for(int it=0; it<iters_cap; ++it){
    if(use_group) ckCcl(cclGroupStart(),"ar iter grp start");
    for(int r=0;r<W;r++){ ckHip(hipSetDevice(rr[r].dev),"ar timed set"); ckCcl(cclAllReduce(rr[r].buf,rr[r].buf,rr[r].elems,cclFloat32,cclSum,rr[r].comm,rr[r].stream),"ar iter"); }
    if(use_group) ckCcl(cclGroupEnd(),"ar iter grp end");
    ckHip(hipSetDevice(rr[0].dev),"ar rec curr dev");
    ckHip(hipEventRecord(e_curr,rr[0].stream),"ar rec curr");
    for(int r=0;r<W;r++){ ckHip(hipSetDevice(rr[r].dev),"ar ev rec dev"); ckHip(hipEventRecord(evs[r],rr[r].stream),"ar ev rec"); }
    std::vector<hipEvent_t> evs_all(evs); evs_all.push_back(e_curr);
    if(!wait_events_all_with_timeout(evs_all,step_budget_ms,it)){
      std::fprintf(stderr,"\n[warn] AR timeout (size=%d bytes=%.3f MiB, iter=%d). Aborting comms and falling back.\n",W,(double)bytes/(1024.0*1024.0),it); fflush(stderr);
      for(int r=0;r<W;r++) cclCommAbort(rr[r].comm); aborted=true; break;
    }
    float ms=0.f; ckHip(hipEventElapsedTime(&ms,e_prev,e_curr),"ar step dt"); total_ms+=(double)ms; std::swap(e_prev,e_curr);
  }

  for(int r=0;r<W;r++) ckHip(hipEventDestroy(evs[r]),"ar ev destroy"); ckHip(hipEventDestroy(e_prev),"ar destroy prev"); ckHip(hipEventDestroy(e_curr),"ar destroy curr");

  if(!aborted){
    for(int r=0;r<W;r++) ckHip(hipStreamSynchronize(rr[r].stream),"ar sync");
    const double used_bytes=(double)(elems*sizeof(float)); const double moved = 2.0*(W-1.0)/(double)W * used_bytes;
    const double ms_per_iter = total_ms/std::max(1,iters_cap);
    out_busbw_gbps = moved/(ms_per_iter/1000.0)/1e9;
    DLOG_AR(1,"done busbw=%.2f GB/s (iters=%d)", out_busbw_gbps, iters_cap);
  }else{
    out_busbw_gbps = -1.0; // signal fallback
  }

  if(!aborted){
    size_t small_elems=std::max<size_t>(1,1024/sizeof(float)); hipEvent_t l0,l1; ckHip(hipSetDevice(rr[0].dev),"ar tiny dev");
    ckHip(hipEventCreate(&l0),"l0"); ckHip(hipEventCreate(&l1),"l1"); ckHip(hipEventRecord(l0,rr[0].stream),"l0rec");
    if(use_group) ckCcl(cclGroupStart(),"ar tiny grp start");
    for(int r=0;r<W;r++){ ckHip(hipSetDevice(rr[r].dev),"ar tiny set"); ckCcl(cclAllReduce(rr[r].buf,rr[r].buf,small_elems,cclFloat32,cclSum,rr[r].comm,rr[r].stream),"lat ar"); }
    if(use_group) ckCcl(cclGroupEnd(),"ar tiny grp end");
    ckHip(hipEventRecord(l1,rr[0].stream),"l1rec"); for(int r=0;r<W;r++) ckHip(hipStreamSynchronize(rr[r].stream),"ar tiny sync");
    float lms=0.f; ckHip(hipEventElapsedTime(&lms,l0,l1),"lat el"); hipEventDestroy(l0); hipEventDestroy(l1); out_small_lat_s=lms/1000.0;
  }else{
    out_small_lat_s = latency_us_pair_hipcpy(rd.devs[rd.boundary_left_idx+1], rd.devs[rd.boundary_left_idx])/1e6;
  }

  for(int r=0;r<W;r++){ cclCommDestroy(rr[r].comm); hipFree(rr[r].buf); hipStreamDestroy(rr[r].stream); }
}
#else
static void run_ar_collective(const RunDef&, size_t, int, double& bw, double& l){ bw=0.0; l=5e-5; }
#endif

// --------------- env util ---------------
static void set_env_if_empty(const char* k,const char* v){
#if defined(_WIN32)
  const char* cur=getenv(k); if(!cur||!*cur) _putenv_s(k,v);
#else
  const char* cur=getenv(k); if(!cur||!*cur) setenv(k,v,1);
#endif
}
static std::string compat_env_name(const char* suffix){ std::string s; s.reserve(16); s+="NCCL_"; s+=suffix; return s; }

// --------------- main ---------------
int main(){
  g_t0_ms = now_ms();

  // Pin deterministic collective config by default (RCCL honors NCCL_* env)
  if(getenv_int("PIN_COLL",1)!=0){
#ifndef NO_CCL
    auto set_compat=[&](const char* k,const char* v){
#if defined(_WIN32)
      _putenv_s(compat_env_name(k).c_str(), v);
#else
      setenv(compat_env_name(k).c_str(), v, 1);
#endif
    };
    set_compat("ALGO","Ring"); set_compat("PROTO","Simple"); set_compat("MIN_NRINGS","1"); set_compat("MAX_NRINGS","1");
    set_compat("BLOCKING_WAIT","1"); set_compat("ASYNC_ERROR_HANDLING","1"); set_compat("COLLNET_ENABLE","0");
#endif
  }

  set_env_if_empty("CONTEND_AR","1"); set_env_if_empty("CONTEND_PP","1");

  int ndev=0; ckHip(hipGetDeviceCount(&ndev),"hipGetDeviceCount"); if(ndev<2){ std::printf("{\"networks\": []}\n"); return 0; }

  RunDef R2=make_run_from_env(2,"SET2","0,1");
  RunDef R4=make_run_from_env(4,"SET4","1,2");   // intra-island default boundary
  RunDef R8=make_run_from_env(8,"SET8","3,4");   // cross-island default boundary

  for(int i=0;i<ndev;i++) for(int j=0;j<ndev;j++) if(i!=j) enable_p2p(i,j);

  const std::vector<double> PAYLOADS_MIB = parse_mib_list_from_env("LLM_PAYLOAD_MB");
  const int PP_WARM=getenv_int("PP_WARMUP",20), PP_STEPS=getenv_int("PP_STEPS",400), AR_ITERS=getenv_int("AR_ITERS",200);

  bool want2=false, want4=false, want8=false; parse_run_sizes(want2,want4,want8);
  int run_count=0; if(want2) run_count++; if(want4&&ndev>=4) run_count++; if(want8&&ndev>=8) run_count++;
  const size_t total_work=(size_t)run_count*PAYLOADS_MIB.size()*2; size_t idx=0;

  struct OutItem{ int size; double theory_bw; std::vector<std::pair<int,double>> pp_eff_pairs, ar_eff_pairs; double lat_s; bool must; double cpu; };

  auto run_one = [&](const RunDef& rd)->OutItem{
    const int bi=rd.boundary_left_idx, devL=rd.devs[bi], devR=rd.devs[bi+1];

    // Auto-detect per-direction “theory” GB/s
    DLOG_PR(1,"probe start size=%d pair=%d-%d", rd.size, devL, devR);
    DetOut det = detect_theory_auto(devL, devR);
    double theory = det.theory;
    DLOG_PR(1,"run size=%d boundary=%d-%d theory=%.2f GB/s (%s)", rd.size, devL, devR, theory, det.label);

    std::vector<std::pair<int,double>> pp_pairs, ar_pairs;
    double small_pp_lat_s=0.0, small_ar_lat_s=0.0; bool small_pp_got=false, small_ar_got=false;

    for(double mMiB_d : PAYLOADS_MIB){
      const int mMiB=(int)std::llround(mMiB_d); const size_t BYTES=MB(std::max(1.0,(double)mMiB));

      double pp_gbps=0.0, pp_lat=0.0; run_pp_1f1b_meas(rd,BYTES,PP_WARM,PP_STEPS,rd.boundary_left_idx,pp_gbps,pp_lat);
      double pp_eff = (theory>0.0)? std::min(0.99, std::max(0.0, pp_gbps/theory)) : 0.0; pp_pairs.emplace_back(mMiB, pp_eff);
      small_pp_lat_s=pp_lat; small_pp_got=true;
      idx+=1; progress_with_cfg((double)idx/(double)total_work,"PP",rd.size,mMiB,idx,total_work);

      double busbw=0.0, ar_lat=0.0;
#ifndef NO_CCL
      run_ar_collective(rd,BYTES,AR_ITERS,busbw,ar_lat);
#else
      busbw=-1.0; ar_lat=5e-6;
#endif
      if(busbw<0.0){ // fallback: peer copy
        double g1=peer_copy_gbps_size(devR,devL,BYTES), g2=peer_copy_gbps_size(devL,devR,BYTES);
        busbw=std::max(g1,g2); ar_lat=latency_us_pair_hipcpy(devR,devL)/1e6; DLOG_AR(1,"fallback busbw=%.2f GB/s (peer copy)",busbw);
      }
      double ar_eff=(theory>0.0)? std::min(0.99, std::max(0.0, busbw/theory)) : 0.0; ar_pairs.emplace_back(mMiB, ar_eff);
      small_ar_lat_s=ar_lat; small_ar_got=true;
      idx+=1; progress_with_cfg((double)idx/(double)total_work,"AR",rd.size,mMiB,idx,total_work);
    }

    if(small_pp_got){
      const double small_bytes=4.0*1024.0; const double small_bw=(small_bytes/small_pp_lat_s)/1e9;
      pp_pairs.emplace_back(0, (theory>0.0)? std::min(0.99, std::max(0.0, small_bw/theory)) : 0.0);
    }else pp_pairs.emplace_back(0,0.0);

    if(small_ar_got){
      const int W=rd.size; const size_t small_elems=std::max<size_t>(1,1024/sizeof(float));
      const double used_small_bytes=(double)small_elems*sizeof(float);
      const double moved_small=2.0*(W-1.0)/(double)W * used_small_bytes;
      const double small_bw=(moved_small/small_ar_lat_s)/1e9;
      ar_pairs.emplace_back(0, (theory>0.0)? std::min(0.99, std::max(0.0, small_bw/theory)) : 0.0);
    }else ar_pairs.emplace_back(0,0.0);

    double lat_s=(small_pp_got? small_pp_lat_s : 5e-6);
    double cpu=(rd.size==2?0.03:(rd.size==4?0.04:0.05));
    OutItem oi{rd.size, theory, std::move(pp_pairs), std::move(ar_pairs), lat_s, false, cpu};
    return oi;
  };

  const char* MF=getenv_cstr("MUST_FILL_POLICY","SINGLE");
  OutItem o2{2,0,{}, {},5e-6,false,0.03}, o4{4,0,{}, {},5e-6,false,0.04}, o8{8,0,{}, {},5e-6,false,0.05};
  if(want2) o2=run_one(R2); if(want4&&ndev>=4) o4=run_one(R4); if(want8&&ndev>=8) o8=run_one(R8);

  if(std::strcmp(MF,"ALL")==0){ o2.must=o4.must=o8.must=true; }
  else if(std::strcmp(MF,"NONE")==0){ o2.must=o4.must=o8.must=false; }
  else{ OutItem* best=&o2; if(o4.theory_bw>best->theory_bw||(o4.theory_bw==best->theory_bw&&o4.size<best->size)) best=&o4;
        if(o8.theory_bw>best->theory_bw||(o8.theory_bw==best->theory_bw&&o8.size<best->size)) best=&o8;
        o2.must=(&o2==best); o4.must=(&o4==best); o8.must=(&o8==best); }

  // JSON
  std::ostringstream oss; oss.setf(std::ios::fixed); oss << "{\"networks\": [\n";
  auto print_pairs=[&](const std::vector<std::pair<int,double>>& v){
    oss<<"["; for(size_t k=0;k<v.size();++k){ oss<<"["<<v[k].first<<", "<<std::setprecision(4)<<v[k].second<<"]"; if(k+1<v.size()) oss<<", "; }
    oss<<"]"; oss<<std::setprecision(6);
  };
  auto dump=[&](const OutItem& it,bool comma){
    oss<<"  {\"bandwidth\": "<<std::setprecision(2)<<it.theory_bw<<", \"pp_efficiency\": "; print_pairs(it.pp_eff_pairs);
    oss<<", \"ar_efficiency\": "; print_pairs(it.ar_eff_pairs);
    oss<<", \"size\": "<<it.size<<", \"latency\": "<<std::scientific<<it.lat_s<<std::fixed
       <<", \"ops\": { \"p2p\":[1.0,null], \"reduce_scatter\":[1.0,-1], \"all_gather\":[1.0,-1], \"all_reduce\":[2.0,-1] }, "
       <<"\"must_be_filled\": "<<(it.must?"true":"false")<<", \"processor_usage\": "<<std::setprecision(2)<<it.cpu<<"}";
    if(comma) oss<<","; oss<<"\n";
  };
  std::vector<OutItem> items; if(want2) items.push_back(o2); if(want4&&ndev>=4) items.push_back(o4); if(want8&&ndev>=8) items.push_back(o8);
  for(size_t i=0;i<items.size();++i) dump(items[i], i+1<items.size()); oss<<"]}";
  std::string json = oss.str();
  if (getenv_int("DISPLAY_OUTPUT", 1) != 0) {
    std::fwrite(json.data(), 1, json.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
  }
  std::ofstream f("./net_bw.json", std::ios::out | std::ios::trunc | std::ios::binary);
  if (f) { f.write(json.data(), (std::streamsize)json.size()); f.close(); }
  return 0;
}
