/**
 * @file cuda_search.cu
 * @brief CudaSearch: the whole bounded BFS in one kernel per outer iteration.
 *
 * See cuda_search.hpp for why the strategy is shaped this way. The filters
 * themselves are not here -- they are ccs::discardCandidate and friends from
 * structures/search_policy.hpp, compiled for the device. That is the point of
 * the policy header: this file decides HOW the space is explored and never what
 * counts as a valid candidate.
 */

#include "modules/cuda_search.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <ctime>

#include <rclcpp/rclcpp.hpp>

namespace {

using L = CudaSearch::L;
using scalar = CudaSearch::scalar;

/// Threads per block. One warp drives one frontier node, so this is 8 nodes in
/// flight; the frontier is usually far wider than that and the block loops.
const uint32_t kThreads = 768;
const uint32_t kWarps = kThreads / 32;

/* -------------------------------------------------------------------------- */
/*                              Candidate ranking                              */
/* -------------------------------------------------------------------------- */

/**
 * @brief Packs (heuristic, edgeIndex) into one unsigned key that sorts exactly
 * as std::pair<float,uint32_t>::operator< does.
 *
 * The heuristic is always > 0 here: distHeur >= 0, and angleHeur = -log(inner)
 * with inner <= 0.8 by construction, so angleHeur > 0. For positive floats the
 * IEEE-754 bit pattern is monotonically increasing, so comparing the packed
 * keys as uint64 compares first by heuristic then by index -- the same total
 * order the host relies on to make the result independent of query order.
 */
__device__ __forceinline__ uint64_t packKey(float heur, uint32_t idx) {
  uint32_t b = __float_as_uint(heur);
  return (static_cast<uint64_t>(b) << 32) | static_cast<uint64_t>(idx);
}

__device__ __forceinline__ float keyHeur(uint64_t k) {
  return __uint_as_float(static_cast<uint32_t>(k >> 32));
}

__device__ __forceinline__ uint32_t keyIdx(uint64_t k) { return static_cast<uint32_t>(k); }

const uint64_t kNoKey = ~uint64_t(0);

/// Inserts one key into an ascending triple, keeping the three smallest.
__device__ __forceinline__ void insert3(uint64_t &a0, uint64_t &a1, uint64_t &a2, uint64_t v) {
  if (v < a0) {
    a2 = a1;
    a1 = a0;
    a0 = v;
  } else if (v < a1) {
    a2 = a1;
    a1 = v;
  } else if (v < a2) {
    a2 = v;
  }
}

/* -------------------------------------------------------------------------- */
/*                                 Best trace                                  */
/* -------------------------------------------------------------------------- */

/**
 * @brief A finished trace competing to be "best".
 *
 * The host picks by: longer wins; on a tie, smaller accumulated heuristic wins;
 * and because its comparison is strict, an earlier trace keeps the title
 * against an exact tie. \a order reproduces that last rule -- it is the trace's
 * position in the host's FIFO, so "earlier" is well defined on the device even
 * though the traces are produced in parallel. The initial best (the host's
 * queue_.front()) is entered with order 0 and everything else is shifted up, so
 * it wins every exact tie exactly as it does on the host.
 */
struct BestRec {
  uint32_t size;
  float sumHeur;
  uint32_t order;
  uint32_t firstEdge;
};

__device__ __forceinline__ bool better(const BestRec &a, const BestRec &b) {
  if (a.size != b.size) return a.size > b.size;
  if (a.sumHeur != b.sumHeur) return a.sumHeur < b.sumHeur;
  return a.order < b.order;
}

/* -------------------------------------------------------------------------- */
/*                                   Kernel                                    */
/* -------------------------------------------------------------------------- */

struct KernelArgs {
  L::EdgeSoA e;
  L::WaySoA w;
  L::SearchConsts p;
  ccs::GridView g;
  L::Trace *frontA;
  L::Trace *frontB;
  uint64_t *childKeys;
  uint32_t *childCount;
  uint32_t *childOffset;
  uint32_t *result;  ///< [0] = seedsEmpty, [1] = chosen edge index
  uint32_t maxFrontier;
  bool wayInShared;  ///< whether the launch reserved shared space for the Way
};

/**
 * @brief Evaluates one node's candidates and returns its best three.
 *
 * \a trace is null for the seed query. Mirrors CpuFastSearch::findNextEdges:
 * grid query, discard filters, heuristic threshold, best max_search_options by
 * (heuristic, index).
 */
__device__ void findNextEdges(const KernelArgs &a, const L::Trace *trace, uint64_t &k0,
                              uint64_t &k1, uint64_t &k2) {
  k0 = k1 = k2 = kNoKey;

  L::SearchContext c;
  c.hasActEdge = false;
  c.actEdgeMid = ccs::vec2<scalar>(0, 0);
  c.actEdgeNormal = ccs::vec2<scalar>(0, 0);
  c.actEdgeHash = 0;
  c.actPos = ccs::vec2<scalar>(0, 0);
  c.lastPos = ccs::vec2<scalar>(0, 0);

  const L::EdgeSoA &e = a.e;
  const L::WaySoA &w = a.w;

  if (trace and trace->size > 0) {
    const uint32_t i = ccs::traceLast(*trace);
    c.hasActEdge = true;
    c.actEdgeMid = e.mid[i];
    c.actEdgeNormal = e.normal[i];
    c.actEdgeHash = e.hash[i];
    c.actPos = e.mid[i];
    if (trace->size >= 2) c.lastPos = e.mid[ccs::tracePrev(*trace)];
  }
  if (w.size > 0) {
    if (not c.hasActEdge) {
      const uint32_t b = w.size - 1;
      c.hasActEdge = true;
      c.actEdgeMid = w.mid[b];
      c.actEdgeNormal = w.normal[b];
      c.actEdgeHash = w.hash[b];
      c.actPos = w.mid[b];
      if (w.size >= 2) c.lastPos = w.mid[w.size - 2];
    } else if (trace->size < 2) {
      c.lastPos = w.mid[w.size - 1];
    }
  }

  if (c.hasActEdge and (w.size >= 2 or trace))
    c.dir = ccs::vecFromTo(c.lastPos, c.actPos);
  else
    c.dir = ccs::vec2<scalar>(1, 0);

  c.avgEdgeLen = ccs::combinedAvgEdgeLen(w, trace);
  c.hasTrace = (trace != nullptr);
  c.traceLoopClosed = trace ? trace->loopClosed : false;
  c.sideCheckEnabled = (w.size >= 2 or trace != nullptr);

  // Grid query. The cell range comes from the shared helper so the device
  // visits exactly the cells the host does.
  const double qx = static_cast<double>(c.actPos.x), qy = static_cast<double>(c.actPos.y);
  if (a.g.n == 0) return;

  int32_t gxLo, gxHi, gyLo, gyHi;
  ccs::gridCellRange(a.g, qx, qy, gxLo, gxHi, gyLo, gyHi);

  for (int32_t gy = gyLo; gy <= gyHi; ++gy) {
    const uint32_t rowBase = static_cast<uint32_t>(gy) * static_cast<uint32_t>(a.g.nx);
    for (int32_t gx = gxLo; gx <= gxHi; ++gx) {
      const uint32_t cell = rowBase + static_cast<uint32_t>(gx);
      const uint32_t begin = a.g.cellStart[cell], end = a.g.cellStart[cell + 1];
      for (uint32_t k = begin; k < end; ++k) {
        const uint32_t i = a.g.items[k];
        if (not ccs::gridHit(a.g, i, qx, qy)) continue;
        if (ccs::discardCandidate(c, w, e.mid[i], e.len[i], e.hash[i], a.p)) continue;
        const scalar h = ccs::heuristic(c.actPos, e.mid[i], c.dir, a.p);
        if (h <= a.p.max_next_heuristic) insert3(k0, k1, k2, packKey(h, i));
      }
    }
  }
}

/**
 * @brief One outer iteration: the seed query, then the bounded BFS, then the
 * best trace's first edge.
 *
 * Launched as a SINGLE block. The work per outer iteration is ~360 nodes of a
 * dozen candidates each, which is far too little to fill 8 SMs; splitting it
 * across blocks would only add grid-wide synchronisation to a workload whose
 * cost is already synchronisation.
 */
__global__ __launch_bounds__(kThreads) void bfsKernel(KernelArgs a) {
  // The Way, staged in shared memory. wayIntersectsWith() and wayContainsEdge()
  // walk ALL of it for every candidate, so at ~48 midpoints and ~12 candidates
  // per node this is by far the most re-read data in the kernel. 24 bytes per
  // element, so a 48-element Way is 1.2 KB -- nothing. The host only launches
  // with shared space when it fits (see runOuterIteration); otherwise
  // a.w already points at device memory and this is skipped.
  extern __shared__ unsigned char sWayRaw[];
  __shared__ uint32_t sScan[kThreads];
  __shared__ uint32_t sFrontierSize;
  __shared__ uint32_t sNextSize;
  __shared__ BestRec sBest[kWarps];
  __shared__ uint32_t sSeedsEmpty;

  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid / 32;
  const uint32_t lane = tid % 32;

  if (a.wayInShared and a.w.size > 0) {
    L::Vec2 *sMid = reinterpret_cast<L::Vec2 *>(sWayRaw);
    L::Vec2 *sNormal = sMid + a.w.size;
    L::Vec2 *sSegMin = sNormal + a.w.size;
    L::Vec2 *sSegMax = sSegMin + a.w.size;
    uint64_t *sHash = reinterpret_cast<uint64_t *>(sSegMax + a.w.size);
    uint64_t *sTable = sHash + a.w.size;

    const bool haveBoxes = (a.w.segMin != nullptr);
    const bool haveTable = (a.w.hashTable != nullptr);
    const uint32_t tableSize = haveTable ? (a.w.hashMask + 1u) : 0u;

    for (uint32_t i = tid; i < a.w.size; i += kThreads) {
      sMid[i] = a.w.mid[i];
      sNormal[i] = a.w.normal[i];
      sHash[i] = a.w.hash[i];
      if (haveBoxes) {
        sSegMin[i] = a.w.segMin[i];
        sSegMax[i] = a.w.segMax[i];
      }
    }
    for (uint32_t i = tid; i < tableSize; i += kThreads) sTable[i] = a.w.hashTable[i];
    __syncthreads();

    a.w.mid = sMid;
    a.w.normal = sNormal;
    a.w.hash = sHash;
    if (haveBoxes) {
      a.w.segMin = sSegMin;
      a.w.segMax = sSegMax;
    }
    if (haveTable) a.w.hashTable = sTable;
  }

  L::Trace *front = a.frontA;
  L::Trace *next = a.frontB;

  /* ---------------------------- seeds (level 0) --------------------------- */
  if (tid == 0) {
    sFrontierSize = 0;
    sSeedsEmpty = 0;
  }
  __syncthreads();

  if (tid == 0) {
    uint64_t k0, k1, k2;
    findNextEdges(a, nullptr, k0, k1, k2);
    const uint64_t keys[3] = {k0, k1, k2};
    uint32_t n = 0;
    const ccs::Vec2T<scalar> wayBack = ccs::wayBackMid(a.w);
    for (uint32_t r = 0; r < 3 and r < static_cast<uint32_t>(a.p.max_search_options); ++r) {
      if (keys[r] == kNoKey) break;
      const uint32_t idx = keyIdx(keys[r]);
      L::Trace t;
      ccs::traceInit(t);
      const bool closesLoop = ccs::wayClosesLoopWith(a.w, a.e.mid[idx], wayBack, a.p);
      ccs::traceAppend(t, idx, keyHeur(keys[r]), a.e.len[idx], closesLoop);
      front[n++] = t;
    }
    sFrontierSize = n;
    sSeedsEmpty = (n == 0) ? 1u : 0u;
  }
  __syncthreads();

  if (sSeedsEmpty) {
    if (tid == 0) {
      a.result[0] = 1;
      a.result[1] = 0;
    }
    return;
  }

  /* ------------------------------- the BFS -------------------------------- */
  // The host's FIFO visits traces in strict level order, and within a level in
  // parent order then child rank. `order` below reproduces exactly that
  // numbering, which is what makes the tie-break reproducible.
  BestRec best;
  best.size = front[0].size;
  best.sumHeur = front[0].sumHeur;
  best.order = 0;
  best.firstEdge = ccs::traceFirst(front[0]);

  BestRec myBest = best;

  // Global visit order, used only to break exact ties the way the host's FIFO
  // does. The frontier is SPARSE -- a parent owns slots [node*3, node*3+3)
  // whether or not it fills them -- so `order` is derived from the slot index
  // rather than from a dense count. That is enough: the tie-break only needs
  // the RELATIVE order of two traces, and slot order preserves the host's
  // parent-then-rank ordering exactly. Making it dense would need a block-wide
  // prefix sum per level, which would cost more than the ties are worth.
  uint32_t orderBase = 1;  // 0 is taken by the initial best

  for (uint32_t level = 0; level < static_cast<uint32_t>(a.p.max_search_tree_height); ++level) {
    const uint32_t nFront = sFrontierSize;
    if (nFront == 0) break;


    // ONE THREAD per frontier node, and the frontier is DENSE.
    //
    // It used to be sparse -- a parent owned slots [node*3, node*3+3) whether
    // or not it filled them -- which meant scanning ~3x more slots than there
    // were live nodes, and left only ~10 of a warp's 32 lanes doing work while
    // the rest hit a dead slot and exited. Compacting costs one block-wide
    // prefix sum per level and buys back both.
    //
    // Pass A evaluates each node and records how many children it wants;
    // the scan turns those counts into write offsets, in node order, so the
    // next frontier comes out in exactly the host FIFO's order; pass B writes
    // the children there.
    for (uint32_t node = tid; node < nFront; node += kThreads) {
      const L::Trace t = front[node];
      const uint32_t order = orderBase + node;

      bool terminal = (t.size >= static_cast<uint32_t>(a.p.max_search_tree_height));
      uint64_t k0 = kNoKey, k1 = kNoKey, k2 = kNoKey;
      if (not terminal) findNextEdges(a, &t, k0, k1, k2);

      uint32_t nChild = 0;
      if (not terminal) {
        const uint64_t keys[3] = {k0, k1, k2};
        for (uint32_t r = 0; r < 3 and r < static_cast<uint32_t>(a.p.max_search_options); ++r) {
          if (keys[r] == kNoKey) break;
          a.childKeys[node * 3u + r] = keys[r];
          ++nChild;
        }
      }
      a.childCount[node] = nChild;

      if (nChild == 0) {
        BestRec r;
        r.size = t.size;
        r.sumHeur = t.sumHeur;
        r.order = order;
        r.firstEdge = ccs::traceFirst(t);
        if (better(r, myBest)) myBest = r;
      }
    }
    __syncthreads();

    // Block-wide exclusive scan of childCount over [0, nFront).
    {
      const uint32_t chunk = (nFront + kThreads - 1) / kThreads;
      const uint32_t lo = tid * chunk;
      const uint32_t hi = (lo + chunk < nFront) ? (lo + chunk) : nFront;
      uint32_t localSum = 0;
      for (uint32_t i = lo; i < hi; ++i) localSum += a.childCount[i];
      sScan[tid] = localSum;
      __syncthreads();

      // 256 entries: a sequential scan by one thread is a few hundred cycles,
      // and cheaper than the syncs a parallel scan of this size would need.
      if (tid == 0) {
        uint32_t running = 0;
        for (uint32_t i = 0; i < kThreads; ++i) {
          const uint32_t v = sScan[i];
          sScan[i] = running;
          running += v;
        }
        sNextSize = running;
      }
      __syncthreads();

      uint32_t running = sScan[tid];
      for (uint32_t i = lo; i < hi; ++i) {
        a.childOffset[i] = running;
        running += a.childCount[i];
      }
    }
    __syncthreads();

    // Pass B: materialise the children at their dense offsets.
    for (uint32_t node = tid; node < nFront; node += kThreads) {
      const uint32_t nChild = a.childCount[node];
      if (nChild == 0) continue;
      const L::Trace t = front[node];
      const ccs::Vec2T<scalar> actPos = a.e.mid[ccs::traceLast(t)];
      const uint32_t base = a.childOffset[node];
      for (uint32_t r = 0; r < nChild; ++r) {
        if (base + r >= a.maxFrontier) break;
        const uint64_t key = a.childKeys[node * 3u + r];
        const uint32_t idx = keyIdx(key);
        L::Trace aux = t;
        const bool closesLoop = ccs::wayClosesLoopWith(a.w, a.e.mid[idx], actPos, a.p);
        ccs::traceAppend(aux, idx, keyHeur(key), a.e.len[idx], closesLoop);
        next[base + r] = aux;
      }
    }
    __syncthreads();

    orderBase += nFront;
    const uint32_t produced = sNextSize;
    if (tid == 0) sFrontierSize = produced;
    __syncthreads();

    L::Trace *tmp = front;
    front = next;
    next = tmp;
  }

  /* --------------------------- reduce the best ---------------------------- */
  // Warp reduction first, then across warps through shared memory.
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    BestRec o;
    o.size = __shfl_down_sync(0xffffffffu, myBest.size, off);
    o.sumHeur = __shfl_down_sync(0xffffffffu, myBest.sumHeur, off);
    o.order = __shfl_down_sync(0xffffffffu, myBest.order, off);
    o.firstEdge = __shfl_down_sync(0xffffffffu, myBest.firstEdge, off);
    if (better(o, myBest)) myBest = o;
  }
  if (lane == 0) sBest[warp] = myBest;
  __syncthreads();

  if (tid == 0) {
    BestRec r = sBest[0];
    for (uint32_t i = 1; i < kWarps; ++i)
      if (better(sBest[i], r)) r = sBest[i];
    a.result[0] = 0;
    a.result[1] = r.firstEdge;
  }
}

}  // namespace

/* -------------------------------------------------------------------------- */
/*                                Host driver                                  */
/* -------------------------------------------------------------------------- */

namespace {

/// Process CPU time (ms), all threads -- a driver thread spinning counts.
double cpuMs() {
  timespec ts;
  if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) != 0) return 0.0;
  return double(ts.tv_sec) * 1e3 + double(ts.tv_nsec) / 1e6;
}

double wallMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

/// Where the callback's time goes, split host/device. Accumulated across the
/// run and reported periodically: "the CUDA backend is cheap on CPU" is only
/// actionable if you know WHICH part of the remaining CPU is which.
struct Phases {
  double kernelMs = 0;                 ///< device-measured kernel duration
  double soaCpu = 0, soaWall = 0;      ///< buildEdgeSoA + grid build + resetWaySoA
  double uploadCpu = 0, uploadWall = 0;///< syncEdgesToDevice memcpys
  double iterCpu = 0, iterWall = 0;    ///< syncWayToDevice + launch + sync, x outer iters
  uint64_t callbacks = 0, launches = 0;
};
Phases g_phases;

}  // namespace

struct CudaSearch::Device {
  L::Vec2 *edgeMid = nullptr, *edgeNormal = nullptr;
  scalar *edgeLen = nullptr;
  uint64_t *edgeHash = nullptr;
  uint32_t capEdgeMid = 0, capEdgeNormal = 0, capEdgeLen = 0, capEdgeHash = 0;

  L::Vec2 *wayMid = nullptr, *wayNormal = nullptr;
  uint64_t *wayHash = nullptr;
  uint32_t capWayMid = 0, capWayNormal = 0, capWayHash = 0;

  L::Vec2 *waySegMin = nullptr, *waySegMax = nullptr;
  uint64_t *wayTable = nullptr;
  uint32_t capSegMin = 0, capSegMax = 0, capTable = 0;

  double *gx = nullptr, *gy = nullptr;
  uint32_t *cellStart = nullptr, *items = nullptr;
  uint32_t capGx = 0, capGy = 0, capItems = 0, capCellStart = 0;

  L::Trace *frontA = nullptr, *frontB = nullptr;
  uint64_t *childKeys = nullptr;   ///< 3 candidate keys per frontier node
  uint32_t *childCount = nullptr;  ///< children each node produced
  uint32_t *childOffset = nullptr; ///< exclusive scan of childCount
  uint32_t *result = nullptr;

  /// Device-side kernel timing, to separate the kernel from the round trip.
  cudaEvent_t evStart = nullptr, evStop = nullptr;
};

namespace {

bool cudaOk(cudaError_t e, const char *what) {
  if (e == cudaSuccess) return true;
  RCLCPP_ERROR(rclcpp::get_logger("cuda_cone_stellation"), "CUDA %s failed: %s", what,
               cudaGetErrorString(e));
  return false;
}

/**
 * @brief Grows a DEVICE buffer.
 *
 * Device memory, not mapped-pinned and not managed. All three "work" on an
 * integrated Orin and it is tempting to conclude the copy is pointless there --
 * it is not. Mapped-pinned is reached by the GPU over the coherent fabric,
 * which is far slower and higher latency than device DRAM, and this kernel's
 * hottest array is the Way: wayIntersectsWith() re-walks all ~48 of its
 * segments for EVERY candidate. Putting that on the host path made the whole
 * frontier scan latency-bound. Managed memory is worse again, since it migrates
 * page by page across the host writes that happen once per outer iteration.
 *
 * The staged copies are kilobytes and happen once per callback (edges, grid) or
 * once per outer iteration (the Way, ~1.2 KB), so paying for them is trivially
 * worth it.
 */
template <typename T>
void ensure(T *&ptr, uint32_t &cap, uint32_t need) {
  if (cap >= need and ptr) return;
  if (ptr) cudaFree(ptr);
  const uint32_t grown = need + need / 2 + 64;
  ptr = nullptr;
  cudaMalloc(&ptr, static_cast<size_t>(grown) * sizeof(T));
  cap = grown;
}

template <typename T>
void upload(T *dst, const T *src, size_t n) {
  if (n) cudaMemcpyAsync(dst, src, n * sizeof(T), cudaMemcpyHostToDevice, 0);
}

}  // namespace

bool CudaSearch::deviceAvailable() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) return false;
  return n > 0;
}

bool CudaSearch::supportsParams(const Params::WayComputer &params, const char **why) {
  const int opts = params.search.max_search_options;
  int height = params.search.max_search_tree_height;
  if (height > CCS_MAX_TRACE_LEN) height = CCS_MAX_TRACE_LEN;

  if (opts > 3) {
    *why = "max_search_options > 3 exceeds the device top-k width";
    return false;
  }

  double widest = 1.0;
  for (int i = 0; i < height; ++i) {
    widest *= double(opts);
    if (widest > double(kMaxFrontier)) {
      *why = "max_search_options^max_search_tree_height exceeds the device frontier capacity";
      return false;
    }
  }
  return true;
}

void CudaSearch::reportBackendDetail() const {
  if (g_phases.callbacks == 0) return;
  const double n = double(g_phases.callbacks);
  RCLCPP_INFO(rclcpp::get_logger("cuda_cone_stellation"),
              "[cuda phases/callback] soa %.3f ms cpu | upload %.3f | iters %.3f cpu / %.3f wall "
              "over %.1f launches | kernel %.3f ms (%.0f%% of iter wall), %.1f us/launch",
              g_phases.soaCpu / n, g_phases.uploadCpu / n, g_phases.iterCpu / n,
              g_phases.iterWall / n, double(g_phases.launches) / n, g_phases.kernelMs / n,
              g_phases.iterWall > 0 ? 100.0 * g_phases.kernelMs / g_phases.iterWall : 0.0,
              g_phases.launches ? 1000.0 * g_phases.kernelMs / double(g_phases.launches) : 0.0);
}

CudaSearch::CudaSearch(const Params::WayComputer &params)
    : wayParams_(params.way), dev_(new Device) {
  // The frontier is only ever touched by the device, so it is a plain device
  // allocation -- mapping it would put it on the slower coherent path for no
  // reason. Only `result` crosses back, and it is two words.
  cudaMalloc(&this->dev_->frontA, size_t(kMaxFrontier) * 3 * sizeof(L::Trace));
  cudaMalloc(&this->dev_->frontB, size_t(kMaxFrontier) * 3 * sizeof(L::Trace));
  cudaMalloc(&this->dev_->childKeys, size_t(kMaxFrontier) * 3 * sizeof(uint64_t));
  cudaMalloc(&this->dev_->childCount, size_t(kMaxFrontier) * sizeof(uint32_t));
  cudaMalloc(&this->dev_->childOffset, size_t(kMaxFrontier) * sizeof(uint32_t));
  cudaHostAlloc(&this->dev_->result, 2 * sizeof(uint32_t), cudaHostAllocMapped);
  cudaEventCreate(&this->dev_->evStart);
  cudaEventCreate(&this->dev_->evStop);
}

CudaSearch::~CudaSearch() {
  Device *d = this->dev_;
  cudaFree(d->edgeMid);
  cudaFree(d->edgeNormal);
  cudaFree(d->edgeLen);
  cudaFree(d->edgeHash);
  cudaFree(d->wayMid);
  cudaFree(d->wayNormal);
  cudaFree(d->wayHash);
  cudaFree(d->waySegMin);
  cudaFree(d->waySegMax);
  cudaFree(d->wayTable);
  cudaFree(d->gx);
  cudaFree(d->gy);
  cudaFree(d->cellStart);
  cudaFree(d->items);
  cudaFree(d->frontA);
  cudaFree(d->frontB);
  cudaFree(d->childKeys);
  cudaFree(d->childCount);
  cudaFree(d->childOffset);
  cudaFreeHost(d->result);
  if (d->evStart) cudaEventDestroy(d->evStart);
  if (d->evStop) cudaEventDestroy(d->evStop);
  delete d;
}

CudaSearch::L::SearchConsts CudaSearch::makeConsts(const Params::WayComputer::Search &params) const {
  L::SearchConsts p;
  p.search_radius = static_cast<scalar>(params.search_radius);
  p.max_angle_diff = static_cast<scalar>(params.max_angle_diff);
  p.edge_len_diff_factor = static_cast<scalar>(params.edge_len_diff_factor);
  p.max_next_heuristic = static_cast<scalar>(params.max_next_heuristic);
  p.heur_dist_ponderation = params.heur_dist_ponderation;
  p.allow_intersection = params.allow_intersection;
  p.max_search_options = params.max_search_options;
  p.max_search_tree_height = params.max_search_tree_height;
  if (p.max_search_tree_height > CCS_MAX_TRACE_LEN) p.max_search_tree_height = CCS_MAX_TRACE_LEN;
  p.max_dist_loop_closure = static_cast<scalar>(this->wayParams_.max_dist_loop_closure);
  p.max_angle_diff_loop_closure = static_cast<scalar>(this->wayParams_.max_angle_diff_loop_closure);
  p.min_loop_size = MIN_LOOP_SIZE;
  return p;
}

namespace {

struct EdgeFields {
  L::Vec2 mid;
  L::Vec2 normal;
  uint64_t hash;
};

EdgeFields edgeFields(const Edge &e) {
  const Point mid = e.midPoint();
  const Vector normal = e.normal();
  EdgeFields f;
  f.mid = ccs::vec2<scalar>(static_cast<scalar>(mid.x), static_cast<scalar>(mid.y));
  f.normal = ccs::vec2<scalar>(static_cast<scalar>(normal.x), static_cast<scalar>(normal.y));
  f.hash = std::hash<Edge>{}(e);
  return f;
}

}  // namespace

void CudaSearch::buildEdgeSoA(const std::vector<Edge> &edges, double searchRadius) {
  this->edgeSoa_.clear();
  this->edgeSoa_.reserve(edges.size());
  this->gridX_.clear();
  this->gridY_.clear();
  this->gridX_.reserve(edges.size());
  this->gridY_.reserve(edges.size());

  for (const Edge &e : edges) {
    const EdgeFields f = edgeFields(e);
    this->edgeSoa_.push(f.mid, f.normal, static_cast<scalar>(e.len), f.hash);
    this->gridX_.push_back(static_cast<double>(f.mid.x));
    this->gridY_.push_back(static_cast<double>(f.mid.y));
  }

  this->grid_.build(this->gridX_.data(), this->gridY_.data(),
                    static_cast<uint32_t>(this->gridX_.size()), searchRadius);
}

void CudaSearch::resetWaySoA(const Way &way) {
  this->waySoa_.clear();
  this->waySoa_.setBuildIndex(true);
  for (const Edge &e : way.edges()) {
    const EdgeFields f = edgeFields(e);
    this->waySoa_.push(f.mid, f.normal, f.hash);
  }
  this->waySoa_.setAvgEdgeLen(static_cast<scalar>(way.getAvgEdgeLen()));
}

void CudaSearch::appendToWaySoA(const Edge &edge, const Way &way) {
  const EdgeFields f = edgeFields(edge);
  this->waySoa_.push(f.mid, f.normal, f.hash);
  this->waySoa_.setAvgEdgeLen(static_cast<scalar>(way.getAvgEdgeLen()));
}

void CudaSearch::syncEdgesToDevice() {
  Device *d = this->dev_;
  const L::EdgeSoA h = this->edgeSoa_.view();
  ensure(d->edgeMid, d->capEdgeMid, h.size);
  ensure(d->edgeNormal, d->capEdgeNormal, h.size);
  ensure(d->edgeLen, d->capEdgeLen, h.size);
  ensure(d->edgeHash, d->capEdgeHash, h.size);

  upload(d->edgeMid, h.mid, h.size);
  upload(d->edgeNormal, h.normal, h.size);
  upload(d->edgeLen, h.len, h.size);
  upload(d->edgeHash, h.hash, h.size);

  const ccs::GridView g = this->grid_.view();
  const uint32_t nCells = static_cast<uint32_t>(g.nx) * static_cast<uint32_t>(g.ny) + 1;
  ensure(d->gx, d->capGx, h.size);
  ensure(d->gy, d->capGy, h.size);
  ensure(d->items, d->capItems, h.size);
  ensure(d->cellStart, d->capCellStart, nCells);

  upload(d->gx, g.xs, h.size);
  upload(d->gy, g.ys, h.size);
  upload(d->items, g.items, h.size);
  upload(d->cellStart, g.cellStart, nCells);
}

void CudaSearch::syncWayToDevice() {
  Device *d = this->dev_;
  const L::WaySoA h = this->waySoa_.view();
  ensure(d->wayMid, d->capWayMid, h.size + 1);
  ensure(d->wayNormal, d->capWayNormal, h.size + 1);
  ensure(d->wayHash, d->capWayHash, h.size + 1);
  upload(d->wayMid, h.mid, h.size);
  upload(d->wayNormal, h.normal, h.size);
  upload(d->wayHash, h.hash, h.size);

  // The accelerating indexes have to cross too, or the kernel reads host
  // pointers it cannot dereference.
  const std::vector<uint64_t> &table = this->waySoa_.hashTable();
  ensure(d->waySegMin, d->capSegMin, h.size + 1);
  ensure(d->waySegMax, d->capSegMax, h.size + 1);
  ensure(d->wayTable, d->capTable, static_cast<uint32_t>(table.size()) + 1);
  upload(d->waySegMin, this->waySoa_.segMin().data(), h.size);
  upload(d->waySegMax, this->waySoa_.segMax().data(), h.size);
  upload(d->wayTable, table.data(), table.size());
  this->tableSize_ = static_cast<uint32_t>(table.size());
}

uint32_t CudaSearch::runOuterIteration(const L::SearchConsts &p, bool &seedsEmpty) {
  const double c0 = cpuMs(), w0 = wallMs();
  ++g_phases.launches;
  Device *d = this->dev_;
  this->syncWayToDevice();

  const L::EdgeSoA he = this->edgeSoa_.view();
  const L::WaySoA hw = this->waySoa_.view();
  const ccs::GridView hg = this->grid_.view();

  KernelArgs a;
  a.e.mid = d->edgeMid;
  a.e.normal = d->edgeNormal;
  a.e.len = d->edgeLen;
  a.e.hash = d->edgeHash;
  a.e.size = he.size;

  a.w.mid = d->wayMid;
  a.w.normal = d->wayNormal;
  a.w.hash = d->wayHash;
  a.w.size = hw.size;
  a.w.avgEdgeLen = hw.avgEdgeLen;
  a.w.hashTable = (hw.hashTable and this->tableSize_) ? d->wayTable : nullptr;
  a.w.hashMask = hw.hashMask;
  a.w.hashEmpty = hw.hashEmpty;
  a.w.segMin = hw.segMin ? d->waySegMin : nullptr;
  a.w.segMax = hw.segMax ? d->waySegMax : nullptr;

  a.p = p;

  a.g = hg;
  a.g.xs = d->gx;
  a.g.ys = d->gy;
  a.g.cellStart = d->cellStart;
  a.g.items = d->items;

  a.frontA = d->frontA;
  a.frontB = d->frontB;
  a.childKeys = d->childKeys;
  a.childCount = d->childCount;
  a.childOffset = d->childOffset;
  a.result = d->result;
  a.maxFrontier = kMaxFrontier * 3;

  // 32 KB ceiling: comfortably inside the 48 KB a block gets by default, and
  // ~1365 Way elements, where the measured steady state is 48. A Way past that
  // (the unlimited-horizon pass over a whole map can produce one) simply runs
  // against device memory instead of failing.
  const size_t wayBytes = size_t(hw.size) * (4 * sizeof(L::Vec2) + sizeof(uint64_t)) +
                          size_t(this->tableSize_) * sizeof(uint64_t);
  a.wayInShared = (wayBytes <= 32u * 1024u);
  const size_t sharedBytes = a.wayInShared ? wayBytes : 0;

  cudaEventRecord(d->evStart, 0);
  bfsKernel<<<1, kThreads, sharedBytes>>>(a);
  cudaEventRecord(d->evStop, 0);
  if (not cudaOk(cudaGetLastError(), "launch")) {
    seedsEmpty = true;
    return 0;
  }
  if (not cudaOk(cudaDeviceSynchronize(), "sync")) {
    seedsEmpty = true;
    return 0;
  }

  float kms = 0;
  if (cudaEventElapsedTime(&kms, d->evStart, d->evStop) == cudaSuccess) g_phases.kernelMs += kms;

  seedsEmpty = (d->result[0] != 0);
  g_phases.iterCpu += cpuMs() - c0;
  g_phases.iterWall += wallMs() - w0;
  return d->result[1];
}

ISearch::Result CudaSearch::computeWay(const std::vector<Edge> &edges,
                                       const Params::WayComputer::Search &params,
                                       Way &way) {
  Result res;
  const CallbackTimer timer(this->stats_);

  way.trimByLocal();

  const L::SearchConsts p = this->makeConsts(params);

  double c0 = cpuMs(), w0 = wallMs();
  this->buildEdgeSoA(edges, static_cast<double>(p.search_radius));
  this->resetWaySoA(way);
  g_phases.soaCpu += cpuMs() - c0;
  g_phases.soaWall += wallMs() - w0;

  c0 = cpuMs();
  w0 = wallMs();
  this->syncEdgesToDevice();
  g_phases.uploadCpu += cpuMs() - c0;
  g_phases.uploadWall += wallMs() - w0;

  ++g_phases.callbacks;

  bool seedsEmpty = false;
  uint32_t nextEdgeInd = this->runOuterIteration(p, seedsEmpty);

  while (rclcpp::ok() and not seedsEmpty and
         (!params.max_way_horizon_size or
          way.sizeAheadOfCar() <= static_cast<uint32_t>(params.max_way_horizon_size))) {
    ++this->stats_.outerIters;
    this->stats_.waySizeSum += this->waySoa_.size();

    const Edge &nextEdge = edges[nextEdgeInd];
    way.addEdge(nextEdge);
    this->appendToWaySoA(nextEdge, way);

    if (way.closesLoop()) {
      res.wayToPublish = way.restructureClosure();
      res.loopClosed = true;
      return res;
    }

    nextEdgeInd = this->runOuterIteration(p, seedsEmpty);
  }

  res.loopClosed = false;
  res.wayToPublish = way;
  return res;
}
