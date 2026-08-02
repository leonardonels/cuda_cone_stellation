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

#include <cstdio>

#include <rclcpp/rclcpp.hpp>

namespace {

using L = CudaSearch::L;
using scalar = CudaSearch::scalar;

/// Threads per block. One warp drives one frontier node, so this is 8 nodes in
/// flight; the frontier is usually far wider than that and the block loops.
const uint32_t kThreads = 256;
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
  uint32_t *result;  ///< [0] = seedsEmpty, [1] = chosen edge index
  uint32_t maxFrontier;
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
    if (trace->size >= 2) c.lastPos = e.mid[trace->edgeInd[trace->size - 2]];
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
  __shared__ uint32_t sFrontierSize;
  __shared__ uint32_t sNextSize;
  __shared__ BestRec sBest[kWarps];
  __shared__ uint32_t sSeedsEmpty;

  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid / 32;
  const uint32_t lane = tid % 32;

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

    if (tid == 0) sNextSize = 0;
    __syncthreads();

    // ONE THREAD per frontier slot, not one warp.
    //
    // A node has ~12 candidates, so a warp would leave 60% of its lanes idle
    // and then pay a five-round shuffle reduction to combine them -- measured
    // at 9.5 us/node against the host's 2.06 us. The parallelism that exists
    // here is ACROSS nodes (a level holds up to 2187 of them), not within one.
    // Thread-per-node also removes the per-batch __syncthreads(), leaving two
    // per level instead of one per eight nodes.
    for (uint32_t node = tid; node < nFront; node += kThreads) {
      const uint32_t slotBase = node * 3u;
      const bool inBounds = (slotBase + 3u <= a.maxFrontier);
      if (not inBounds) continue;

      const L::Trace t = front[node];

      // Own all three child slots unconditionally: whatever this node does not
      // fill has to be stamped dead, or the next level would read a trace left
      // over from two levels ago in the same buffer.
      next[slotBase + 0].size = 0;
      next[slotBase + 1].size = 0;
      next[slotBase + 2].size = 0;

      if (t.size == 0) continue;  // dead slot from the previous level

      const uint32_t order = orderBase + node;

      bool terminal = false;
      uint64_t k0 = kNoKey, k1 = kNoKey, k2 = kNoKey;
      if (t.size >= static_cast<uint32_t>(a.p.max_search_tree_height)) {
        terminal = true;
      } else {
        findNextEdges(a, &t, k0, k1, k2);
      }

      const uint64_t keys[3] = {k0, k1, k2};
      uint32_t nChild = 0;
      if (not terminal) {
        const ccs::Vec2T<scalar> actPos = a.e.mid[ccs::traceLast(t)];
        for (uint32_t r = 0; r < 3 and r < static_cast<uint32_t>(a.p.max_search_options); ++r) {
          if (keys[r] == kNoKey) break;
          const uint32_t idx = keyIdx(keys[r]);
          L::Trace aux = t;
          const bool closesLoop = ccs::wayClosesLoopWith(a.w, a.e.mid[idx], actPos, a.p);
          ccs::traceAppend(aux, idx, keyHeur(keys[r]), a.e.len[idx], closesLoop);
          next[slotBase + nChild] = aux;
          ++nChild;
        }
      }

      if (terminal or nChild == 0) {
        BestRec r;
        r.size = t.size;
        r.sumHeur = t.sumHeur;
        r.order = order;
        r.firstEdge = ccs::traceFirst(t);
        if (better(r, myBest)) myBest = r;
      } else {
        atomicMax(&sNextSize, slotBase + nChild);
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

struct CudaSearch::Device {
  L::Vec2 *edgeMid = nullptr, *edgeNormal = nullptr;
  scalar *edgeLen = nullptr;
  uint64_t *edgeHash = nullptr;
  uint32_t capEdgeMid = 0, capEdgeNormal = 0, capEdgeLen = 0, capEdgeHash = 0;

  L::Vec2 *wayMid = nullptr, *wayNormal = nullptr;
  uint64_t *wayHash = nullptr;
  uint32_t capWayMid = 0, capWayNormal = 0, capWayHash = 0;

  double *gx = nullptr, *gy = nullptr;
  uint32_t *cellStart = nullptr, *items = nullptr;
  uint32_t capGx = 0, capGy = 0, capItems = 0, capCellStart = 0;

  L::Trace *frontA = nullptr, *frontB = nullptr;
  uint32_t *result = nullptr;
};

namespace {

bool cudaOk(cudaError_t e, const char *what) {
  if (e == cudaSuccess) return true;
  RCLCPP_ERROR(rclcpp::get_logger("local_planner"), "CUDA %s failed: %s", what,
               cudaGetErrorString(e));
  return false;
}

/**
 * @brief Grows a zero-copy buffer.
 *
 * cudaHostAlloc(..., cudaHostAllocMapped), NOT cudaMallocManaged. Both "work"
 * on an integrated Orin, but managed memory is migrated page by page between
 * host and device access, and this search touches the Way from the host and
 * then reads it from the kernel once per OUTER ITERATION -- ~20 times per
 * callback. Measured, that migration cost ~450 us per outer iteration against a
 * 19 us launch+sync, i.e. it was the entire runtime. Mapped pinned memory is
 * physically the same DRAM for both processors on Tegra, so there is nothing to
 * migrate.
 */
template <typename T>
void ensure(T *&ptr, uint32_t &cap, uint32_t need) {
  if (cap >= need and ptr) return;
  if (ptr) cudaFreeHost(ptr);
  const uint32_t grown = need + need / 2 + 64;
  ptr = nullptr;
  cudaHostAlloc(&ptr, static_cast<size_t>(grown) * sizeof(T), cudaHostAllocMapped);
  cap = grown;
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

CudaSearch::CudaSearch(const Params::WayComputer &params) : wayParams_(params.way), dev_(new Device) {
  // The frontier is only ever touched by the device, so it is a plain device
  // allocation -- mapping it would put it on the slower coherent path for no
  // reason. Only `result` crosses back, and it is two words.
  cudaMalloc(&this->dev_->frontA, size_t(kMaxFrontier) * 3 * sizeof(L::Trace));
  cudaMalloc(&this->dev_->frontB, size_t(kMaxFrontier) * 3 * sizeof(L::Trace));
  cudaHostAlloc(&this->dev_->result, 2 * sizeof(uint32_t), cudaHostAllocMapped);
}

CudaSearch::~CudaSearch() {
  Device *d = this->dev_;
  cudaFreeHost(d->edgeMid);
  cudaFreeHost(d->edgeNormal);
  cudaFreeHost(d->edgeLen);
  cudaFreeHost(d->edgeHash);
  cudaFreeHost(d->wayMid);
  cudaFreeHost(d->wayNormal);
  cudaFreeHost(d->wayHash);
  cudaFreeHost(d->gx);
  cudaFreeHost(d->gy);
  cudaFreeHost(d->cellStart);
  cudaFreeHost(d->items);
  cudaFree(d->frontA);
  cudaFree(d->frontB);
  cudaFreeHost(d->result);
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

  memcpy(d->edgeMid, h.mid, size_t(h.size) * sizeof(L::Vec2));
  memcpy(d->edgeNormal, h.normal, size_t(h.size) * sizeof(L::Vec2));
  memcpy(d->edgeLen, h.len, size_t(h.size) * sizeof(scalar));
  memcpy(d->edgeHash, h.hash, size_t(h.size) * sizeof(uint64_t));

  const ccs::GridView g = this->grid_.view();
  const uint32_t nCells = static_cast<uint32_t>(g.nx) * static_cast<uint32_t>(g.ny) + 1;
  ensure(d->gx, d->capGx, h.size);
  ensure(d->gy, d->capGy, h.size);
  ensure(d->items, d->capItems, h.size);
  ensure(d->cellStart, d->capCellStart, nCells);

  if (h.size) {
    memcpy(d->gx, g.xs, size_t(h.size) * sizeof(double));
    memcpy(d->gy, g.ys, size_t(h.size) * sizeof(double));
    memcpy(d->items, g.items, size_t(h.size) * sizeof(uint32_t));
  }
  memcpy(d->cellStart, g.cellStart, size_t(nCells) * sizeof(uint32_t));
}

void CudaSearch::syncWayToDevice() {
  Device *d = this->dev_;
  const L::WaySoA h = this->waySoa_.view();
  ensure(d->wayMid, d->capWayMid, h.size + 1);
  ensure(d->wayNormal, d->capWayNormal, h.size + 1);
  ensure(d->wayHash, d->capWayHash, h.size + 1);
  if (h.size) {
    memcpy(d->wayMid, h.mid, size_t(h.size) * sizeof(L::Vec2));
    memcpy(d->wayNormal, h.normal, size_t(h.size) * sizeof(L::Vec2));
    memcpy(d->wayHash, h.hash, size_t(h.size) * sizeof(uint64_t));
  }
}

uint32_t CudaSearch::runOuterIteration(const L::SearchConsts &p, bool &seedsEmpty) {
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

  a.p = p;

  a.g = hg;
  a.g.xs = d->gx;
  a.g.ys = d->gy;
  a.g.cellStart = d->cellStart;
  a.g.items = d->items;

  a.frontA = d->frontA;
  a.frontB = d->frontB;
  a.result = d->result;
  a.maxFrontier = kMaxFrontier * 3;

  bfsKernel<<<1, kThreads>>>(a);
  if (not cudaOk(cudaGetLastError(), "launch")) {
    seedsEmpty = true;
    return 0;
  }
  if (not cudaOk(cudaDeviceSynchronize(), "sync")) {
    seedsEmpty = true;
    return 0;
  }

  seedsEmpty = (d->result[0] != 0);
  return d->result[1];
}

ISearch::Result CudaSearch::computeWay(const std::vector<Edge> &edges,
                                       const Params::WayComputer::Search &params,
                                       Way &way) {
  Result res;
  const CallbackTimer timer(this->stats_);

  way.trimByLocal();

  const L::SearchConsts p = this->makeConsts(params);
  this->buildEdgeSoA(edges, static_cast<double>(p.search_radius));
  this->resetWaySoA(way);
  this->syncEdgesToDevice();

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
