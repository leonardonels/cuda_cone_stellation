/**
 * @file cuda_one_shot_search.cu
 * @brief CudaOneShotSearch: the whole callback in one kernel launch.
 *
 * See cuda_one_shot_search.hpp for why this backend exists and what had to move
 * onto the device for it to. The filters are still ccs::discardCandidate and
 * friends from structures/search_policy.hpp -- this file decides HOW the space
 * is explored and never what counts as a valid candidate.
 *
 * The bounded BFS below is a deliberate copy of the one in cuda_search.cu, so
 * that the backend it is being measured against stays untouched. Keep them in
 * step, or collapse them onto a shared device header once one of the two wins.
 */

#include "modules/cuda_one_shot_search.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <ctime>

#include <rclcpp/rclcpp.hpp>

namespace {

using L = CudaOneShotSearch::L;
using scalar = CudaOneShotSearch::scalar;

/**
 * @brief Threads per block. One thread drives one frontier node.
 *
 * 384, not the 768 this started at, and NOT because the block loops -- it
 * almost never does. Measured on bag 6: 359 nodes per outer iteration spread
 * over 7 levels, so the levels are roughly 1, 2, 6, 14, 34, 83, 203 wide. Every
 * one of them fits in a single block-wide round at any of these sizes, which is
 * why the search wall is flat across 768/384/256 (13.88 / 13.37 / 13.58 ms) and
 * only degrades once the widest level no longer fits (128 -> 14.96, 64 ->
 * 19.48). 384 is the smallest size that still covers the 203-wide level with
 * room to spare, and it wins the ~3.7% back from the two costs that scale with
 * the block rather than with the work: the per-level exclusive scan, which
 * thread 0 walks over all kThreads entries, and Scratch::scan itself, which is
 * 4 bytes of shared memory per thread taken away from the Way.
 *
 * All four sizes produce identical digests, as they must -- the block size is
 * strategy, and none of it reaches the policy.
 */
const uint32_t kThreads = 384;
const uint32_t kWarps = kThreads / 32;

/// Free-slot sentinel of the Way's edge-hash table. Must match the value the
/// host writes into the image it uploads.
const uint64_t kEmptyHash = ~uint64_t(0);

/// What the kernel reports about why it stopped.
const uint32_t kStatusStop = 0;    ///< seeds ran out, or the horizon was reached
const uint32_t kStatusClosed = 1;  ///< the Way closed its loop
const uint32_t kStatusCapped = 2;  ///< out of iterations or Way capacity; relaunch

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
 * though the traces are produced in parallel.
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
/*                                Kernel state                                 */
/* -------------------------------------------------------------------------- */

struct KernelArgs {
  /* ---- the search, unchanged from the per-iteration backend ---- */
  L::EdgeSoA e;
  L::SearchConsts p;
  ccs::GridView g;
  L::Trace *frontA;
  L::Trace *frontB;
  uint64_t *childKeys;
  uint32_t *childCount;
  uint32_t *childOffset;
  uint32_t maxFrontier;

  /**
   * @brief Exact double copies of the two edge quantities Way's OWN
   * bookkeeping consumes: the midpoint (Way::closesLoop) and the length
   * (Way::addEdge). The search runs in float; these two do not, because the
   * host does them in double and both feed threshold comparisons.
   */
  const double *edgeMidXD;
  const double *edgeMidYD;
  const double *edgeLenD;

  /* ---- the Way, device-owned and mutable for the whole callback ---- */
  L::Vec2 *wayMid;
  L::Vec2 *wayNormal;
  L::Vec2 *waySegMin;
  L::Vec2 *waySegMax;
  uint64_t *wayHash;
  uint64_t *wayTable;
  uint32_t wayInitSize;  ///< elements the host uploaded
  uint32_t wayCap;       ///< elements the arrays can hold
  uint32_t tableCap;     ///< power of two
  double wayAvgD;        ///< Way::getAvgEdgeLen() on entry, in double
  double wayFrontX;      ///< way.front().midPoint(), in double
  double wayFrontY;
  double maxDistLoopClosure;
  uint32_t minLoopSize;
  bool tableOk;
  bool wayInShared;

  /* ---- outer-loop control ---- */
  uint32_t sizeAheadBase;  ///< way.sizeAheadOfCar() on entry
  int32_t maxHorizon;      ///< params.max_way_horizon_size; 0 means unlimited
  uint32_t maxIters;

  /* ---- results ---- */
  uint32_t *chosen;  ///< edge indices appended, in order
  uint32_t *result;  ///< [0] = number appended, [1] = status
};

/**
 * @brief The growing Way, as the block sees it.
 *
 * The pointers are into shared memory when the whole Way image fits there and
 * into device memory otherwise, which is the only difference between the two
 * cases -- everything downstream reads them without caring.
 */
struct WayState {
  L::Vec2 *mid;
  L::Vec2 *normal;
  L::Vec2 *segMin;
  L::Vec2 *segMax;
  uint64_t *hash;
  uint64_t *table;
  double avgD;    ///< the authority; Way keeps this in double
  double frontX;  ///< first midpoint, double, for closesLoop
  double frontY;
  double lastX;  ///< last appended midpoint, double, for closesLoop
  double lastY;
  uint32_t size;
  uint32_t mask;
  uint32_t tableOk;
  float avgF;  ///< (float)avgD, which is what the filters actually read
};

/// Block-wide scratch, reused by every outer iteration.
struct Scratch {
  uint32_t scan[kThreads];
  uint32_t frontierSize;
  uint32_t nextSize;
  uint32_t seedsEmpty;
  BestRec best[kWarps];
};

/// The Way as the shared policy predicates want it.
__device__ __forceinline__ L::WaySoA wayView(const WayState &sw) {
  L::WaySoA w;
  w.mid = sw.mid;
  w.normal = sw.normal;
  w.hash = sw.hash;
  w.size = sw.size;
  w.avgEdgeLen = sw.avgF;
  w.hashTable = sw.tableOk ? sw.table : nullptr;
  w.hashMask = sw.mask;
  w.hashEmpty = kEmptyHash;
  w.segMin = sw.segMin;
  w.segMax = sw.segMax;
  return w;
}

/* -------------------------------------------------------------------------- */
/*                          Way::addEdge, on the device                        */
/* -------------------------------------------------------------------------- */

/**
 * @brief Appends one edge to the device's Way. SINGLE THREAD.
 *
 * Reproduces, in order, what the host loop did between two launches:
 * WaySoAHost::push (the float mirror, the segment bounding box, the hash-table
 * insert) and Way::addEdge (the avgEdgeLen recurrence, in double, with the
 * divisor taken AFTER the push exactly as the original does).
 */
__device__ void appendWay(WayState &sw, const KernelArgs &a, uint32_t idx) {
  const uint32_t k = sw.size;

  const L::Vec2 m = a.e.mid[idx];
  const uint64_t h = a.e.hash[idx];
  sw.mid[k] = m;
  sw.normal[k] = a.e.normal[idx];
  sw.hash[k] = h;

  // Segment k is (mid[k-1], mid[k]); index 0 is never read, and the host still
  // writes the midpoint there, so do the same.
  if (k >= 1) {
    const L::Vec2 q = sw.mid[k - 1];
    sw.segMin[k] = ccs::vec2<scalar>(q.x < m.x ? q.x : m.x, q.y < m.y ? q.y : m.y);
    sw.segMax[k] = ccs::vec2<scalar>(q.x > m.x ? q.x : m.x, q.y > m.y ? q.y : m.y);
  } else {
    sw.segMin[k] = m;
    sw.segMax[k] = m;
  }

  // An Edge hash colliding with the sentinel would make the table lie, so the
  // host gives up on the table rather than answer wrongly and falls back to the
  // linear scan. Same here, for the rest of the callback.
  if (sw.tableOk) {
    if (h == kEmptyHash) {
      sw.tableOk = 0u;
    } else {
      uint32_t i = static_cast<uint32_t>(h) & sw.mask;
      while (sw.table[i] != kEmptyHash and sw.table[i] != h) i = (i + 1u) & sw.mask;
      sw.table[i] = h;
    }
  }

  sw.avgD += (a.edgeLenD[idx] - sw.avgD) / static_cast<double>(k + 1);
  sw.avgF = static_cast<float>(sw.avgD);

  sw.lastX = a.edgeMidXD[idx];
  sw.lastY = a.edgeMidYD[idx];
  if (k == 0) {
    sw.frontX = sw.lastX;
    sw.frontY = sw.lastY;
  }
  sw.size = k + 1;
}

/// Way::closesLoop, in double. SINGLE THREAD, called right after appendWay.
__device__ __forceinline__ bool wayClosesLoop(const WayState &sw, const KernelArgs &a) {
  if (sw.size < a.minLoopSize) return false;
  const double dx = sw.frontX - sw.lastX, dy = sw.frontY - sw.lastY;
  return dx * dx + dy * dy <= a.maxDistLoopClosure * a.maxDistLoopClosure;
}

/* -------------------------------------------------------------------------- */
/*                                   The BFS                                   */
/* -------------------------------------------------------------------------- */

/**
 * @brief Evaluates one node's candidates and returns its best three.
 *
 * \a trace is null for the seed query. Mirrors CpuFastSearch::findNextEdges:
 * grid query, discard filters, heuristic threshold, best max_search_options by
 * (heuristic, index).
 */
__device__ void findNextEdges(const KernelArgs &a, const L::WaySoA &w, const L::Trace *trace,
                              uint64_t &k0, uint64_t &k1, uint64_t &k2) {
  k0 = k1 = k2 = kNoKey;

  L::SearchContext c;
  c.hasActEdge = false;
  c.actEdgeMid = ccs::vec2<scalar>(0, 0);
  c.actEdgeNormal = ccs::vec2<scalar>(0, 0);
  c.actEdgeHash = 0;
  c.actPos = ccs::vec2<scalar>(0, 0);
  c.lastPos = ccs::vec2<scalar>(0, 0);

  const L::EdgeSoA &e = a.e;

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
 * BLOCK-WIDE: every thread must call it, and both what it returns and the flag
 * it sets are the same in every thread, so the caller can branch on them
 * without diverging. That uniformity is what makes the outer loop below legal.
 */
__device__ uint32_t bfsOuterIteration(const KernelArgs &a, const L::WaySoA &w, Scratch &s,
                                      bool &seedsEmpty) {
  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid / 32;
  const uint32_t lane = tid % 32;

  L::Trace *front = a.frontA;
  L::Trace *next = a.frontB;

  /* ---------------------------- seeds (level 0) --------------------------- */
  if (tid == 0) {
    uint64_t k0, k1, k2;
    findNextEdges(a, w, nullptr, k0, k1, k2);
    const uint64_t keys[3] = {k0, k1, k2};
    uint32_t n = 0;
    const ccs::Vec2T<scalar> wayBack = ccs::wayBackMid(w);
    for (uint32_t r = 0; r < 3 and r < static_cast<uint32_t>(a.p.max_search_options); ++r) {
      if (keys[r] == kNoKey) break;
      const uint32_t idx = keyIdx(keys[r]);
      L::Trace t;
      ccs::traceInit(t);
      const bool closesLoop = ccs::wayClosesLoopWith(w, a.e.mid[idx], wayBack, a.p);
      ccs::traceAppend(t, idx, keyHeur(keys[r]), a.e.len[idx], closesLoop);
      front[n++] = t;
    }
    s.frontierSize = n;
    s.seedsEmpty = (n == 0) ? 1u : 0u;
  }
  __syncthreads();

  if (s.seedsEmpty) {
    seedsEmpty = true;
    return 0;
  }
  seedsEmpty = false;

  /* ------------------------------- the BFS -------------------------------- */
  // The host's FIFO visits traces in strict level order, and within a level in
  // parent order then child rank. `order` below reproduces exactly that
  // numbering, which is what makes the tie-break reproducible.
  BestRec myBest;
  myBest.size = front[0].size;
  myBest.sumHeur = front[0].sumHeur;
  myBest.order = 0;
  myBest.firstEdge = ccs::traceFirst(front[0]);

  uint32_t orderBase = 1;  // 0 is taken by the initial best

  for (uint32_t level = 0; level < static_cast<uint32_t>(a.p.max_search_tree_height); ++level) {
    const uint32_t nFront = s.frontierSize;
    if (nFront == 0) break;

    // ONE THREAD per frontier node, and the frontier is DENSE. Pass A evaluates
    // each node and records how many children it wants; the scan turns those
    // counts into write offsets, in node order, so the next frontier comes out
    // in exactly the host FIFO's order; pass B writes the children there.
    for (uint32_t node = tid; node < nFront; node += kThreads) {
      const L::Trace t = front[node];
      const uint32_t order = orderBase + node;

      bool terminal = (t.size >= static_cast<uint32_t>(a.p.max_search_tree_height));
      uint64_t k0 = kNoKey, k1 = kNoKey, k2 = kNoKey;
      if (not terminal) findNextEdges(a, w, &t, k0, k1, k2);

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
      s.scan[tid] = localSum;
      __syncthreads();

      // kThreads entries: a sequential scan by one thread is a few hundred
      // cycles, and cheaper than the syncs a parallel scan of this size needs.
      if (tid == 0) {
        uint32_t running = 0;
        for (uint32_t i = 0; i < kThreads; ++i) {
          const uint32_t v = s.scan[i];
          s.scan[i] = running;
          running += v;
        }
        s.nextSize = running;
      }
      __syncthreads();

      uint32_t running = s.scan[tid];
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
        const bool closesLoop = ccs::wayClosesLoopWith(w, a.e.mid[idx], actPos, a.p);
        ccs::traceAppend(aux, idx, keyHeur(key), a.e.len[idx], closesLoop);
        next[base + r] = aux;
      }
    }
    __syncthreads();

    orderBase += nFront;
    const uint32_t produced = s.nextSize;
    if (tid == 0) s.frontierSize = produced;
    __syncthreads();

    L::Trace *tmp = front;
    front = next;
    next = tmp;
  }

  /* --------------------------- reduce the best ---------------------------- */
  // Warp reduction first, then across warps through shared memory. Unlike the
  // per-iteration backend, EVERY thread finishes the cross-warp reduction: the
  // outer loop needs the answer to be uniform, not just known to thread 0.
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    BestRec o;
    o.size = __shfl_down_sync(0xffffffffu, myBest.size, off);
    o.sumHeur = __shfl_down_sync(0xffffffffu, myBest.sumHeur, off);
    o.order = __shfl_down_sync(0xffffffffu, myBest.order, off);
    o.firstEdge = __shfl_down_sync(0xffffffffu, myBest.firstEdge, off);
    if (better(o, myBest)) myBest = o;
  }
  if (lane == 0) s.best[warp] = myBest;
  __syncthreads();

  BestRec r = s.best[0];
  for (uint32_t i = 1; i < kWarps; ++i)
    if (better(s.best[i], r)) r = s.best[i];

  // s is about to be reused by the next outer iteration; nobody may write
  // s.frontierSize before everyone has finished reading s.best.
  __syncthreads();
  return r.firstEdge;
}

/* -------------------------------------------------------------------------- */
/*                                   Kernel                                    */
/* -------------------------------------------------------------------------- */

/**
 * @brief The whole callback: grow the Way until it stops growing.
 *
 * Launched as a SINGLE block, for the same reason the per-iteration backend is
 * -- one outer iteration is ~360 nodes of a dozen candidates each, far too
 * little to fill 8 SMs -- and here there is a second, harder reason: the outer
 * loop is strictly sequential, so a grid would need a grid-wide barrier per
 * appended midpoint. One block gets that barrier for free from __syncthreads.
 */
__global__ __launch_bounds__(kThreads) void oneShotKernel(KernelArgs a) {
  // The Way, staged in shared memory. wayIntersectsWith() and wayContainsEdge()
  // walk ALL of it for every candidate, so at ~48 midpoints and ~12 candidates
  // per node this is by far the most re-read data in the kernel -- and unlike
  // the per-iteration backend, it is re-read for the WHOLE callback rather than
  // for one launch, so staging it pays ~18x more here. The reservation is for
  // the Way's final capacity, not its current size, because it grows in place.
  extern __shared__ unsigned char sRaw[];
  __shared__ Scratch s;
  __shared__ WayState sw;
  __shared__ uint32_t sN;       ///< midpoints appended so far
  __shared__ uint32_t sStatus;  ///< why the loop stopped

  const uint32_t tid = threadIdx.x;

  if (tid == 0) {
    if (a.wayInShared) {
      L::Vec2 *m = reinterpret_cast<L::Vec2 *>(sRaw);
      sw.mid = m;
      sw.normal = m + a.wayCap;
      sw.segMin = sw.normal + a.wayCap;
      sw.segMax = sw.segMin + a.wayCap;
      sw.hash = reinterpret_cast<uint64_t *>(sw.segMax + a.wayCap);
      sw.table = sw.hash + a.wayCap;
    } else {
      sw.mid = a.wayMid;
      sw.normal = a.wayNormal;
      sw.segMin = a.waySegMin;
      sw.segMax = a.waySegMax;
      sw.hash = a.wayHash;
      sw.table = a.wayTable;
    }
    sw.size = a.wayInitSize;
    sw.mask = a.tableCap - 1u;
    sw.tableOk = a.tableOk ? 1u : 0u;
    sw.avgD = a.wayAvgD;
    sw.avgF = static_cast<float>(a.wayAvgD);
    sw.frontX = a.wayFrontX;
    sw.frontY = a.wayFrontY;
    sw.lastX = 0.0;  // only read after an append, which always sets it
    sw.lastY = 0.0;
    sN = 0;
    sStatus = kStatusStop;
  }
  __syncthreads();

  if (a.wayInShared) {
    for (uint32_t i = tid; i < a.wayInitSize; i += kThreads) {
      sw.mid[i] = a.wayMid[i];
      sw.normal[i] = a.wayNormal[i];
      sw.segMin[i] = a.waySegMin[i];
      sw.segMax[i] = a.waySegMax[i];
      sw.hash[i] = a.wayHash[i];
    }
    for (uint32_t i = tid; i < a.tableCap; i += kThreads) sw.table[i] = a.wayTable[i];
    __syncthreads();
  }

  /* ----------------------------- the outer loop --------------------------- */
  // This is WayComputer's loop, verbatim in structure: search, then decide
  // whether to append, then append and re-check the stopping conditions. Every
  // condition it branches on is block-uniform, so no thread ever leaves the
  // loop alone and the __syncthreads() inside the BFS stay collective.
  for (;;) {
    const L::WaySoA w = wayView(sw);

    bool seedsEmpty = false;
    const uint32_t idx = bfsOuterIteration(a, w, s, seedsEmpty);
    if (seedsEmpty) break;

    // The host checked way.sizeAheadOfCar() BEFORE appending, and sizeToCar_
    // does not move during the loop, so this is the entry value plus the count.
    if (a.maxHorizon > 0 and (a.sizeAheadBase + sN) > static_cast<uint32_t>(a.maxHorizon)) break;

    // Out of room, or out of the iteration budget that keeps rclcpp::ok()
    // reachable. Neither is a stop: the host replays what we have and relaunches.
    if (sN >= a.maxIters or sw.size >= a.wayCap) {
      if (tid == 0) sStatus = kStatusCapped;
      break;
    }

    if (tid == 0) {
      appendWay(sw, a, idx);
      a.chosen[sN] = idx;
      sN = sN + 1u;
      if (wayClosesLoop(sw, a)) sStatus = kStatusClosed;
    }
    __syncthreads();

    if (sStatus == kStatusClosed) break;
  }

  if (tid == 0) {
    a.result[0] = sN;
    a.result[1] = sStatus;
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

/// Where the callback's time goes, split host/device, in the same shape
/// CudaSearch reports so the two are directly comparable.
struct Phases {
  double kernelMs = 0;
  double soaCpu = 0, soaWall = 0;
  double uploadCpu = 0, uploadWall = 0;
  double iterCpu = 0, iterWall = 0;
  uint64_t callbacks = 0, launches = 0;
};
Phases g_phases;

}  // namespace

struct CudaOneShotSearch::Device {
  L::Vec2 *edgeMid = nullptr, *edgeNormal = nullptr;
  scalar *edgeLen = nullptr;
  uint64_t *edgeHash = nullptr;
  uint32_t capEdgeMid = 0, capEdgeNormal = 0, capEdgeLen = 0, capEdgeHash = 0;

  double *edgeMidXD = nullptr, *edgeMidYD = nullptr, *edgeLenD = nullptr;
  uint32_t capMidXD = 0, capMidYD = 0, capLenD = 0;

  L::Vec2 *wayMid = nullptr, *wayNormal = nullptr;
  L::Vec2 *waySegMin = nullptr, *waySegMax = nullptr;
  uint64_t *wayHash = nullptr, *wayTable = nullptr;
  uint32_t capWayMid = 0, capWayNormal = 0, capSegMin = 0, capSegMax = 0;
  uint32_t capWayHash = 0, capTable = 0;

  double *gx = nullptr, *gy = nullptr;
  uint32_t *cellStart = nullptr, *items = nullptr;
  uint32_t capGx = 0, capGy = 0, capItems = 0, capCellStart = 0;

  L::Trace *frontA = nullptr, *frontB = nullptr;
  uint64_t *childKeys = nullptr;
  uint32_t *childCount = nullptr;
  uint32_t *childOffset = nullptr;

  /// Mapped-pinned: two words and a short index list, read once per launch.
  uint32_t *chosen = nullptr;
  uint32_t *result = nullptr;

  cudaEvent_t evStart = nullptr, evStop = nullptr;
};

namespace {

bool cudaOk(cudaError_t e, const char *what) {
  if (e == cudaSuccess) return true;
  RCLCPP_ERROR(rclcpp::get_logger("cuda_cone_stellation"), "CUDA %s failed: %s", what,
               cudaGetErrorString(e));
  return false;
}

/// Grows a DEVICE buffer. See cuda_search.cu for why this is plain device
/// memory rather than mapped-pinned or managed: the Way is the hottest array in
/// the kernel, and reaching it over the coherent fabric made the frontier scan
/// latency-bound.
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

uint32_t pow2AtLeast(uint32_t v) {
  uint32_t c = 16;
  while (c < v) c <<= 1;
  return c;
}

/// The host's WaySoAHost::probeInsert, so the image the device starts from is
/// built by the same rule the device continues with.
void probeInsert(std::vector<uint64_t> &t, uint64_t h) {
  const uint32_t mask = static_cast<uint32_t>(t.size() - 1);
  uint32_t i = static_cast<uint32_t>(h) & mask;
  while (t[i] != kEmptyHash) {
    if (t[i] == h) return;
    i = (i + 1) & mask;
  }
  t[i] = h;
}

}  // namespace

bool CudaOneShotSearch::deviceAvailable() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) return false;
  return n > 0;
}

bool CudaOneShotSearch::supportsParams(const Params::WayComputer &params, const char **why) {
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

void CudaOneShotSearch::reportBackendDetail() const {
  if (g_phases.callbacks == 0) return;
  const double n = double(g_phases.callbacks);
  RCLCPP_INFO(rclcpp::get_logger("cuda_cone_stellation"),
              "[cuda-one-shot-search phases/callback] soa %.3f ms cpu | upload %.3f | iters "
              "%.3f cpu / %.3f wall over %.2f launches | kernel %.3f ms (%.0f%% of iter wall), "
              "%.1f us/launch",
              g_phases.soaCpu / n, g_phases.uploadCpu / n, g_phases.iterCpu / n,
              g_phases.iterWall / n, double(g_phases.launches) / n, g_phases.kernelMs / n,
              g_phases.iterWall > 0 ? 100.0 * g_phases.kernelMs / g_phases.iterWall : 0.0,
              g_phases.launches ? 1000.0 * g_phases.kernelMs / double(g_phases.launches) : 0.0);
}

CudaOneShotSearch::CudaOneShotSearch(const Params::WayComputer &params)
    : wayParams_(params.way), dev_(new Device) {
  cudaMalloc(&this->dev_->frontA, size_t(kMaxFrontier) * 3 * sizeof(L::Trace));
  cudaMalloc(&this->dev_->frontB, size_t(kMaxFrontier) * 3 * sizeof(L::Trace));
  cudaMalloc(&this->dev_->childKeys, size_t(kMaxFrontier) * 3 * sizeof(uint64_t));
  cudaMalloc(&this->dev_->childCount, size_t(kMaxFrontier) * sizeof(uint32_t));
  cudaMalloc(&this->dev_->childOffset, size_t(kMaxFrontier) * sizeof(uint32_t));
  cudaHostAlloc(&this->dev_->result, 2 * sizeof(uint32_t), cudaHostAllocMapped);
  cudaHostAlloc(&this->dev_->chosen, size_t(kMaxItersPerLaunch) * sizeof(uint32_t),
                cudaHostAllocMapped);
  cudaEventCreate(&this->dev_->evStart);
  cudaEventCreate(&this->dev_->evStop);
}

CudaOneShotSearch::~CudaOneShotSearch() {
  Device *d = this->dev_;
  cudaFree(d->edgeMid);
  cudaFree(d->edgeNormal);
  cudaFree(d->edgeLen);
  cudaFree(d->edgeHash);
  cudaFree(d->edgeMidXD);
  cudaFree(d->edgeMidYD);
  cudaFree(d->edgeLenD);
  cudaFree(d->wayMid);
  cudaFree(d->wayNormal);
  cudaFree(d->waySegMin);
  cudaFree(d->waySegMax);
  cudaFree(d->wayHash);
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
  cudaFreeHost(d->chosen);
  if (d->evStart) cudaEventDestroy(d->evStart);
  if (d->evStop) cudaEventDestroy(d->evStop);
  delete d;
}

CudaOneShotSearch::L::SearchConsts CudaOneShotSearch::makeConsts(
    const Params::WayComputer::Search &params) const {
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
  double midXD, midYD;
};

EdgeFields edgeFields(const Edge &e) {
  const Point mid = e.midPoint();
  const Vector normal = e.normal();
  EdgeFields f;
  f.mid = ccs::vec2<scalar>(static_cast<scalar>(mid.x), static_cast<scalar>(mid.y));
  f.normal = ccs::vec2<scalar>(static_cast<scalar>(normal.x), static_cast<scalar>(normal.y));
  f.hash = std::hash<Edge>{}(e);
  f.midXD = mid.x;
  f.midYD = mid.y;
  return f;
}

}  // namespace

void CudaOneShotSearch::buildEdgeSoA(const std::vector<Edge> &edges, double searchRadius) {
  this->edgeSoa_.clear();
  this->edgeSoa_.reserve(edges.size());
  this->gridX_.clear();
  this->gridY_.clear();
  this->gridX_.reserve(edges.size());
  this->gridY_.reserve(edges.size());
  this->edgeMidXD_.clear();
  this->edgeMidYD_.clear();
  this->edgeLenD_.clear();
  this->edgeMidXD_.reserve(edges.size());
  this->edgeMidYD_.reserve(edges.size());
  this->edgeLenD_.reserve(edges.size());

  for (const Edge &e : edges) {
    const EdgeFields f = edgeFields(e);
    this->edgeSoa_.push(f.mid, f.normal, static_cast<scalar>(e.len), f.hash);
    // The grid takes the FLOAT midpoint widened back, because its radius test
    // has to be the one the reference backend performs.
    this->gridX_.push_back(static_cast<double>(f.mid.x));
    this->gridY_.push_back(static_cast<double>(f.mid.y));
    // Way's own bookkeeping takes the ORIGINAL double. Different quantity.
    this->edgeMidXD_.push_back(f.midXD);
    this->edgeMidYD_.push_back(f.midYD);
    this->edgeLenD_.push_back(e.len);
  }

  this->grid_.build(this->gridX_.data(), this->gridY_.data(),
                    static_cast<uint32_t>(this->gridX_.size()), searchRadius);
}

void CudaOneShotSearch::buildWayImage(const Way &way, uint32_t cap) {
  this->wayCap_ = cap;

  this->wayMid_.clear();
  this->wayNormal_.clear();
  this->wayHash_.clear();
  this->waySegMin_.clear();
  this->waySegMax_.clear();

  for (const Edge &e : way.edges()) {
    const EdgeFields f = edgeFields(e);
    const size_t k = this->wayMid_.size();
    this->wayMid_.push_back(f.mid);
    this->wayNormal_.push_back(f.normal);
    this->wayHash_.push_back(f.hash);
    if (k >= 1) {
      const L::Vec2 &a = this->wayMid_[k - 1], &b = this->wayMid_[k];
      this->waySegMin_.push_back(
          ccs::vec2<scalar>(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y));
      this->waySegMax_.push_back(
          ccs::vec2<scalar>(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y));
    } else {
      this->waySegMin_.push_back(f.mid);  // index 0 is never read
      this->waySegMax_.push_back(f.mid);
    }
  }

  // Sized for the Way's FINAL capacity, not its current size, since the device
  // keeps inserting into it. 2x rather than the host builder's 4x: the table
  // has to live in shared memory alongside the Way, the bound only binds if the
  // Way grows to `cap` (it does not -- steady state is ~48 against a cap of a
  // few hundred), and the table's capacity does not affect WHICH hashes it
  // reports as members, only how long the probes are.
  this->tableCap_ = pow2AtLeast(2u * cap);
  this->wayTable_.assign(this->tableCap_, kEmptyHash);
  this->wayTableOk_ = true;
  for (const uint64_t &h : this->wayHash_) {
    // An Edge hash colliding with the sentinel would make the table lie, so
    // fall back to the linear scan rather than answer wrongly.
    if (h == kEmptyHash) {
      this->wayTableOk_ = false;
      break;
    }
    probeInsert(this->wayTable_, h);
  }
}

void CudaOneShotSearch::syncEdgesToDevice() {
  Device *d = this->dev_;
  const L::EdgeSoA h = this->edgeSoa_.view();
  ensure(d->edgeMid, d->capEdgeMid, h.size);
  ensure(d->edgeNormal, d->capEdgeNormal, h.size);
  ensure(d->edgeLen, d->capEdgeLen, h.size);
  ensure(d->edgeHash, d->capEdgeHash, h.size);
  ensure(d->edgeMidXD, d->capMidXD, h.size);
  ensure(d->edgeMidYD, d->capMidYD, h.size);
  ensure(d->edgeLenD, d->capLenD, h.size);

  upload(d->edgeMid, h.mid, h.size);
  upload(d->edgeNormal, h.normal, h.size);
  upload(d->edgeLen, h.len, h.size);
  upload(d->edgeHash, h.hash, h.size);
  upload(d->edgeMidXD, this->edgeMidXD_.data(), h.size);
  upload(d->edgeMidYD, this->edgeMidYD_.data(), h.size);
  upload(d->edgeLenD, this->edgeLenD_.data(), h.size);

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

void CudaOneShotSearch::syncWayToDevice() {
  Device *d = this->dev_;
  const uint32_t cap = this->wayCap_;
  const uint32_t n = static_cast<uint32_t>(this->wayMid_.size());

  // Allocated to the FINAL capacity: the kernel appends into the tail, so the
  // room has to be there before it starts.
  ensure(d->wayMid, d->capWayMid, cap);
  ensure(d->wayNormal, d->capWayNormal, cap);
  ensure(d->waySegMin, d->capSegMin, cap);
  ensure(d->waySegMax, d->capSegMax, cap);
  ensure(d->wayHash, d->capWayHash, cap);
  ensure(d->wayTable, d->capTable, this->tableCap_);

  upload(d->wayMid, this->wayMid_.data(), n);
  upload(d->wayNormal, this->wayNormal_.data(), n);
  upload(d->waySegMin, this->waySegMin_.data(), n);
  upload(d->waySegMax, this->waySegMax_.data(), n);
  upload(d->wayHash, this->wayHash_.data(), n);
  upload(d->wayTable, this->wayTable_.data(), this->tableCap_);
}

ISearch::Result CudaOneShotSearch::computeWay(const std::vector<Edge> &edges,
                                              const Params::WayComputer::Search &params,
                                              Way &way) {
  Result res;
  const CallbackTimer timer(this->stats_);

  way.trimByLocal();

  const L::SearchConsts p = this->makeConsts(params);

  double c0 = cpuMs(), w0 = wallMs();
  this->buildEdgeSoA(edges, static_cast<double>(p.search_radius));
  g_phases.soaCpu += cpuMs() - c0;
  g_phases.soaWall += wallMs() - w0;

  c0 = cpuMs();
  w0 = wallMs();
  this->syncEdgesToDevice();
  g_phases.uploadCpu += cpuMs() - c0;
  g_phases.uploadWall += wallMs() - w0;

  ++g_phases.callbacks;

  Device *d = this->dev_;
  const L::EdgeSoA he = this->edgeSoa_.view();
  const ccs::GridView hg = this->grid_.view();

  // The loop runs more than once only when a launch came back capped, which the
  // steady state never does -- ~18 appends against a cap of 512.
  for (;;) {
    const double ic = cpuMs(), iw = wallMs();
    ++g_phases.launches;

    // The Way cannot outgrow the edge set plus what it already holds: filter 3
    // refuses an edge already in the Way. "Cannot in practice" is not a bound
    // though, so exceeding it comes back as kStatusCapped rather than
    // corrupting memory.
    const uint32_t wayNow = static_cast<uint32_t>(way.size());
    const uint32_t cap = wayNow + he.size + 8u;
    this->buildWayImage(way, cap);
    this->syncWayToDevice();

    KernelArgs a;
    a.e.mid = d->edgeMid;
    a.e.normal = d->edgeNormal;
    a.e.len = d->edgeLen;
    a.e.hash = d->edgeHash;
    a.e.size = he.size;

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
    a.maxFrontier = kMaxFrontier * 3;

    a.edgeMidXD = d->edgeMidXD;
    a.edgeMidYD = d->edgeMidYD;
    a.edgeLenD = d->edgeLenD;

    a.wayMid = d->wayMid;
    a.wayNormal = d->wayNormal;
    a.waySegMin = d->waySegMin;
    a.waySegMax = d->waySegMax;
    a.wayHash = d->wayHash;
    a.wayTable = d->wayTable;
    a.wayInitSize = wayNow;
    a.wayCap = cap;
    a.tableCap = this->tableCap_;
    a.tableOk = this->wayTableOk_;
    a.wayAvgD = way.getAvgEdgeLen();
    a.wayFrontX = wayNow ? way.front().midPoint().x : 0.0;
    a.wayFrontY = wayNow ? way.front().midPoint().y : 0.0;
    a.maxDistLoopClosure = this->wayParams_.max_dist_loop_closure;
    a.minLoopSize = MIN_LOOP_SIZE;

    a.sizeAheadBase = way.sizeAheadOfCar();
    a.maxHorizon = params.max_way_horizon_size;
    a.maxIters = kMaxItersPerLaunch;

    a.chosen = d->chosen;
    a.result = d->result;

    // 32 KB ceiling, leaving the ~3.6 KB of static shared this kernel also
    // needs comfortably inside the 48 KB a block gets by default. At the
    // measured steady state (|way| ~ 48, ~120 candidate edges) the reservation
    // is ~15 KB, so the Way stays in shared for the whole callback; a bigger
    // map simply runs against device memory instead of failing.
    const size_t wayBytes = size_t(cap) * (4 * sizeof(L::Vec2) + sizeof(uint64_t)) +
                            size_t(this->tableCap_) * sizeof(uint64_t);
    a.wayInShared = (wayBytes <= 32u * 1024u);
    const size_t sharedBytes = a.wayInShared ? wayBytes : 0;

    cudaEventRecord(d->evStart, 0);
    oneShotKernel<<<1, kThreads, sharedBytes>>>(a);
    cudaEventRecord(d->evStop, 0);
    if (not cudaOk(cudaGetLastError(), "launch") or
        not cudaOk(cudaDeviceSynchronize(), "sync")) {
      res.loopClosed = false;
      res.wayToPublish = way;
      return res;
    }

    float kms = 0;
    if (cudaEventElapsedTime(&kms, d->evStart, d->evStop) == cudaSuccess) g_phases.kernelMs += kms;

    const uint32_t nChosen = d->result[0];
    const uint32_t status = d->result[1];

    g_phases.iterCpu += cpuMs() - ic;
    g_phases.iterWall += wallMs() - iw;

    // Replay into the REAL Way, with the real Way::addEdge, so the published
    // Way is built by the same code every other backend builds it with. The
    // device's Way is a mirror that answers the filters, never the output.
    for (uint32_t i = 0; i < nChosen; ++i) {
      ++this->stats_.outerIters;
      this->stats_.waySizeSum += way.size();
      way.addEdge(edges[d->chosen[i]]);
    }

    if (status == kStatusClosed) {
      res.wayToPublish = way.restructureClosure();
      res.loopClosed = true;
      return res;
    }
    if (status != kStatusCapped) break;
    if (not rclcpp::ok()) break;
  }

  res.loopClosed = false;
  res.wayToPublish = way;
  return res;
}
