/**
 * @file search_types.hpp
 * @brief LAYOUT: the flat data the search operates on, and nothing else.
 *
 * This file and search_policy.hpp were one file (soa.hpp) that mixed three
 * different kinds of decision. Separating them is what lets the backends
 * diverge where they should and forbids them from diverging where they must
 * not:
 *
 *   LAYOUT   (here)               how the data is arranged. Shared because it
 *                                 costs nothing to share, but a backend that
 *                                 wants its own arrangement is free to build
 *                                 one -- nothing downstream depends on these
 *                                 types except through the policy functions.
 *
 *   POLICY   (search_policy.hpp)  what a valid way IS: the discard filters,
 *                                 the heuristic, loop closure. MUST be shared.
 *                                 Two backends that disagree here do not
 *                                 disagree by a rounding error, they drive
 *                                 different racing lines.
 *
 *   STRATEGY (each backend)       how the space is explored: the spatial
 *                                 index, the shape of the tree walk, what runs
 *                                 in parallel. MUST NOT be shared -- a CPU and
 *                                 a GPU want genuinely different answers, and
 *                                 forcing one shape on both is how you get a
 *                                 GPU port that inherits a host bottleneck.
 *
 * Everything here is templated on the scalar type. It used to be a single
 * package-wide typedef fixed by one CMake flag, which meant the CPU backend
 * and the device backend could not pick differently -- the one coupling that
 * the ISearch split did not actually remove. Now each backend instantiates
 * these with the scalar it wants.
 *
 * Nothing in this header may depend on ROS, Eigen, or any pointer-chasing
 * container: it is compiled by both the host and the device backend.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * @brief Marks everything that must remain callable from both backends.
 */
#if defined(__CUDACC__)
#define CCS_HD __host__ __device__
#else
#define CCS_HD
#endif

/**
 * @brief Upper bound on max_search_tree_height.
 *
 * This no longer sizes anything -- a Trace does not store its path (see
 * TraceT). It remains the cap the search clamps max_search_tree_height to, and
 * the bound the device backend sizes its frontier against.
 */
#ifndef CCS_MAX_TRACE_LEN
#define CCS_MAX_TRACE_LEN 16
#endif

namespace ccs {

/* -------------------------------------------------------------------------- */
/*                                   Vector                                    */
/* -------------------------------------------------------------------------- */

template <typename S>
struct Vec2T {
  S x, y;
};

template <typename S>
CCS_HD inline Vec2T<S> vec2(S x, S y) {
  Vec2T<S> v;
  v.x = x;
  v.y = y;
  return v;
}

/* -------------------------------------------------------------------------- */
/*                                    Data                                     */
/* -------------------------------------------------------------------------- */

/**
 * @brief One callback's candidate edges, flattened. An index into these arrays
 * is the \a edgeInd the search passes around, and it matches the index into the
 * std::vector<Edge> the host built this from.
 */
template <typename S>
struct EdgeSoAT {
  const Vec2T<S> *mid;  ///< midpoint, local frame
  const Vec2T<S> *normal;  ///< Vector(n0, n1).rotClock()
  const S *len;         ///< edge length (global frame, as the original computes it)
  const uint64_t *hash; ///< Edge hash, built from the two Node ids
  uint32_t size;
};

/**
 * @brief The Way so far, in path order. Grows by exactly one element per outer
 * iteration of computeWay.
 */
template <typename S>
struct WaySoAT {
  const Vec2T<S> *mid;
  const Vec2T<S> *normal;
  const uint64_t *hash;
  uint32_t size;
  S avgEdgeLen;

  /**
   * @brief Optional indexes over the Way, both purely accelerative.
   *
   * The two Way predicates are O(|way|) per CANDIDATE and, with |way| ~ 48 and
   * ~12 candidates per node, they are most of the search's inner work -- 33% of
   * the host backend's time was filter 6 alone, measured. These let the same
   * predicates answer the same question without the scan. Leaving them null is
   * always valid and always gives the identical answer; it is the frozen
   * reference backend's choice.
   *
   * hashTable: open-addressed, power-of-two, `hashEmpty` marks a free slot.
   * segMin/segMax: bounding box of segment (mid[k], mid[k-1]), indexed by k,
   * valid for k in [1, size-1].
   */
  const uint64_t *hashTable;
  uint32_t hashMask;      ///< table size - 1
  uint64_t hashEmpty;     ///< sentinel occupying free slots
  const Vec2T<S> *segMin;
  const Vec2T<S> *segMax;
};

/**
 * @brief The search thresholds, flattened out of Params.
 */
template <typename S>
struct SearchConstsT {
  S search_radius;
  S max_angle_diff;
  S edge_len_diff_factor;
  S max_next_heuristic;
  /**
   * @brief Deliberately float, not S. The original computes
   * `(1 - heur_dist_ponderation)` in float because the parameter is float, and
   * only then widens. Storing it as S changes that subtraction's rounding and
   * therefore the heuristic, and therefore the chosen path.
   */
  float heur_dist_ponderation;
  bool allow_intersection;
  int max_search_tree_height;
  int max_search_options;

  // From Params::WayComputer::Way -- needed by the loop-closure predicate.
  S max_dist_loop_closure;
  S max_angle_diff_loop_closure;
  uint32_t min_loop_size;
};

/**
 * @brief An edge path in the tree search.
 *
 * The original was a shared_ptr chain so that cloning was O(1). Here it is a
 * value: cloning is a copy and there is no allocation on either backend.
 *
 * It does NOT store the path. It used to -- uint32_t edgeInd[CCS_MAX_TRACE_LEN]
 * -- but only three positions were ever read: edgeInd[0] (traceFirst, which is
 * what the search ultimately returns), edgeInd[size-1] (traceLast) and
 * edgeInd[size-2] (the previous position, for the direction vector). The other
 * thirteen were written every append and never looked at.
 *
 * Dropping them takes a Trace from 80 bytes to 28, which matters most on the
 * device: a dynamically indexed local array cannot live in registers, so every
 * append and every copy went to local memory, and the kernel copies two Traces
 * per node. It also shrinks the host backends' BFS queue by the same factor.
 */
template <typename S>
struct TraceT {
  uint32_t first;  ///< edgeInd[0]
  uint32_t prev;   ///< edgeInd[size - 2], valid once size >= 2
  uint32_t last;   ///< edgeInd[size - 1]
  uint32_t size;
  S sumHeur;
  S avgEdgeLen;
  bool loopClosed;
};

template <typename S>
CCS_HD inline void traceInit(TraceT<S> &t) {
  t.first = 0;
  t.prev = 0;
  t.last = 0;
  t.size = 0;
  t.sumHeur = S(0);
  t.avgEdgeLen = S(0);
  t.loopClosed = false;
}

/**
 * @brief Appends one edge, mirroring Trace::Connection's constructor exactly.
 *
 * sumHeur is accumulated left to right; the original recursed right to left,
 * but floating point addition is commutative so the two agree bit for bit.
 */
template <typename S>
CCS_HD inline void traceAppend(TraceT<S> &t, uint32_t edgeInd, S heur, S edgeLen, bool loopClosed) {
  if (t.size == 0) {
    t.avgEdgeLen = edgeLen;
    t.loopClosed = loopClosed;
  } else {
    t.avgEdgeLen = t.avgEdgeLen + ((edgeLen - t.avgEdgeLen) / S(t.size + 1));
    t.loopClosed = t.loopClosed or loopClosed;
  }
  t.sumHeur = t.sumHeur + heur;
  if (t.size == 0) t.first = edgeInd;
  t.prev = t.last;
  t.last = edgeInd;
  t.size++;
}

/// Index of the first edge of the Trace -- what treeSearch ultimately returns.
template <typename S>
CCS_HD inline uint32_t traceFirst(const TraceT<S> &t) {
  return t.first;
}

/// Index of the last edge appended.
template <typename S>
CCS_HD inline uint32_t traceLast(const TraceT<S> &t) {
  return t.last;
}

/// Index of the edge before the last, i.e. the old edgeInd[size - 2]. Only
/// meaningful when size >= 2, which is what every caller already checks.
template <typename S>
CCS_HD inline uint32_t tracePrev(const TraceT<S> &t) {
  return t.prev;
}

/**
 * @brief Everything findNextEdges derives once, before it looks at any
 * candidate. On a device backend this is what gets broadcast to the block.
 */
template <typename S>
struct SearchContextT {
  bool hasActEdge;
  Vec2T<S> actEdgeMid;
  Vec2T<S> actEdgeNormal;
  uint64_t actEdgeHash;
  Vec2T<S> actPos;
  Vec2T<S> lastPos;
  Vec2T<S> dir;
  S avgEdgeLen;          ///< combinedAvgEdgeLen, hoisted out of filter 5
  bool hasTrace;         ///< the original's `actTrace != nullptr`
  bool traceLoopClosed;
  bool sideCheckEnabled; ///< the original's `(way.size() >= 2 or actTrace)`
};

/* -------------------------------------------------------------------------- */
/*                             Host-side builders                              */
/* -------------------------------------------------------------------------- */

/**
 * @brief Owns the storage behind an EdgeSoA. Rebuilt once per callback, which
 * on Orin is free (integrated memory, no PCIe copy).
 */
template <typename S>
class EdgeSoAHostT {
 public:
  void clear() {
    mid_.clear();
    normal_.clear();
    len_.clear();
    hash_.clear();
  }

  void reserve(size_t n) {
    mid_.reserve(n);
    normal_.reserve(n);
    len_.reserve(n);
    hash_.reserve(n);
  }

  void push(Vec2T<S> mid, Vec2T<S> normal, S len, uint64_t hash) {
    mid_.push_back(mid);
    normal_.push_back(normal);
    len_.push_back(len);
    hash_.push_back(hash);
  }

  size_t size() const { return mid_.size(); }

  EdgeSoAT<S> view() const {
    EdgeSoAT<S> v;
    v.mid = mid_.data();
    v.normal = normal_.data();
    v.len = len_.data();
    v.hash = hash_.data();
    v.size = static_cast<uint32_t>(mid_.size());
    return v;
  }

 private:
  std::vector<Vec2T<S>> mid_, normal_;
  std::vector<S> len_;
  std::vector<uint64_t> hash_;
};

/**
 * @brief Owns the storage behind a WaySoA.
 *
 * avgEdgeLen is never recomputed here: it is copied from the Way, which stays
 * the authority. That keeps the two representations in agreement by
 * construction instead of by two copies of the same recurrence.
 */
template <typename S>
class WaySoAHostT {
 public:
  /**
   * @brief Whether to maintain the accelerating indexes (see WaySoAT).
   *
   * Off by default so the reference backend keeps exercising the plain scans.
   */
  void setBuildIndex(bool on) { buildIndex_ = on; }

  void clear() {
    mid_.clear();
    normal_.clear();
    hash_.clear();
    segMin_.clear();
    segMax_.clear();
    table_.clear();
    tableOk_ = true;
    avgEdgeLen_ = S(0);
  }

  void push(Vec2T<S> mid, Vec2T<S> normal, uint64_t hash) {
    mid_.push_back(mid);
    normal_.push_back(normal);
    hash_.push_back(hash);

    if (not buildIndex_) return;

    // Segment k = (mid[k], mid[k-1]) becomes available when element k lands.
    const size_t k = mid_.size() - 1;
    if (k >= 1) {
      const Vec2T<S> &a = mid_[k - 1], &b = mid_[k];
      segMin_.push_back(vec2<S>(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y));
      segMax_.push_back(vec2<S>(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y));
    } else {
      segMin_.push_back(mid);  // index 0 is never read
      segMax_.push_back(mid);
    }

    insertHash(hash);
  }

  void setAvgEdgeLen(S v) { avgEdgeLen_ = v; }

  size_t size() const { return mid_.size(); }

  WaySoAT<S> view() const {
    WaySoAT<S> v;
    v.mid = mid_.data();
    v.normal = normal_.data();
    v.hash = hash_.data();
    v.size = static_cast<uint32_t>(mid_.size());
    v.avgEdgeLen = avgEdgeLen_;
    v.hashTable = (buildIndex_ and tableOk_ and not table_.empty()) ? table_.data() : nullptr;
    v.hashMask = table_.empty() ? 0u : static_cast<uint32_t>(table_.size() - 1);
    v.hashEmpty = kEmpty();
    v.segMin = buildIndex_ ? segMin_.data() : nullptr;
    v.segMax = buildIndex_ ? segMax_.data() : nullptr;
    return v;
  }

  const std::vector<uint64_t> &hashTable() const { return table_; }
  const std::vector<Vec2T<S>> &segMin() const { return segMin_; }
  const std::vector<Vec2T<S>> &segMax() const { return segMax_; }
  uint64_t hashEmptyValue() const { return kEmpty(); }

 private:
  /// Free-slot sentinel. A function, not a static const member: the latter is
  /// odr-used here (passed by reference into assign/compare) and would need an
  /// out-of-line definition, which nvcc and gcc disagree about at C++14.
  static uint64_t kEmpty() { return ~uint64_t(0); }

  void insertHash(uint64_t h) {
    // An Edge hash colliding with the sentinel would make the table lie, so
    // give up on it rather than answer wrongly; view() then falls back to the
    // linear scan.
    if (h == kEmpty()) {
      tableOk_ = false;
      return;
    }
    // Keep the load factor under 1/2 so probes stay short.
    if (table_.empty() or (hash_.size() * 2 > table_.size())) rebuildTable();
    else probeInsert(table_, h);
  }

  void rebuildTable() {
    size_t cap = 16;
    while (cap < hash_.size() * 4) cap <<= 1;
    table_.assign(cap, kEmpty());
    for (const uint64_t &h : hash_) {
      if (h == kEmpty()) {
        tableOk_ = false;
        return;
      }
      probeInsert(table_, h);
    }
  }

  static void probeInsert(std::vector<uint64_t> &t, uint64_t h) {
    const uint32_t mask = static_cast<uint32_t>(t.size() - 1);
    uint32_t i = static_cast<uint32_t>(h) & mask;
    while (t[i] != kEmpty()) {
      if (t[i] == h) return;
      i = (i + 1) & mask;
    }
    t[i] = h;
  }

  std::vector<Vec2T<S>> mid_, normal_;
  std::vector<uint64_t> hash_;
  std::vector<Vec2T<S>> segMin_, segMax_;
  std::vector<uint64_t> table_;
  bool buildIndex_ = false;
  bool tableOk_ = true;
  S avgEdgeLen_ = S(0);
};

/**
 * @brief Names every layout type for one scalar, so a backend writes
 * `using L = ccs::Layout<float>;` once instead of spelling `ccs::Vec2T<float>`
 * at each use.
 */
template <typename S>
struct Layout {
  using scalar = S;
  using Vec2 = Vec2T<S>;
  using EdgeSoA = EdgeSoAT<S>;
  using WaySoA = WaySoAT<S>;
  using SearchConsts = SearchConstsT<S>;
  using Trace = TraceT<S>;
  using SearchContext = SearchContextT<S>;
  using EdgeSoAHost = EdgeSoAHostT<S>;
  using WaySoAHost = WaySoAHostT<S>;
};

}  // namespace ccs
