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
 * @brief Upper bound on a Trace's length, i.e. on max_search_tree_height.
 * A Trace is a value type here, so this fixes its size; the search clamps
 * max_search_tree_height to it.
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
 * The original was a shared_ptr chain so that cloning was O(1). Here the depth
 * is bounded by max_search_tree_height (<= CCS_MAX_TRACE_LEN), so a Trace is a
 * fixed-size value: cloning is a memcpy and there is no allocation on either
 * backend.
 */
template <typename S>
struct TraceT {
  uint32_t edgeInd[CCS_MAX_TRACE_LEN];
  uint32_t size;
  S sumHeur;
  S avgEdgeLen;
  bool loopClosed;
};

template <typename S>
CCS_HD inline void traceInit(TraceT<S> &t) {
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
  t.edgeInd[t.size] = edgeInd;
  t.size++;
}

/// Index of the first edge of the Trace -- what treeSearch ultimately returns.
template <typename S>
CCS_HD inline uint32_t traceFirst(const TraceT<S> &t) {
  return t.edgeInd[0];
}

/// Index of the last edge appended.
template <typename S>
CCS_HD inline uint32_t traceLast(const TraceT<S> &t) {
  return t.edgeInd[t.size - 1];
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
  void clear() {
    mid_.clear();
    normal_.clear();
    hash_.clear();
    avgEdgeLen_ = S(0);
  }

  void push(Vec2T<S> mid, Vec2T<S> normal, uint64_t hash) {
    mid_.push_back(mid);
    normal_.push_back(normal);
    hash_.push_back(hash);
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
    return v;
  }

 private:
  std::vector<Vec2T<S>> mid_, normal_;
  std::vector<uint64_t> hash_;
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
