/**
 * @file soa.hpp
 * @brief Device-friendly, flat representation of everything the tree search
 * touches, plus the search predicates themselves.
 *
 * Nothing in this header may depend on ROS, Eigen, or any pointer-chasing
 * container: it is compiled by both the host and the device backend, and it is
 * the ONE place the filter logic lives. If a filter is changed here, both
 * backends change together — that is the point.
 *
 * The class graph the original search used (Node/Edge/Triangle with
 * std::array + unordered_set + const members, Trace as a shared_ptr chain,
 * KDTree as a pointer tree) does not cross to the device. What crosses is:
 *   - EdgeSoA : the callback's candidate edges, struct-of-arrays
 *   - WaySoA  : the Way so far, in path order (the shared-memory resident)
 *   - Trace   : a fixed-size array, because the BFS depth is bounded by
 *               max_search_tree_height, so no device-side allocation is needed
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

/**
 * @brief The scalar type of the whole search.
 *
 * Orin's FP64 rate is ~1/32 of its FP32 rate, so the device path wants float.
 * The filters are threshold comparisons, so a 1-ulp difference does not produce
 * a slightly different number, it flips a candidate and produces a *different
 * racing line* — which is why both backends must agree on this type rather than
 * each picking its own. Default is double: that is what keeps the CPU backend
 * bit-identical to the pre-refactor implementation.
 */
#if defined(CCS_SCALAR_FLOAT32)
using scalar = float;
#else
using scalar = double;
#endif

/* -------------------------------------------------------------------------- */
/*                                    Math                                     */
/* -------------------------------------------------------------------------- */

namespace detail {
CCS_HD inline double m_abs(double v) { return ::fabs(v); }
CCS_HD inline float m_abs(float v) { return ::fabsf(v); }
CCS_HD inline double m_sqrt(double v) { return ::sqrt(v); }
CCS_HD inline float m_sqrt(float v) { return ::sqrtf(v); }
CCS_HD inline double m_atan2(double y, double x) { return ::atan2(y, x); }
CCS_HD inline float m_atan2(float y, float x) { return ::atan2f(y, x); }
CCS_HD inline double m_log(double v) { return ::log(v); }
CCS_HD inline float m_log(float v) { return ::logf(v); }
}  // namespace detail

struct Vec2 {
  scalar x, y;
};

CCS_HD inline Vec2 vec2(scalar x, scalar y) {
  Vec2 v;
  v.x = x;
  v.y = y;
  return v;
}

/// b - a, i.e. the original Vector(a, b).
CCS_HD inline Vec2 vecFromTo(Vec2 a, Vec2 b) { return vec2(b.x - a.x, b.y - a.y); }

CCS_HD inline scalar distSq(Vec2 a, Vec2 b) {
  const scalar dx = a.x - b.x, dy = a.y - b.y;
  return dx * dx + dy * dy;
}

CCS_HD inline scalar dist(Vec2 a, Vec2 b) { return detail::m_sqrt(distSq(a, b)); }

CCS_HD inline scalar dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }

/// The original Vector::angleWith: signed angle from \a a to \a b.
CCS_HD inline scalar angleWith(Vec2 a, Vec2 b) {
  const scalar det = a.x * b.y - a.y * b.x;
  return detail::m_atan2(det, dot(a, b));
}

CCS_HD inline Vec2 rotClock(Vec2 v) { return vec2(v.y, -v.x); }

CCS_HD inline bool pointBehind(Vec2 futPos, Vec2 actPos, Vec2 dir) {
  return dot(vecFromTo(actPos, futPos), dir) < scalar(0);
}

CCS_HD inline bool ccw(Vec2 A, Vec2 B, Vec2 C) {
  return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x);
}

CCS_HD inline bool segmentsIntersect(Vec2 A, Vec2 B, Vec2 C, Vec2 D) {
  return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D);
}

/* -------------------------------------------------------------------------- */
/*                                    Data                                     */
/* -------------------------------------------------------------------------- */

/**
 * @brief One callback's candidate edges, flattened. An index into these arrays
 * is the \a edgeInd the search passes around, and it matches the index into the
 * std::vector<Edge> the host built this from.
 */
struct EdgeSoA {
  const Vec2 *mid;      ///< midpoint, local frame
  const Vec2 *normal;   ///< Vector(n0, n1).rotClock()
  const scalar *len;    ///< edge length (global frame, as the original computes it)
  const uint64_t *hash; ///< Edge hash, built from the two Node ids
  uint32_t size;
};

/**
 * @brief The Way so far, in path order. Grows by exactly one element per outer
 * iteration of computeWay. This is the array that belongs in shared memory:
 * filter 6 (intersectsWith) walks all of it for every candidate.
 */
struct WaySoA {
  const Vec2 *mid;
  const Vec2 *normal;
  const uint64_t *hash;
  uint32_t size;
  scalar avgEdgeLen;
};

/**
 * @brief The search thresholds, flattened out of Params.
 */
struct SearchConsts {
  scalar search_radius;
  scalar max_angle_diff;
  scalar edge_len_diff_factor;
  scalar max_next_heuristic;
  /**
   * @brief Deliberately float, not scalar. The original computes
   * `(1 - heur_dist_ponderation)` in float because the parameter is float, and
   * only then widens. Storing it as double changes that subtraction's rounding
   * and therefore the heuristic, and therefore the chosen path.
   */
  float heur_dist_ponderation;
  bool allow_intersection;
  int max_search_tree_height;
  int max_search_options;

  // From Params::WayComputer::Way — needed by the loop-closure predicate.
  scalar max_dist_loop_closure;
  scalar max_angle_diff_loop_closure;
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
struct Trace {
  uint32_t edgeInd[CCS_MAX_TRACE_LEN];
  uint32_t size;
  scalar sumHeur;
  scalar avgEdgeLen;
  bool loopClosed;
};

CCS_HD inline void traceInit(Trace &t) {
  t.size = 0;
  t.sumHeur = scalar(0);
  t.avgEdgeLen = scalar(0);
  t.loopClosed = false;
}

/**
 * @brief Appends one edge, mirroring Trace::Connection's constructor exactly.
 *
 * sumHeur is accumulated left to right; the original recursed right to left,
 * but floating point addition is commutative so the two agree bit for bit.
 */
CCS_HD inline void traceAppend(Trace &t, uint32_t edgeInd, scalar heur, scalar edgeLen, bool loopClosed) {
  if (t.size == 0) {
    t.avgEdgeLen = edgeLen;
    t.loopClosed = loopClosed;
  } else {
    t.avgEdgeLen = t.avgEdgeLen + ((edgeLen - t.avgEdgeLen) / scalar(t.size + 1));
    t.loopClosed = t.loopClosed or loopClosed;
  }
  t.sumHeur = t.sumHeur + heur;
  t.edgeInd[t.size] = edgeInd;
  t.size++;
}

/// Index of the first edge of the Trace — what treeSearch ultimately returns.
CCS_HD inline uint32_t traceFirst(const Trace &t) { return t.edgeInd[0]; }

/// Index of the last edge appended.
CCS_HD inline uint32_t traceLast(const Trace &t) { return t.edgeInd[t.size - 1]; }

/* -------------------------------------------------------------------------- */
/*                              Way predicates                                 */
/* -------------------------------------------------------------------------- */

/// Way::containsEdge.
CCS_HD inline bool wayContainsEdge(const WaySoA &way, uint64_t hash) {
  for (uint32_t i = 0; i < way.size; ++i)
    if (way.hash[i] == hash) return true;
  return false;
}

/**
 * @brief Way::intersectsWith. O(|way|) per candidate, and the single most
 * expensive thing in the search (~66% of the discard time) despite rejecting
 * almost nothing. On the device this is the coalesced inner loop.
 */
CCS_HD inline bool wayIntersectsWith(const WaySoA &way, Vec2 candMid) {
  if (way.size <= 2) return false;
  const Vec2 s2p1 = way.mid[way.size - 1];
  const Vec2 s2p2 = candMid;
  // The original walks backwards from the penultimate midpoint; the segment set
  // is the same either way, and only the early-out order differs.
  for (int32_t k = int32_t(way.size) - 2; k >= 1; --k) {
    if (segmentsIntersect(way.mid[k], way.mid[k - 1], s2p1, s2p2)) return true;
  }
  return false;
}

/**
 * @brief Way::closesLoopWith. \a actPos is what the original takes as
 * \a lastPosInTrace, or the Way's last midpoint when that is null.
 *
 * The original reads way.back() unconditionally to build actPos, which is
 * garbage on an empty Way but always discarded by the size short-circuit
 * below; callers here just pass anything when the Way is empty.
 */
CCS_HD inline bool wayClosesLoopWith(const WaySoA &way, Vec2 candMid, Vec2 actPos, const SearchConsts &p) {
  if (way.size + 1 < p.min_loop_size) return false;
  if (distSq(way.mid[0], candMid) > p.max_dist_loop_closure * p.max_dist_loop_closure) return false;
  return detail::m_abs(angleWith(vecFromTo(way.mid[0], way.mid[1]), vecFromTo(actPos, candMid))) <=
         p.max_angle_diff_loop_closure;
}

/// The Way's last midpoint, or the origin when it is empty (never used then).
CCS_HD inline Vec2 wayBackMid(const WaySoA &way) {
  return way.size ? way.mid[way.size - 1] : vec2(scalar(0), scalar(0));
}

/* -------------------------------------------------------------------------- */
/*                                  Search                                     */
/* -------------------------------------------------------------------------- */

/**
 * @brief Everything findNextEdges derives once, before it loops over the
 * candidates. On the device this is what gets broadcast to the block.
 */
struct SearchContext {
  bool hasActEdge;
  Vec2 actEdgeMid;
  Vec2 actEdgeNormal;
  uint64_t actEdgeHash;
  Vec2 actPos;
  Vec2 lastPos;
  Vec2 dir;
  scalar avgEdgeLen;     ///< combinedAvgEdgeLen, hoisted out of filter 5
  bool hasTrace;         ///< the original's `actTrace != nullptr`
  bool traceLoopClosed;
  bool sideCheckEnabled; ///< the original's `(way.size() >= 2 or actTrace)`
};

/**
 * @brief WayComputer::avgEdgeLen: the average edge length of the Way and the
 * Trace combined, weighted by their sizes.
 */
CCS_HD inline scalar combinedAvgEdgeLen(const WaySoA &way, const Trace *trace) {
  if (not trace or trace->size == 0) return way.avgEdgeLen;
  if (way.size == 0) return trace->avgEdgeLen;
  const scalar n = scalar(trace->size + way.size);
  return ((trace->avgEdgeLen * scalar(trace->size)) / n) + ((way.avgEdgeLen * scalar(way.size)) / n);
}

/**
 * @brief WayComputer::getHeuristic. Smaller is better.
 */
CCS_HD inline scalar heuristic(Vec2 actPos, Vec2 nextPos, Vec2 dir, const SearchConsts &p) {
  const scalar kPiHalf = scalar(M_PI_2);
  const scalar distHeur = dist(actPos, nextPos);
  const scalar angle = angleWith(vecFromTo(actPos, nextPos), dir);
  scalar inner = ((kPiHalf - detail::m_abs(angle)) / kPiHalf) - scalar(0.2);
  // std::max(0, inner) semantics, NaN included: max returns the first argument
  // unless it is strictly less than the second.
  inner = (scalar(0) < inner) ? inner : scalar(0);
  const scalar angleHeur = -detail::m_log(inner);  // inner == 0 => +inf => rejected
  return p.heur_dist_ponderation * distHeur + (1 - p.heur_dist_ponderation) * angleHeur;
}

/**
 * @brief The six discard filters, in the original's short-circuit order.
 *
 * Measured selectivity on a 112-cone track: f1 10.2%, f2 63.7%, f3 0.1%,
 * f4 1.4%, f5 0.0%, f6 0.0%, kept 24.7%. f2 is the cheap early-out that does
 * almost all the work; f6 rejects nothing and costs the most.
 *
 * @return true if the candidate must be discarded.
 */
CCS_HD inline bool discardCandidate(const SearchContext &c, const WaySoA &way, Vec2 candMid, scalar candLen,
                                    uint64_t candHash, const SearchConsts &p) {
  // The original guards the whole disjunction with `actEdge and (...)`.
  if (not c.hasActEdge) return false;

  // 1. Remove itself from being the next one.
  if (candHash == c.actEdgeHash) return true;

  // 2. Remove any edge whose midpoint creates an angle too closed with the last one.
  if (detail::m_abs(angleWith(c.dir, vecFromTo(c.actPos, candMid))) > p.max_angle_diff) return true;

  // Shared by filters 3 and 6, and pure, so it is hoisted.
  const bool closesLoop = wayClosesLoopWith(way, candMid, wayBackMid(way), p);
  const bool traceClosedLoop = c.hasTrace and c.traceLoopClosed;

  // 3. [Only before appending the edge that closes the loop] Remove any edge
  //    already in the path that is not the one closing the loop.
  if (not closesLoop and not traceClosedLoop and wayContainsEdge(way, candHash)) return true;

  // 4. Remove any edge whose midpoint and lastPos are on the same side of
  //    actEdge (avoid bouncing on a track limit).
  if (c.sideCheckEnabled and pointBehind(c.actEdgeMid, c.lastPos, c.actEdgeNormal) ==
                                 pointBehind(c.actEdgeMid, candMid, c.actEdgeNormal))
    return true;

  // 5. Remove any edge whose length is too far from the way's average.
  if (candLen < (scalar(1) - p.edge_len_diff_factor) * c.avgEdgeLen or
      candLen > (scalar(1) + p.edge_len_diff_factor) * c.avgEdgeLen)
    return true;

  // 6. [If not allow_intersection, only before closing the loop] Remove any
  //    edge which, appended, would create an intersection.
  if (not p.allow_intersection and not traceClosedLoop and not closesLoop and wayIntersectsWith(way, candMid))
    return true;

  return false;
}

/* -------------------------------------------------------------------------- */
/*                             Host-side builders                              */
/* -------------------------------------------------------------------------- */

/**
 * @brief Owns the storage behind an EdgeSoA. Rebuilt once per callback, which
 * on Orin is free (integrated memory, no PCIe copy).
 */
class EdgeSoAHost {
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

  void push(Vec2 mid, Vec2 normal, scalar len, uint64_t hash) {
    mid_.push_back(mid);
    normal_.push_back(normal);
    len_.push_back(len);
    hash_.push_back(hash);
  }

  size_t size() const { return mid_.size(); }

  EdgeSoA view() const {
    EdgeSoA v;
    v.mid = mid_.data();
    v.normal = normal_.data();
    v.len = len_.data();
    v.hash = hash_.data();
    v.size = static_cast<uint32_t>(mid_.size());
    return v;
  }

 private:
  std::vector<Vec2> mid_, normal_;
  std::vector<scalar> len_;
  std::vector<uint64_t> hash_;
};

/**
 * @brief Owns the storage behind a WaySoA.
 *
 * avgEdgeLen is never recomputed here: it is copied from the Way, which stays
 * the authority. That keeps the two representations in agreement by
 * construction instead of by two copies of the same recurrence.
 */
class WaySoAHost {
 public:
  void clear() {
    mid_.clear();
    normal_.clear();
    hash_.clear();
    avgEdgeLen_ = scalar(0);
  }

  void push(Vec2 mid, Vec2 normal, uint64_t hash) {
    mid_.push_back(mid);
    normal_.push_back(normal);
    hash_.push_back(hash);
  }

  void setAvgEdgeLen(scalar v) { avgEdgeLen_ = v; }

  size_t size() const { return mid_.size(); }

  WaySoA view() const {
    WaySoA v;
    v.mid = mid_.data();
    v.normal = normal_.data();
    v.hash = hash_.data();
    v.size = static_cast<uint32_t>(mid_.size());
    v.avgEdgeLen = avgEdgeLen_;
    return v;
  }

 private:
  std::vector<Vec2> mid_, normal_;
  std::vector<uint64_t> hash_;
  scalar avgEdgeLen_ = scalar(0);
};

}  // namespace ccs
