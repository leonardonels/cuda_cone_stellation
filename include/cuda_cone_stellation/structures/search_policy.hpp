/**
 * @file search_policy.hpp
 * @brief POLICY: what a valid way is. The one thing every backend must share.
 *
 * These are the predicates the original WayComputer search applied, restated
 * against the flat layout in search_types.hpp. A backend chooses its own
 * strategy -- its spatial index, the shape of its tree walk, what it runs in
 * parallel -- but it does not get to choose these. They are threshold
 * comparisons, so a disagreement here is not a slightly different number, it is
 * a different candidate accepted and therefore a different racing line.
 *
 * If a filter changes here, every backend changes with it. That is the point.
 *
 * See search_types.hpp for the layout/policy/strategy split these two files
 * exist to enforce.
 */

#pragma once

#include <cstring>

#include "structures/search_types.hpp"

namespace ccs {

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

#ifdef CCS_PERTURB_TRANSCENDENTALS
/*
 * TEMPORARY EXPERIMENT -- not part of the build.
 *
 * Emulates a device math library that is within a couple of ulp of glibc, at
 * the rates actually measured on this Orin (atan2f: 39.3% of inputs differ, max
 * 2 ulp; logf: 5.0%, max 1 ulp; sqrtf: exact, so untouched). The trigger is a
 * function of the result's own bits, so it is deterministic and reproducible
 * while being uncorrelated with the input in any way the search could exploit.
 *
 * The question it answers: does a 1-2 ulp disagreement in the transcendentals
 * move the racing line a little, or does it produce a different one?
 */
CCS_HD inline float m_atan2_perturbed(float y, float x) {
  float r = ::atan2f(y, x);
  uint32_t b;
  __builtin_memcpy(&b, &r, 4);
  const uint32_t sel = b & 7u;
  if (sel < 2u) b += 1u;        // ~25% by 1 ulp
  else if (sel < 3u) b += 2u;   // ~12.5% by 2 ulp
  __builtin_memcpy(&r, &b, 4);
  return r;
}
CCS_HD inline float m_log_perturbed(float v) {
  float r = ::logf(v);
  uint32_t b;
  __builtin_memcpy(&b, &r, 4);
  if ((b & 31u) == 0u) b += 1u;  // ~3% by 1 ulp
  __builtin_memcpy(&r, &b, 4);
  return r;
}
#endif

}  // namespace detail

/// b - a, i.e. the original Vector(a, b).
template <typename S>
CCS_HD inline Vec2T<S> vecFromTo(Vec2T<S> a, Vec2T<S> b) {
  return vec2<S>(b.x - a.x, b.y - a.y);
}

template <typename S>
CCS_HD inline S distSq(Vec2T<S> a, Vec2T<S> b) {
  const S dx = a.x - b.x, dy = a.y - b.y;
  return dx * dx + dy * dy;
}

template <typename S>
CCS_HD inline S dist(Vec2T<S> a, Vec2T<S> b) {
  return detail::m_sqrt(distSq(a, b));
}

template <typename S>
CCS_HD inline S dot(Vec2T<S> a, Vec2T<S> b) {
  return a.x * b.x + a.y * b.y;
}

/// The original Vector::angleWith: signed angle from \a a to \a b.
template <typename S>
CCS_HD inline S angleWith(Vec2T<S> a, Vec2T<S> b) {
  const S det = a.x * b.y - a.y * b.x;
#ifdef CCS_PERTURB_TRANSCENDENTALS
  return detail::m_atan2_perturbed(det, dot(a, b));
#else
  return detail::m_atan2(det, dot(a, b));
#endif
}

template <typename S>
CCS_HD inline Vec2T<S> rotClock(Vec2T<S> v) {
  return vec2<S>(v.y, -v.x);
}

template <typename S>
CCS_HD inline bool pointBehind(Vec2T<S> futPos, Vec2T<S> actPos, Vec2T<S> dir) {
  return dot(vecFromTo(actPos, futPos), dir) < S(0);
}

template <typename S>
CCS_HD inline bool ccw(Vec2T<S> A, Vec2T<S> B, Vec2T<S> C) {
  return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x);
}

template <typename S>
CCS_HD inline bool segmentsIntersect(Vec2T<S> A, Vec2T<S> B, Vec2T<S> C, Vec2T<S> D) {
  return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D);
}

/* -------------------------------------------------------------------------- */
/*                              Way predicates                                 */
/* -------------------------------------------------------------------------- */

/**
 * @brief Way::containsEdge.
 *
 * Answers from way.hashTable when the caller built one, otherwise scans. Both
 * paths decide membership in the SAME set of hashes, so the answer is identical
 * either way -- the table is an index, not a different rule.
 */
template <typename S>
CCS_HD inline bool wayContainsEdge(const WaySoAT<S> &way, uint64_t hash) {
  if (way.hashTable != nullptr) {
    if (hash == way.hashEmpty) return false;  // sentinel is never a member
    uint32_t i = static_cast<uint32_t>(hash) & way.hashMask;
    for (;;) {
      const uint64_t slot = way.hashTable[i];
      if (slot == way.hashEmpty) return false;
      if (slot == hash) return true;
      i = (i + 1) & way.hashMask;
    }
  }
  for (uint32_t i = 0; i < way.size; ++i)
    if (way.hash[i] == hash) return true;
  return false;
}

/**
 * @brief Way::intersectsWith. O(|way|) per candidate, and the single most
 * expensive filter despite rejecting almost nothing.
 *
 * Note that the FIXED endpoint of the tested segment is the WAY's last
 * midpoint, not the trace's -- that is what the original does, and it is what
 * makes this filter prunable: for a whole outer iteration one endpoint does not
 * move. A backend is free to exploit that; it is not free to change which
 * segments are tested.
 *
 * Restricting this to the Way segments within reach of that fixed endpoint was
 * tried and abandoned: measured on rosbag__6 the Way holds ~48 midpoints and
 * the reach bound ((max_search_tree_height + 1) * search_radius = 40 m) covers
 * 43 of its 46 segments, so the subset saved ~1.5% and cost a per-outer-
 * iteration walk of the Way. The Way is short because trimByLocal() keeps it
 * that way; a reach filter would only pay on a Way several times longer than
 * this one ever gets. A tighter index (per-segment bounding boxes, or a grid
 * over the segments) is the version that could work -- the reach bound is not.
 */
template <typename S>
CCS_HD inline bool wayIntersectsWith(const WaySoAT<S> &way, Vec2T<S> candMid) {
  if (way.size <= 2) return false;
  const Vec2T<S> s2p1 = way.mid[way.size - 1];
  const Vec2T<S> s2p2 = candMid;

  // Bounding box of the tested segment. One endpoint is the Way's back, which
  // does not move for a whole outer iteration, so this box is small whenever
  // the candidate is nearby -- which is most of them.
  const S qMinX = s2p1.x < s2p2.x ? s2p1.x : s2p2.x;
  const S qMaxX = s2p1.x > s2p2.x ? s2p1.x : s2p2.x;
  const S qMinY = s2p1.y < s2p2.y ? s2p1.y : s2p2.y;
  const S qMaxY = s2p1.y > s2p2.y ? s2p1.y : s2p2.y;

  const bool haveBoxes = (way.segMin != nullptr and way.segMax != nullptr);

  // The original walks backwards from the penultimate midpoint; the segment set
  // is the same either way, and only the early-out order differs.
  for (int32_t k = int32_t(way.size) - 2; k >= 1; --k) {
    if (haveBoxes) {
      // Disjoint boxes cannot intersect, so skipping is exact, not heuristic.
      // Four comparisons against four ccw() determinants (twelve multiplies).
      const Vec2T<S> lo = way.segMin[k], hi = way.segMax[k];
      if (hi.x < qMinX or lo.x > qMaxX or hi.y < qMinY or lo.y > qMaxY) continue;
    }
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
template <typename S>
CCS_HD inline bool wayClosesLoopWith(const WaySoAT<S> &way, Vec2T<S> candMid, Vec2T<S> actPos,
                                     const SearchConstsT<S> &p) {
  if (way.size + 1 < p.min_loop_size) return false;
  if (distSq(way.mid[0], candMid) > p.max_dist_loop_closure * p.max_dist_loop_closure) return false;
  return detail::m_abs(angleWith(vecFromTo(way.mid[0], way.mid[1]), vecFromTo(actPos, candMid))) <=
         p.max_angle_diff_loop_closure;
}

/// The Way's last midpoint, or the origin when it is empty (never used then).
template <typename S>
CCS_HD inline Vec2T<S> wayBackMid(const WaySoAT<S> &way) {
  return way.size ? way.mid[way.size - 1] : vec2<S>(S(0), S(0));
}

/* -------------------------------------------------------------------------- */
/*                             Candidate policy                                */
/* -------------------------------------------------------------------------- */

/**
 * @brief WayComputer::avgEdgeLen: the average edge length of the Way and the
 * Trace combined, weighted by their sizes.
 */
template <typename S>
CCS_HD inline S combinedAvgEdgeLen(const WaySoAT<S> &way, const TraceT<S> *trace) {
  if (not trace or trace->size == 0) return way.avgEdgeLen;
  if (way.size == 0) return trace->avgEdgeLen;
  const S n = S(trace->size + way.size);
  return ((trace->avgEdgeLen * S(trace->size)) / n) + ((way.avgEdgeLen * S(way.size)) / n);
}

/**
 * @brief WayComputer::getHeuristic. Smaller is better.
 */
template <typename S>
CCS_HD inline S heuristic(Vec2T<S> actPos, Vec2T<S> nextPos, Vec2T<S> dir, const SearchConstsT<S> &p) {
  const S kPiHalf = S(M_PI_2);
  const S distHeur = dist(actPos, nextPos);
  const S angle = angleWith(vecFromTo(actPos, nextPos), dir);
  S inner = ((kPiHalf - detail::m_abs(angle)) / kPiHalf) - S(0.2);
  // std::max(0, inner) semantics, NaN included: max returns the first argument
  // unless it is strictly less than the second.
  inner = (S(0) < inner) ? inner : S(0);
#ifdef CCS_PERTURB_TRANSCENDENTALS
  const S angleHeur = -detail::m_log_perturbed(inner);
#else
  const S angleHeur = -detail::m_log(inner);  // inner == 0 => +inf => rejected
#endif
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
template <typename S>
CCS_HD inline bool discardCandidate(const SearchContextT<S> &c, const WaySoAT<S> &way, Vec2T<S> candMid,
                                    S candLen, uint64_t candHash, const SearchConstsT<S> &p) {
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
  if (candLen < (S(1) - p.edge_len_diff_factor) * c.avgEdgeLen or
      candLen > (S(1) + p.edge_len_diff_factor) * c.avgEdgeLen)
    return true;

  // 6. [If not allow_intersection, only before closing the loop] Remove any
  //    edge which, appended, would create an intersection.
  if (not p.allow_intersection and not traceClosedLoop and not closesLoop and
      wayIntersectsWith(way, candMid))
    return true;

  return false;
}

}  // namespace ccs
