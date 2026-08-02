/**
 * @file cpu_search.hpp
 * @brief Host implementation of ISearch, and the reference every other backend
 * is validated against.
 *
 * This is the original WayComputer search, restated against the flat layout in
 * structures/search_types.hpp. It deliberately keeps the original's control
 * flow (a level-synchronous BFS with a FIFO, one midpoint appended per outer
 * iteration) because its job is to be the reference: a disagreement with
 * another backend should point at that backend, never at a difference in what
 * it is being compared to.
 *
 * That also means this class is NOT the place to make the CPU fast. Its
 * strategy is frozen. A faster host search belongs in its own ISearch, free to
 * pick a different spatial index and a different tree walk, and judged by
 * whether its way matches this one.
 */

#pragma once

#include <utility>
#include <vector>

#include "modules/isearch.hpp"
#include "structures/search_policy.hpp"
#include "utils/KDTree.hpp"

class CpuSearch final : public ISearch {
 public:
  /**
   * @brief This backend's scalar.
   *
   * Per-backend, not package-wide. It used to be one CMake flag that fixed the
   * scalar for everything at once, which meant the host and the device could
   * not choose differently -- and they want different things: Orin's FP64 rate
   * is ~1/32 of its FP32 rate, so a device backend wants float, while this one
   * wants whatever keeps it a faithful reference.
   */
#if defined(CCS_CPU_SCALAR_FLOAT32)
  using scalar = float;
#else
  using scalar = double;
#endif

  /// Every layout type at this backend's scalar.
  using L = ccs::Layout<scalar>;

  explicit CpuSearch(const Params::WayComputer &params);

  const char *name() const override {
#if defined(CCS_CPU_SCALAR_FLOAT32)
    return "cpu/float32";
#else
    return "cpu/double";
#endif
  }

  Result computeWay(const std::vector<Edge> &edges,
                    const Params::WayComputer::Search &params,
                    Way &way) override;

 private:
  /**
   * @brief A candidate: its heuristic and its index into the EdgeSoA. Ordered
   * by (heuristic, index), so the ordering is total and does not depend on the
   * order the radius query returned the candidates in.
   */
  using HeurIdx = std::pair<scalar, uint32_t>;

  /**
   * @brief Way-level parameters, needed to build the loop-closure thresholds.
   */
  Params::WayComputer::Way wayParams_;

  /* Scratch, owned across callbacks so the steady state does not allocate. */
  L::EdgeSoAHost edgeSoa_;
  L::WaySoAHost waySoa_;
  KDTree midpointsKDT_;
  std::vector<HeurIdx> nextEdges_;
  std::vector<HeurIdx> childEdges_;
  std::vector<HeurIdx> privilegeRunner_;
  std::vector<L::Trace> queue_;

  /**
   * @brief Flattens the Params into the device-shaped constants, clamping
   * max_search_tree_height to what a Trace can hold.
   */
  L::SearchConsts makeConsts(const Params::WayComputer::Search &params) const;

  /**
   * @brief Rebuilds the EdgeSoA and the midpoint k-d tree for this callback.
   */
  void buildEdgeSoA(const std::vector<Edge> &edges);

  /**
   * @brief Rebuilds the WaySoA from \a way (called once, after trimming).
   */
  void resetWaySoA(const Way &way);

  /**
   * @brief Mirrors an append to the Way into the WaySoA. The Way stays the
   * authority for avgEdgeLen.
   */
  void appendToWaySoA(const Edge &edge, const Way &way);

  /**
   * @brief Builds the SearchContext that the filters read: current edge,
   * position, previous position, direction and combined average edge length.
   */
  L::SearchContext makeContext(const L::Trace *trace) const;

  /**
   * @brief Finds all admissible next edges from \a trace (or from the Way's
   * end when \a trace is null), best first, at most max_search_options of them.
   */
  void findNextEdges(std::vector<HeurIdx> &out, const L::Trace *trace, const L::SearchConsts &p);

  /**
   * @brief Bounded-depth BFS over the candidate edges. Returns the index of the
   * FIRST edge of the longest (tie-broken by accumulated heuristic) trace.
   */
  uint32_t treeSearch(const std::vector<HeurIdx> &seeds, const L::SearchConsts &p, float timeLimit);
};
