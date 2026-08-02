/**
 * @file cpu_search.hpp
 * @brief Host implementation of ISearch, and the reference the device backend
 * must match.
 *
 * This is the original WayComputer search, restated against the flat SoA in
 * structures/soa.hpp. It deliberately keeps the original's control flow (a
 * level-synchronous BFS with a FIFO, one midpoint appended per outer
 * iteration) so that a disagreement with the device backend can only come from
 * the kernel, never from a difference in the reference.
 */

#pragma once

#include <utility>
#include <vector>

#include "modules/isearch.hpp"
#include "structures/soa.hpp"
#include "utils/KDTree.hpp"

class CpuSearch final : public ISearch {
 public:
  explicit CpuSearch(const Params::WayComputer &params);

  const char *name() const override {
#if defined(CCS_SCALAR_FLOAT32)
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
  using HeurIdx = std::pair<ccs::scalar, uint32_t>;

  /**
   * @brief Way-level parameters, needed to build the loop-closure thresholds.
   */
  Params::WayComputer::Way wayParams_;

  /* Scratch, owned across callbacks so the steady state does not allocate. */
  ccs::EdgeSoAHost edgeSoa_;
  ccs::WaySoAHost waySoa_;
  KDTree midpointsKDT_;
  std::vector<HeurIdx> nextEdges_;
  std::vector<HeurIdx> childEdges_;
  std::vector<HeurIdx> privilegeRunner_;
  std::vector<ccs::Trace> queue_;

  /**
   * @brief Flattens the Params into the device-shaped constants, clamping
   * max_search_tree_height to what a ccs::Trace can hold.
   */
  ccs::SearchConsts makeConsts(const Params::WayComputer::Search &params) const;

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
  ccs::SearchContext makeContext(const ccs::Trace *trace) const;

  /**
   * @brief Finds all admissible next edges from \a trace (or from the Way's
   * end when \a trace is null), best first, at most max_search_options of them.
   */
  void findNextEdges(std::vector<HeurIdx> &out, const ccs::Trace *trace, const ccs::SearchConsts &p);

  /**
   * @brief Bounded-depth BFS over the candidate edges. Returns the index of the
   * FIRST edge of the longest (tie-broken by accumulated heuristic) trace.
   */
  uint32_t treeSearch(const std::vector<HeurIdx> &seeds, const ccs::SearchConsts &p, float timeLimit);
};
