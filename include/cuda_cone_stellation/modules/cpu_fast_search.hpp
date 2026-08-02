/**
 * @file cpu_fast_search.hpp
 * @brief Production host backend: same POLICY as CpuSearch, different STRATEGY.
 *
 * CpuSearch is frozen as the reference, so this is where the host gets fast.
 * The two must produce the same Way -- that is what the offline bag replay
 * checks -- but they are free to disagree about how they get there.
 *
 * What differs from the reference, and why:
 *
 *   - The radius query is a uniform grid (utils/grid_index.hpp) instead of
 *     KDTree. Measured at 78% of the reference's per-node cost, so this is the
 *     whole point of the class. The query radius is a constant, which is
 *     exactly the case a grid is built for and the case a k-d tree's
 *     flexibility is wasted on.
 *
 *   - The candidate buffer is a member reused across queries, so a query does
 *     not allocate. KDTree::neighborhood_indices returns a fresh vector every
 *     call, i.e. once per BFS node.
 *
 * The tree walk itself is deliberately UNCHANGED from the reference: keeping it
 * identical is what makes a digest mismatch mean "the index is wrong" rather
 * than "something in here is different". Changing the walk is a separate step,
 * to be taken against a passing baseline.
 */

#pragma once

#include <utility>
#include <vector>

#include "modules/isearch.hpp"
#include "structures/search_policy.hpp"
#include "utils/grid_index.hpp"

class CpuFastSearch final : public ISearch {
 public:
  /**
   * @brief Matches CpuSearch's scalar: the two are meant to be swappable, and
   * comparing them is only meaningful at the same width.
   */
#if defined(CCS_CPU_SCALAR_FLOAT32)
  using scalar = float;
#else
  using scalar = double;
#endif

  using L = ccs::Layout<scalar>;

  explicit CpuFastSearch(const Params::WayComputer &params);

  const char *name() const override {
#if defined(CCS_CPU_SCALAR_FLOAT32)
    return "cpu-fast/float32";
#else
    return "cpu-fast/double";
#endif
  }

  Result computeWay(const std::vector<Edge> &edges,
                    const Params::WayComputer::Search &params,
                    Way &way) override;

 private:
  using HeurIdx = std::pair<scalar, uint32_t>;

  Params::WayComputer::Way wayParams_;

  /* Scratch, owned across callbacks so the steady state does not allocate. */
  L::EdgeSoAHost edgeSoa_;
  L::WaySoAHost waySoa_;
  ccs::GridIndex grid_;
  std::vector<double> gridX_, gridY_;
  std::vector<uint32_t> candidates_;  ///< reused by every radius query

  std::vector<HeurIdx> nextEdges_;
  std::vector<HeurIdx> childEdges_;
  std::vector<HeurIdx> privilegeRunner_;
  std::vector<L::Trace> queue_;

  L::SearchConsts makeConsts(const Params::WayComputer::Search &params) const;

  /**
   * @brief Rebuilds the EdgeSoA and the grid for this callback.
   *
   * \a searchRadius is the radius the grid sizes its cells to, and it is passed
   * in rather than read from a member because the failsafe path runs the whole
   * search again with inflated parameters -- a grid built for the normal radius
   * would be wrong for it.
   */
  void buildEdgeSoA(const std::vector<Edge> &edges, double searchRadius);

  void resetWaySoA(const Way &way);
  void appendToWaySoA(const Edge &edge, const Way &way);

  L::SearchContext makeContext(const L::Trace *trace) const;
  void findNextEdges(std::vector<HeurIdx> &out, const L::Trace *trace, const L::SearchConsts &p);
  uint32_t treeSearch(const std::vector<HeurIdx> &seeds, const L::SearchConsts &p, float timeLimit);
};
