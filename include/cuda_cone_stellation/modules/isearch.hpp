/**
 * @file isearch.hpp
 * @brief Backend-agnostic interface to the tree search.
 *
 * computeWay() is ~99% of the planner's cost, so it is the one thing that gets
 * a backend swap. Everything around it (triangulation filtering, midpoint
 * filtering, the tf, visualization) stays in WayComputer, which owns an ISearch
 * and is otherwise unchanged.
 */

#pragma once

#include <vector>

#include "structures/Edge.hpp"
#include "structures/Way.hpp"
#include "utils/Params.hpp"

class ISearch {
 public:
  /**
   * @brief What computeWay produces besides its mutation of the Way.
   */
  struct Result {
    /**
     * @brief Whether the search stopped because the Way closed its loop.
     */
    bool loopClosed = false;

    /**
     * @brief The Way to hand to the publishers. On loop closure this is the
     * restructured (complete) loop, otherwise it is the Way itself.
     */
    Way wayToPublish;
  };

  virtual ~ISearch() = default;

  /**
   * @brief Name of the backend, for logging.
   */
  virtual const char *name() const = 0;

  /**
   * @brief Takes all candidate Edge(s) and grows \a way one midpoint at a time
   * until no midpoint can be added, the horizon is reached, or the loop closes.
   *
   * \a way is trimmed to the car and then appended to in place, exactly as the
   * original WayComputer::computeWay did.
   *
   * @param[in] edges candidate edges for this callback, in a stable order
   * @param[in] params search thresholds (normal or failsafe)
   * @param[in,out] way
   */
  virtual Result computeWay(const std::vector<Edge> &edges,
                            const Params::WayComputer::Search &params,
                            Way &way) = 0;
};
