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

#include <chrono>
#include <cstdint>
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

  /**
   * @brief Counters every backend maintains, so that two of them replayed on
   * the same bag can be compared directly.
   *
   * What each field counts is defined HERE, not by whichever backend happens
   * to be running: a backend is free to pick its own search strategy, but not
   * its own definition of "a node" or "a candidate evaluation", or the numbers
   * stop being an apples-to-apples comparison. A backend that genuinely has no
   * analogue for a counter leaves it at zero rather than repurposing it.
   */
  struct Stats {
    uint64_t callbacks = 0;      ///< computeWay() calls
    uint64_t outerIters = 0;     ///< midpoints appended to the Way
    uint64_t bfsNodes = 0;       ///< traces expanded, i.e. findNextEdges calls
    uint64_t candidateEvals = 0; ///< candidates put through the discard filters
    uint64_t timeLimitHits = 0;  ///< tree searches cut short by the time limit
    uint64_t waySizeSum = 0;     ///< sum of |way| over outer iterations, for the mean
    std::vector<double> callbackMs;  ///< wall time of each computeWay()
    /**
     * @brief PROCESS CPU time of each computeWay(), summed over all threads.
     *
     * Distinct from callbackMs on purpose. The problem that started this work
     * was a saturated core starving the forward computation, not latency -- so
     * a backend that waits on a device for 25 ms while consuming 1 ms of CPU is
     * a better answer than one that spends 15 ms of CPU, even though it looks
     * worse on wall time. Only measuring both makes that visible.
     */
    std::vector<double> callbackCpuMs;
  };

  virtual ~ISearch() = default;

  /**
   * @brief Name of the backend, for logging.
   */
  virtual const char *name() const = 0;

  const Stats &stats() const { return this->stats_; }

  /**
   * @brief Logs mean/p50/p95/max of the callback times plus the counters,
   * followed by whatever the backend adds through reportBackendDetail().
   *
   * Cumulative over the whole run, and meant to be called ONCE at a natural end
   * point -- it sorts the entire history to get its percentiles, so it is not
   * something to do on a timer.
   */
  void reportStats() const;

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

 protected:
  /**
   * @brief Optional extra line(s) for reportStats(), for whatever a particular
   * backend can say that the shared counters cannot.
   */
  virtual void reportBackendDetail() const {}

  /**
   * @brief Backends update this in place. Not private: maintaining it is part
   * of implementing the interface, not an optional extra.
   */
  Stats stats_;

  /**
   * @brief RAII timer a backend puts at the top of computeWay: counts the
   * callback and records its wall time, whichever way the function returns.
   */
  class CallbackTimer {
   public:
    explicit CallbackTimer(Stats &s);
    ~CallbackTimer();

   private:
    Stats &stats_;
    std::chrono::steady_clock::time_point begin_;
    double beginCpu_;
  };
};
