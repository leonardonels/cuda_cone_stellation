/**
 * @file cuda_one_shot_search.hpp
 * @brief Device backend #2: the WHOLE callback in one kernel launch.
 *
 * Same policy, same inner strategy, one difference from CudaSearch: who drives
 * the OUTER loop.
 *
 * WHY THIS EXISTS. cuda_search.hpp ends by naming the biggest remaining lever
 * on the device backend, and this is that lever built. The per-launch cost
 * there is driver overhead -- argument marshalling, the launch itself, and a
 * blocking cudaDeviceSynchronize -- and NO kernel change touches it. The only
 * fix is fewer launches, and the floor is one.
 *
 * WHAT IT BOUGHT. Measured on 2026 Cremona Rosbag 6, 1500 callbacks, against
 * the `cuda` backend on the same run:
 *
 *                          cuda       one-shot
 *   launches / callback    20.3       1.00
 *   search CPU / callback  2.78 ms    0.39 ms     (7.1x less)
 *   search wall / callback 16.59 ms   13.35 ms    (20% faster)
 *   kernel / callback      14.97 ms   12.93 ms
 *   CPU as % of wall       17%        3%
 *
 * The CPU drop is the point and was the prediction. The wall drop was not: it
 * is the ~2 ms of per-launch work INSIDE the kernel that a single launch also
 * stops repeating -- most of it re-staging the Way into shared memory 20 times
 * instead of once. Digests are identical to cpu, cpu-fast and cuda on bags 6,
 * 7 and 8.
 *
 * WHAT THE HOST LOOP ACTUALLY DID, and therefore what had to move:
 *
 *   1. Way::addEdge          -- the avgEdgeLen recurrence, in double.
 *   2. Way::closesLoop       -- |front - back| against max_dist_loop_closure,
 *                               in double.
 *   3. sizeAheadOfCar()      -- a counter; sizeToCar_ does not move during the
 *                               loop, so this is just the append count plus a
 *                               base the host passes in.
 *   4. WaySoAHost::push      -- the float mirror, the per-segment bounding
 *                               boxes and the edge-hash table the two O(|way|)
 *                               filters are accelerated with.
 *
 * ARITHMETIC, and why two of those say "in double". The search itself runs in
 * float on purpose (Orin's FP64 rate is ~1/32 of FP32). But 1 and 2 are NOT
 * part of the search -- they are Way's own bookkeeping, which the host does in
 * double, and both feed threshold comparisons: avgEdgeLen sets filter 5's
 * accept band, and closesLoop decides when the callback stops. Doing them in
 * float would let this backend diverge from the other three on inputs near a
 * threshold, which is precisely the failure the digest comparison exists to
 * catch. They cost a few dozen double operations per callback -- irrelevant
 * even at 1/32 rate -- so they are done in double, from double copies of the
 * edge midpoints and lengths uploaded alongside the float SoA.
 *
 * WHAT STILL CROSSES BACK. One buffer of chosen edge indices and a status word.
 * The host replays those indices into the real Way with the real Way::addEdge,
 * so the published Way is built by the same code as every other backend; the
 * device's Way is a mirror used to answer the filters, never the output.
 *
 * TWO REASONS A LAUNCH STILL RETURNS EARLY, both handled by relaunching:
 *   - the device Way hit its capacity (sized to |way| + |edges|, which the
 *     filters make unreachable in practice, but "in practice" is not a bound);
 *   - the iteration cap, which exists so rclcpp::ok() still gets checked
 *     during a shutdown. The measured steady state is 19.5 appends against a
 *     cap of 512, i.e. 1.00 launches per callback.
 *
 * That means neither the relaunch path nor the device-memory Way path ever runs
 * in the steady state, so both were validated by forcing them: with the cap
 * lowered to 3 (6.83 launches/callback) AND the Way pinned to device memory,
 * the digest over 1500 callbacks is still identical. Anything that changes how
 * the host hands Way state to a launch should be re-checked the same way,
 * because the steady state will not exercise it.
 *
 * WHERE THE KERNEL'S TIME GOES, measured with clock64() counters inside the
 * kernel (bag 6, 1500 callbacks), as a percentage of kernel time:
 *
 *   staging 0.0 | seed 6.3 | passA 84.5 | scan 5.1 | passB 2.6 | reduce 0.3
 *
 * So ~91% is findNextEdges, and it is ARITHMETIC AND DIVERGENCE, not memory.
 * That is worth stating plainly because the obvious next optimisation assumes
 * the opposite, and it is wrong:
 *
 * STAGING THE EDGES AND THE GRID IN SHARED MEMORY WAS TRIED AND ABANDONED.
 * The reasoning was that the candidate set is only ~300 edges, re-read ~4300
 * times per outer iteration from device memory, so it belonged in shared like
 * the Way. It was built -- packed 24-byte candidate records permuted into grid
 * order so a cell's candidates are contiguous, cellStart and the edge SoA
 * staged alongside, xs/ys/items dropped from the upload entirely, a greedy
 * priority allocator, and an opt-in past the 48 KB block limit. It produced
 * byte-identical digests and it was 1% SLOWER (13.35 -> 13.53 ms), because the
 * premise was false: the whole working set is a few tens of KB, so it already
 * lived in this SM's 128 KB L1, and an L1 hit and a shared-memory read cost
 * about the same. The clock64 line above is what settled it -- staging is
 * 0.0% of the kernel. Do not rebuild it without first showing a profile in
 * which memory is actually the cost.
 *
 * What DID pay, and is in the code, is the float screen on ccs::gridHit: the
 * radius test is FP64 on a device with a 1/32 FP64 rate and it runs on ~42
 * points per node against the ~12 that survive. Screening it in float and
 * settling only the ambiguous band exactly is worth 1.6-3.2%.
 *
 * The remaining ~85% is the filter chain and the heuristic themselves --
 * atan2f in filter 2 and in the heuristic, logf in the heuristic,
 * wayIntersectsWith in filter 6 -- executed by warps whose 32 lanes each drive
 * a DIFFERENT frontier node, so the warp pays the union of every lane's path
 * through the six filters. Reducing that divergence, not moving data around,
 * is where the next real win is.
 *
 * RELATIONSHIP TO CudaSearch. This is a SEPARATE file that duplicates the
 * device BFS rather than sharing it, deliberately: cuda_search.cu is the
 * measured baseline this backend is being compared against, and it stays
 * byte-for-byte untouched while that comparison is being made. The cost is
 * real -- a change to the BFS now has to be made twice -- so if this backend
 * wins, the two should be collapsed onto a shared device header rather than
 * left as two copies.
 */

#pragma once

#include <cstdint>
#include <vector>

#include "modules/isearch.hpp"
#include "structures/search_policy.hpp"
#include "utils/grid_index.hpp"

class CudaOneShotSearch final : public ISearch {
 public:
  /// float32, for the same reason CudaSearch uses it: Orin's FP64 rate.
  using scalar = float;
  using L = ccs::Layout<scalar>;

  /// Whether a usable device is present.
  static bool deviceAvailable();

  /**
   * @brief Whether this backend can honour \a params. Identical bound to
   * CudaSearch: the frontier is a fixed-capacity buffer, and a configuration
   * whose BFS could overflow it is refused rather than silently truncated.
   */
  static bool supportsParams(const Params::WayComputer &params, const char **why);

  explicit CudaOneShotSearch(const Params::WayComputer &params);
  ~CudaOneShotSearch() override;

  const char *name() const override { return "cuda-one-shot-search/float32"; }

  Result computeWay(const std::vector<Edge> &edges,
                    const Params::WayComputer::Search &params,
                    Way &way) override;

 private:
  /// Frontier capacity in traces; max_search_options^max_search_tree_height.
  static const uint32_t kMaxFrontier = 4096;

  /**
   * @brief Outer iterations one launch performs before handing control back.
   *
   * Not a performance knob -- the steady state is ~18. It exists so that a
   * pathological callback cannot make the node ignore rclcpp::ok() for an
   * unbounded time, which the host loop it replaced checked every iteration.
   */
  static const uint32_t kMaxItersPerLaunch = 512;

  void reportBackendDetail() const override;

  Params::WayComputer::Way wayParams_;

  /* ------------------------------ host staging ----------------------------- */

  L::EdgeSoAHost edgeSoa_;
  ccs::GridIndex grid_;
  std::vector<double> gridX_, gridY_;

  /**
   * @brief Exact double copies of what Way's own bookkeeping consumes.
   *
   * Separate from the grid's xs/ys, which are the FLOAT midpoints widened back
   * to double: those exist to reproduce the reference backend's radius test,
   * these exist to reproduce Way::addEdge and Way::closesLoop. Same edges,
   * different values, and conflating them would be a silent divergence.
   */
  std::vector<double> edgeMidXD_, edgeMidYD_, edgeLenD_;

  /// The Way image handed to the device at the start of a launch.
  std::vector<L::Vec2> wayMid_, wayNormal_, waySegMin_, waySegMax_;
  std::vector<uint64_t> wayHash_, wayTable_;
  bool wayTableOk_ = true;
  uint32_t wayCap_ = 0, tableCap_ = 0;

  struct Device;
  Device *dev_;

  L::SearchConsts makeConsts(const Params::WayComputer::Search &params) const;
  void buildEdgeSoA(const std::vector<Edge> &edges, double searchRadius);

  /**
   * @brief Builds the flat Way image plus its accelerating indexes, sized so
   * the device can append \a cap - |way| more elements without reallocating.
   */
  void buildWayImage(const Way &way, uint32_t cap);

  void syncEdgesToDevice();
  void syncWayToDevice();
};
