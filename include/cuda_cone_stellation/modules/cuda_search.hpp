/**
 * @file cuda_search.hpp
 * @brief Device backend: same POLICY as the host backends, GPU-native STRATEGY.
 *
 * The strategy here is NOT a port of the host tree walk, and that distinction
 * is the whole reason the layout/policy/strategy split exists. Three measured
 * facts shaped it, all taken on this Orin before any of it was written:
 *
 *   1. LATENCY IS THE BUDGET, NOT ARITHMETIC. A whole callback is ~84k
 *      candidate evaluations -- microseconds of GPU compute. What it is not is
 *      parallel: ~19.5 outer iterations, each a 7-level BFS, is ~136 dependent
 *      steps. An empty launch + sync costs 19.07 us here, so a kernel per BFS
 *      LEVEL would spend 2.6 ms/callback in launch overhead alone and win
 *      nothing. Therefore: ONE kernel per outer iteration, the entire bounded
 *      BFS resident on the device, and only the chosen edge index crosses back.
 *      That is 19.5 syncs/callback ~= 0.37 ms, which is affordable.
 *
 *   2. THE SPATIAL INDEX STAYS. The tempting GPU-native move is to drop the
 *      index and brute-force all N candidates per node, since the GPU has the
 *      lanes. It does not pay: the grid keeps a node's candidate set at ~12
 *      instead of N, and the work is latency-bound anyway, so brute force buys
 *      nothing and burns the memory bandwidth that the frontier needs. The
 *      device reuses the host's grid (utils/grid_index.hpp) through GridView.
 *
 *   3. THE DEVICE CAN MATCH THE HOST BIT FOR BIT -- but only with the right
 *      compile flags, and it is the OPPOSITE of the usual advice. Measured:
 *      with nvcc's default FMA contraction the device reproduces host aarch64
 *      results exactly (0 of 200k samples differ on ccw()'s determinant),
 *      because host GCC contracts too; with -fmad=false, 26% differ. So the
 *      CUDA target must NOT pass -fmad=false. Device atan2f/logf do differ from
 *      glibc (39.3% / 5.0% of inputs, <= 2 ulp), but emulating exactly that on
 *      the host over 4500 callbacks produced byte-identical paths, so it is not
 *      expected to move the racing line. Expected, not guaranteed: the digest
 *      comparison is what settles it per bag.
 *
 * Memory: Orin is an integrated GPU (cudaDeviceProp::integrated == 1), so
 * managed allocations are physically the same DRAM the host wrote. There is no
 * PCIe copy to amortise and no reason to double-buffer the SoA.
 */

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "modules/isearch.hpp"
#include "structures/search_policy.hpp"
#include "utils/grid_index.hpp"

class CudaSearch final : public ISearch {
 public:
  /**
   * @brief float32. Orin's FP64 rate is ~1/32 of its FP32 rate, and the host
   * backends are already at float32, so this also keeps the comparison
   * like-for-like.
   */
  using scalar = float;
  using L = ccs::Layout<scalar>;

  /**
   * @brief Whether a usable device is present. Called by the factory before
   * constructing, so that a build with USE_CUDA still runs on a machine
   * without a GPU.
   */
  static bool deviceAvailable();

  /**
   * @brief Whether this backend can honour \a params.
   *
   * The device frontier is a fixed-capacity buffer, so a configuration whose
   * BFS could exceed it is refused up front rather than silently truncated --
   * truncation would change the chosen path, which is exactly the kind of
   * quiet divergence the whole validation setup exists to prevent. The factory
   * falls back to the host backend when this returns false.
   */
  static bool supportsParams(const Params::WayComputer &params, const char **why);

  explicit CudaSearch(const Params::WayComputer &params);
  ~CudaSearch() override;

  const char *name() const override { return "cuda/float32"; }

  Result computeWay(const std::vector<Edge> &edges,
                    const Params::WayComputer::Search &params,
                    Way &way) override;

 private:
  /**
   * @brief Frontier capacity, in traces.
   *
   * The BFS branches by at most max_search_options and runs at most
   * max_search_tree_height levels, so the widest level holds
   * max_search_options^max_search_tree_height traces. At the shipped config
   * (3, 7) that is 2187. supportsParams() enforces the bound.
   */
  static const uint32_t kMaxFrontier = 4096;

  Params::WayComputer::Way wayParams_;

  /* Host-side staging, mirroring CpuFastSearch so the two build the same SoA. */
  L::EdgeSoAHost edgeSoa_;
  L::WaySoAHost waySoa_;
  ccs::GridIndex grid_;
  std::vector<double> gridX_, gridY_;

  /* Device (managed) storage. Sized on demand, reused across callbacks. */
  struct Device;
  Device *dev_;

  L::SearchConsts makeConsts(const Params::WayComputer::Search &params) const;
  void buildEdgeSoA(const std::vector<Edge> &edges, double searchRadius);
  void resetWaySoA(const Way &way);
  void appendToWaySoA(const Edge &edge, const Way &way);

  /**
   * @brief Uploads the current EdgeSoA/grid/Way to the device and runs one
   * outer iteration's worth of work: the seed query plus the whole bounded BFS.
   *
   * @param[out] seedsEmpty true when the seed query found nothing, which is the
   * host loop's termination condition.
   * @return the index of the first edge of the best trace, when \a seedsEmpty
   * is false.
   */
  uint32_t runOuterIteration(const L::SearchConsts &p, bool &seedsEmpty);

  /// Mirrors the growing Way into device memory (one appended element).
  void syncWayToDevice();
  void syncEdgesToDevice();
};
