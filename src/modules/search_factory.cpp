/**
 * @file search_factory.cpp
 * @brief createSearch implementation.
 */

#include "modules/search_factory.hpp"

#include <rclcpp/rclcpp.hpp>

#include "modules/cpu_fast_search.hpp"
#include "modules/cpu_search.hpp"

#ifdef USE_CUDA
#include "modules/cuda_search.hpp"
#endif

std::unique_ptr<ISearch> createSearch(const Params::WayComputer &params) {
  std::unique_ptr<ISearch> backend;
  const std::string &want = params.search_backend;

  // Which backend to run is a configuration choice, not an environment one:
  //   cpu       frozen reference (KDTree, original tree walk). Slow by design;
  //             it exists to be the thing the others are validated against.
  //   cpu-fast  production host backend. Lowest latency.
  //   cuda      device backend. Higher latency, ~4x less CPU.
  //
  // Measured per callback on 2026 Cremona Rosbag (1500 callbacks), all three producing
  // byte-identical Ways:
  //
  //              wall       CPU       core @ 20 Hz
  //   cpu        46.5 ms    46.5 ms    93%
  //   cpu-fast   11.8 ms    11.6 ms    23%
  //   cuda       16.3 ms     2.8 ms     5.6%
  //
  // If the constraint is how fresh the centerline is, take cpu-fast; if it is a
  // core being starved of time for the forward computation, take cuda.
  if (want == "cpu") {
    backend.reset(new CpuSearch(params));
  } else if (want == "cuda") {
#ifdef USE_CUDA
    const char *why = nullptr;
    if (not CudaSearch::deviceAvailable()) {
      RCLCPP_WARN(rclcpp::get_logger("cuda_cone_stellation"),
                  "search_backend=cuda but no CUDA device is available; using cpu-fast.");
    } else if (not CudaSearch::supportsParams(params, &why)) {
      // Refuse rather than truncate: a device frontier that silently overflowed
      // would change the chosen path, which is the one failure mode the
      // validation setup cannot see.
      RCLCPP_WARN(rclcpp::get_logger("cuda_cone_stellation"),
                  "search_backend=cuda cannot honour this configuration (%s); using cpu-fast.",
                  why ? why : "unspecified");
    } else {
      backend.reset(new CudaSearch(params));
    }
#else
    RCLCPP_WARN(rclcpp::get_logger("cuda_cone_stellation"),
                "search_backend=cuda but the package was built without USE_CUDA; using cpu-fast.");
#endif
  } else if (want != "cpu-fast" and not want.empty()) {
    RCLCPP_WARN(rclcpp::get_logger("cuda_cone_stellation"),
                "Unknown search_backend '%s'; using cpu-fast.", want.c_str());
  }

  if (not backend) backend.reset(new CpuFastSearch(params));

  RCLCPP_INFO(rclcpp::get_logger("cuda_cone_stellation"), "Search backend: %s", backend->name());
  return backend;
}
