/**
 * @file search_factory.cpp
 * @brief createSearch implementation.
 */

#include "modules/search_factory.hpp"

#include <cstdlib>
#include <cstring>

#include <rclcpp/rclcpp.hpp>

#include "modules/cpu_fast_search.hpp"
#include "modules/cpu_search.hpp"

#ifdef USE_CUDA
#include "modules/cuda_search.hpp"
#endif

std::unique_ptr<ISearch> createSearch(const Params::WayComputer &params) {
  std::unique_ptr<ISearch> backend;

  // An environment variable rather than a ROS parameter on purpose: this exists
  // to A/B two backends over the same bag, and it has to be settable per
  // process without editing the config the run is meant to be reproducing.
  //   CCS_SEARCH_BACKEND=cpu       frozen reference (KDTree). Slow by design.
  //   CCS_SEARCH_BACKEND=cpu-fast  production host backend (default)
  //   CCS_SEARCH_BACKEND=cuda      device backend, if built and present
  //
  // The device backend is OPT-IN even when it is built, because it is currently
  // SLOWER than the host one: measured on rosbag__6, 25.40 ms/callback against
  // cpu-fast's 15.46 ms. It is bit-identical and correct, just not yet worth
  // running -- see cuda_search.hpp for where the time goes. Defaulting to it
  // because it exists would be a regression.
  const char *requested = std::getenv("CCS_SEARCH_BACKEND");
  const bool asked = (requested != nullptr);
  const bool askedCuda = asked and std::strcmp(requested, "cuda") == 0;

  if (asked and std::strcmp(requested, "cpu") == 0) {
    backend.reset(new CpuSearch(params));
  } else if (asked and std::strcmp(requested, "cpu-fast") == 0) {
    backend.reset(new CpuFastSearch(params));
  } else if (asked and not askedCuda) {
    RCLCPP_WARN(rclcpp::get_logger("local_planner"),
                "Unknown CCS_SEARCH_BACKEND '%s', falling back to the default.", requested);
  }

#ifdef USE_CUDA
  if (not backend and askedCuda) {
    const char *why = nullptr;
    if (not CudaSearch::deviceAvailable()) {
      if (askedCuda)
        RCLCPP_WARN(rclcpp::get_logger("local_planner"),
                    "CCS_SEARCH_BACKEND=cuda requested but no device is available.");
    } else if (not CudaSearch::supportsParams(params, &why)) {
      // Refuse rather than truncate: a device frontier that silently overflowed
      // would change the chosen path, which is the one failure mode the
      // validation setup cannot see.
      RCLCPP_WARN(rclcpp::get_logger("local_planner"),
                  "CUDA backend cannot honour this configuration (%s); using the host backend.",
                  why ? why : "unspecified");
    } else {
      backend.reset(new CudaSearch(params));
    }
  }
#else
  if (askedCuda)
    RCLCPP_WARN(rclcpp::get_logger("local_planner"),
                "CCS_SEARCH_BACKEND=cuda requested but the package was built without USE_CUDA.");
#endif

  if (not backend) backend.reset(new CpuFastSearch(params));

  RCLCPP_INFO(rclcpp::get_logger("local_planner"), "Search backend: %s", backend->name());
  return backend;
}
