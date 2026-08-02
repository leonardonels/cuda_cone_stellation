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
  if (not backend and (not asked or askedCuda)) {
    if (CudaSearch::deviceAvailable())
      backend.reset(new CudaSearch(params));
    else if (askedCuda)
      RCLCPP_WARN(rclcpp::get_logger("local_planner"),
                  "CCS_SEARCH_BACKEND=cuda requested but no device is available.");
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
