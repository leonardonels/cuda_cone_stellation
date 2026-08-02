/**
 * @file search_factory.cpp
 * @brief createSearch implementation.
 */

#include "modules/search_factory.hpp"

#include <rclcpp/rclcpp.hpp>

#include "modules/cpu_search.hpp"

#ifdef USE_CUDA
#include "modules/cuda_search.hpp"
#endif

std::unique_ptr<ISearch> createSearch(const Params::WayComputer &params) {
  std::unique_ptr<ISearch> backend;

#ifdef USE_CUDA
  if (CudaSearch::deviceAvailable()) {
    backend.reset(new CudaSearch(params));
  }
#endif

  if (not backend) backend.reset(new CpuSearch(params));

  RCLCPP_INFO(rclcpp::get_logger("local_planner"), "Search backend: %s", backend->name());
  return backend;
}
