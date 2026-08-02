/**
 * @file isearch.cpp
 * @brief The parts of ISearch that are shared by every backend: the callback
 * timer and the statistics dump.
 *
 * These deliberately live in the interface and not in a backend. Comparing two
 * backends is only meaningful if both are measured by the same instrument.
 */

#include "modules/isearch.hpp"

#include <algorithm>

#include <rclcpp/rclcpp.hpp>

ISearch::CallbackTimer::CallbackTimer(Stats &s)
    : stats_(s), begin_(std::chrono::steady_clock::now()) {}

ISearch::CallbackTimer::~CallbackTimer() {
  const std::chrono::duration<double, std::milli> elapsed =
      std::chrono::steady_clock::now() - this->begin_;
  ++this->stats_.callbacks;
  this->stats_.callbackMs.push_back(elapsed.count());
}

namespace {

/// Nearest-rank percentile of an already sorted, non-empty vector.
double percentile(const std::vector<double> &sorted, double q) {
  const size_t i = static_cast<size_t>(q * (sorted.size() - 1) + 0.5);
  return sorted[i];
}

}  // namespace

void ISearch::reportStats() const {
  const Stats &s = this->stats_;
  if (s.callbackMs.empty()) return;

  std::vector<double> ms = s.callbackMs;
  std::sort(ms.begin(), ms.end());

  double sum = 0;
  for (const double &v : ms) sum += v;

  RCLCPP_INFO(rclcpp::get_logger("local_planner"),
              "[search:%s] %lu callbacks | mean %.2f ms  p50 %.2f  p95 %.2f  max %.2f | "
              "per callback: %.1f midpoints, %.0f nodes, %.0f cand | "
              "|way| %.1f | timeLimitHits %lu",
              this->name(), static_cast<unsigned long>(s.callbacks), sum / ms.size(),
              percentile(ms, 0.50), percentile(ms, 0.95), ms.back(),
              double(s.outerIters) / s.callbacks, double(s.bfsNodes) / s.callbacks,
              double(s.candidateEvals) / s.callbacks,
              s.outerIters ? double(s.waySizeSum) / s.outerIters : 0.0,
              static_cast<unsigned long>(s.timeLimitHits));
}
