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
#include <ctime>

#include <rclcpp/rclcpp.hpp>

namespace {

/// Process CPU time in milliseconds, summed over every thread -- so a driver
/// thread spinning on a device sync is counted, which is exactly the cost this
/// is here to expose.
double processCpuMs() {
  timespec ts;
  if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) != 0) return 0.0;
  return double(ts.tv_sec) * 1e3 + double(ts.tv_nsec) / 1e6;
}

}  // namespace

ISearch::CallbackTimer::CallbackTimer(Stats &s)
    : stats_(s), begin_(std::chrono::steady_clock::now()), beginCpu_(processCpuMs()) {}

ISearch::CallbackTimer::~CallbackTimer() {
  const std::chrono::duration<double, std::milli> elapsed =
      std::chrono::steady_clock::now() - this->begin_;
  ++this->stats_.callbacks;
  this->stats_.callbackMs.push_back(elapsed.count());
  this->stats_.callbackCpuMs.push_back(processCpuMs() - this->beginCpu_);
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

  std::vector<double> cpu = s.callbackCpuMs;
  std::sort(cpu.begin(), cpu.end());
  double cpuSum = 0;
  for (const double &v : cpu) cpuSum += v;
  const double cpuMean = cpu.empty() ? 0.0 : cpuSum / cpu.size();

  RCLCPP_INFO(rclcpp::get_logger("local_planner"),
              "[search:%s] %lu callbacks | wall mean %.2f ms  p50 %.2f  p95 %.2f  max %.2f | "
              "CPU mean %.2f ms (%.0f%% of wall) | "
              "per callback: %.1f midpoints, %.0f nodes, %.0f cand | "
              "|way| %.1f | timeLimitHits %lu",
              this->name(), static_cast<unsigned long>(s.callbacks), sum / ms.size(),
              percentile(ms, 0.50), percentile(ms, 0.95), ms.back(), cpuMean,
              sum > 0 ? 100.0 * cpuSum / sum : 0.0,
              double(s.outerIters) / s.callbacks, double(s.bfsNodes) / s.callbacks,
              double(s.candidateEvals) / s.callbacks,
              s.outerIters ? double(s.waySizeSum) / s.outerIters : 0.0,
              static_cast<unsigned long>(s.timeLimitHits));

  this->reportBackendDetail();
}
