/**
 * @file search_bench.cpp
 * @brief Offline, deterministic replay of a rosbag through the real planner.
 *
 * Why this exists rather than `ros2 bag play` into the live node: the cone
 * subscription is KeepLast(1) at 20 Hz while odometry arrives at ~100 Hz, so
 * which odometry message is "the latest" when a cone callback fires depends on
 * scheduling. Two runs of the SAME backend therefore see different transforms
 * and produce different ways, and a slow backend additionally drops cone
 * messages a fast one does not. Neither is a property of the search, and both
 * make an A/B meaningless.
 *
 * Here the bag is read in timestamp order and every message is delivered, so
 * the input sequence is a function of the bag alone. Any difference in the
 * output digest between two runs is then attributable to the backend.
 *
 * It drives AutocrossPlanner's own callbacks rather than reimplementing the
 * pipeline, so what is measured is the production path, not a copy of it that
 * can drift from it.
 *
 * Usage:
 *   search_bench <bag_dir> <digest_out> [max_callbacks] \
 *       --ros-args --params-file <config.yaml> [-p autocross/search_backend:=cuda]
 *
 * The backend is a ROS parameter like any other, so it is overridden the way
 * any other parameter is.
 *
 * max_callbacks stops the replay early. The way grows as the cone map does, so
 * a truncated run is NOT a scaled-down version of a full one -- use it to
 * iterate, and the full bag to conclude.
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_options.hpp>

#include "autocross.h"

namespace {

/// FNV-1a over the raw bytes of the path's coordinates. Exact, not tolerant:
/// the point of the digest is to detect ANY divergence, and a tolerance would
/// have to be justified against a threshold-comparison search where a 1-ulp
/// difference flips a candidate and changes the whole line.
uint64_t digestPath(const std::vector<Point> &path) {
  uint64_t h = 1469598103934665603ull;
  auto mix = [&h](double v) {
    uint64_t bits;
    static_assert(sizeof(bits) == sizeof(v), "double is not 64-bit");
    __builtin_memcpy(&bits, &v, sizeof(bits));
    for (int i = 0; i < 8; ++i) {
      h ^= (bits >> (8 * i)) & 0xff;
      h *= 1099511628211ull;
    }
  };
  for (const Point &p : path) {
    mix(p.x);
    mix(p.y);
  }
  return h;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: search_bench <bag_dir> <digest_out> --ros-args --params-file <yaml>\n");
    return 2;
  }
  const std::string bagPath = argv[1];
  const std::string digestPath_ = argv[2];
  const uint64_t maxCallbacks =
      (argc > 3 and argv[3][0] != '-') ? strtoull(argv[3], nullptr, 10) : UINT64_MAX;

  rclcpp::init(argc, argv);

  // The YAML keys the planner reads are namespaced under this node name, so it
  // has to match what the launch file uses.
  auto nh = std::make_shared<rclcpp::Node>("cuda_cone_stellation");

  // Time::tock() logs one INFO per callback on the root logger; that alone is
  // milliseconds of formatting per callback and would land inside the numbers
  // being measured.
  // rcutils loggers inherit from the root, so quieting the root would also hide
  // the search statistics -- which are the entire point of this program; hence
  // the second call putting local_planner back to INFO.
  if (rcutils_logging_set_logger_level("", RCUTILS_LOG_SEVERITY_WARN) != RCUTILS_RET_OK or
      rcutils_logging_set_logger_level("local_planner", RCUTILS_LOG_SEVERITY_INFO) !=
          RCUTILS_RET_OK) {
    fprintf(stderr, "warning: could not set logger levels; timings will include log formatting\n");
  }

  auto centerLinePub = nh->create_publisher<visualization_msgs::msg::Marker>("planning/center_line", 1);
  auto centerLineCompletedPub =
      nh->create_publisher<visualization_msgs::msg::Marker>("planning/center_line_completed", 1);

  AutocrossPlanner planner(nh, centerLinePub, centerLineCompletedPub);

  rosbag2_storage::StorageOptions storage;
  storage.uri = bagPath;
  storage.storage_id = "sqlite3";
  rosbag2_cpp::ConverterOptions converter;
  converter.input_serialization_format = "cdr";
  converter.output_serialization_format = "cdr";

  rosbag2_cpp::Reader reader;
  reader.open(storage, converter);

  const rclcpp::Serialization<nav_msgs::msg::Odometry> odomSer;
  const rclcpp::Serialization<visualization_msgs::msg::Marker> markerSer;

  FILE *digestFile = fopen(digestPath_.c_str(), "w");
  if (not digestFile) {
    fprintf(stderr, "cannot write %s\n", digestPath_.c_str());
    return 1;
  }

  // A digest answers "identical or not". Once two backends CANNOT be bit
  // identical -- which is the case the moment a device transcendental is
  // involved -- the useful question becomes "how far apart", and that needs the
  // points themselves. Binary, little-endian: per callback a uint32 count then
  // that many (double,double) pairs.
  FILE *pathFile = nullptr;
  if (const char *pathDump = std::getenv("CCS_BENCH_DUMP_PATHS")) {
    pathFile = fopen(pathDump, "wb");
    if (not pathFile) {
      fprintf(stderr, "cannot write %s\n", pathDump);
      return 1;
    }
  }

  uint64_t nOdom = 0, nCones = 0;
  double pipelineMsTotal = 0;
  std::vector<double> pipelineMs;

  while (reader.has_next()) {
    const auto bagMsg = reader.read_next();
    rclcpp::SerializedMessage serialized(*bagMsg->serialized_data);

    if (bagMsg->topic_name == "/Odometry") {
      auto odom = std::make_shared<nav_msgs::msg::Odometry>();
      odomSer.deserialize_message(&serialized, odom.get());
      planner.wayComputer->stateCallback(odom);
      ++nOdom;
    } else if (bagMsg->topic_name == "/slam/cones_positions") {
      auto cones = std::make_shared<visualization_msgs::msg::Marker>();
      markerSer.deserialize_message(&serialized, cones.get());

      const auto t0 = std::chrono::steady_clock::now();
      planner.slamConesCallback(cones);
      const std::chrono::duration<double, std::milli> dt = std::chrono::steady_clock::now() - t0;

      pipelineMsTotal += dt.count();
      pipelineMs.push_back(dt.count());
      ++nCones;

      const std::vector<Point> path = planner.wayComputer->getPath();
      fprintf(digestFile, "%llu %zu %016llx\n", static_cast<unsigned long long>(nCones), path.size(),
              static_cast<unsigned long long>(digestPath(path)));

      if (pathFile) {
        const uint32_t n = static_cast<uint32_t>(path.size());
        fwrite(&n, sizeof(n), 1, pathFile);
        for (const Point &pt : path) {
          fwrite(&pt.x, sizeof(double), 1, pathFile);
          fwrite(&pt.y, sizeof(double), 1, pathFile);
        }
      }

      if (nCones >= maxCallbacks) break;
    }
  }

  fclose(digestFile);
  if (pathFile) fclose(pathFile);

  std::sort(pipelineMs.begin(), pipelineMs.end());
  const double p50 = pipelineMs.empty() ? 0 : pipelineMs[pipelineMs.size() / 2];
  const double p95 = pipelineMs.empty() ? 0 : pipelineMs[size_t(0.95 * (pipelineMs.size() - 1) + 0.5)];

#ifdef CCS_PERTURB_TRANSCENDENTALS
  printf("\n*** BUILT WITH CCS_PERTURB_TRANSCENDENTALS (device-math emulation) ***\n");
#endif
  printf("\n=== bag %s ===\n", bagPath.c_str());
  printf("odometry msgs        : %llu\n", static_cast<unsigned long long>(nOdom));
  printf("cone callbacks       : %llu\n", static_cast<unsigned long long>(nCones));
  printf("full pipeline / cb   : mean %.2f ms  p50 %.2f  p95 %.2f  max %.2f\n",
         nCones ? pipelineMsTotal / nCones : 0.0, p50, p95, pipelineMs.empty() ? 0.0 : pipelineMs.back());
  printf("wall total           : %.1f s\n", pipelineMsTotal / 1000.0);
  planner.wayComputer->reportSearchStats();

  rclcpp::shutdown();
  return 0;
}
