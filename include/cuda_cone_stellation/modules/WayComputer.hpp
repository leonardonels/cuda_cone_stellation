/**
 * @file WayComputer.hpp
 * @author Oriol Gorriz (origovi2000@gmail.com)
 * @brief Contains the WayComputer class specification
 * @version 1.0
 * @date 2022-10-31
 *
 * @copyright Copyright (c) 2022 BCN eMotorsport
 */

#pragma once

#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
// #include <visualization_msgs/msg/path_limits.hpp>
// #include <visualization_msgs/msg/tracklimits.hpp>
/*#include <as_msgs/CarState.h>
#include <as_msgs/Tracklimits.h>*/
#include <tf2_eigen/tf2_eigen.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <tf2/impl/utils.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <Eigen/Geometry> 

#include <fstream>
#include <memory>

#include "modules/Visualization.hpp"
#include "modules/isearch.hpp"
#include "structures/Vector.hpp"
#include "structures/Way.hpp"
#include "utils/KDTree.hpp"
#include "utils/Params.hpp"
#include "utils/Failsafe.hpp"
#include "utils/constants.hpp"
#include "utils/definitions.hpp"

/**
 * @brief A class that has all tools and functions to compute the Way.
 * It takes the Delaunay set and the car position to do so.
 */
class WayComputer {
 private:
  /**
   * @brief All parameters related to the WayComputer class.
   */
  const Params::WayComputer params_;

  /**
   * @brief Failsafe of search parameters.
   */
  Failsafe<Params::WayComputer::Search> generalFailsafe_;

  /**
   * @brief The search backend. Owns everything that used to be this class's
   * private search machinery, and is the only piece that changes between the
   * CPU and the CUDA path.
   */
  std::unique_ptr<ISearch> search_;

  /**
   * @brief The result of the computation and last iteration's result.
   */
  Way way_, lastWay_;

  /**
   * @brief This Way object had to be created to solve the non-stop loop
   * calculation. It is the Way that will be published every time (for both
   * full & partial).
   */
  Way wayToPublish_;

  /**
   * @brief Last data timestamp.
   */
  rclcpp::Time lastStamp_;

  /**
   * @brief Whether or not \a way_ has its loop closed.
   */
  bool isLoopClosed_ = false;

  /**
   * @brief The transform between global and local frame.
   */
  Eigen::Affine3d localTf_;

  /**
   * @brief Whether or not \a localTf_ is valid.
   */
  bool localTfValid_ = false;

  /**
   * @brief Filters the TriangleSet and removes all unwanted triangles.
   *
   * @param[in,out] triangulation
   */
  void filterTriangulation(TriangleSet &triangulation) const;

  /**
   * @brief Filters the Edges by their midpoints and removes all unwanted Edge(s).
   *
   * @param[in,out] edges
   * @param[in] triangulation
   */
  void filterMidpoints(EdgeSet &edges, const TriangleSet &triangulation) const;

  /**
   * @brief Runs the search backend over \a edges and stores its outcome in
   * \a way_, \a wayToPublish_ and \a isLoopClosed_. Uses \a params as its
   * parameters.
   *
   * @param[in] edges
   * @param[in] params
   */
  void computeWay(const std::vector<Edge> &edges, const Params::WayComputer::Search &params);

 public:
  /**
   * @brief Construct a new Way Computer object.
   *
   * @param[in] params
   */
  WayComputer(const Params::WayComputer &params);

  /**
   * @brief Callback of the car's state.
   *
   * @param[in] data
   */
  void stateCallback(const nav_msgs::msg::Odometry::SharedPtr data);

  /**
   * @brief Logs the search backend's cumulative statistics. Exposed so an
   * offline harness can print the summary at the end of a replay, instead of
   * having to scrape the periodic dump out of the node's log.
   */
  void reportSearchStats() const { this->search_->reportStats(); }

  /**
   * @brief Takes the Delaunay triangle set and computes the Way.
   *
   * @param[in,out] triangulation
   * @param[in] stamp
   * @param[in] unlimitedHorizon Ignore max_way_horizon_size and extend the Way
   * as far as the cone map allows. This is what produces the *complete* track
   * centerline, and it is orders of magnitude more expensive than a capped
   * search, so it is meant to be used once (on lap completion), not every
   * callback.
   */
  void update(TriangleSet &triangulation, const rclcpp::Time &stamp, bool unlimitedHorizon = false);

  /**
   * @brief Returns if the loop has been closed.
   */
  const bool &isLoopClosed() const;

  /**
   * @brief Writes the Way to the file path specified.
   *
   * @param[in] file_path
   */
  void writeWayToFile(const std::string &file_path) const;

  /**
   * @brief Returns if the attribute localTf is valid.
   */
  const bool &isLocalTfValid() const;

  /**
   * @brief Returns the transformation from global to local.
   */
  const Eigen::Affine3d &getLocalTf() const;

  /**
   * @brief Returns the centerline vector in global coordinates.
   */
  std::vector<Point> getPath() const;

  /**
   * @brief Returns the track limits in global coordinates.
   */
  Tracklimits getTracklimits() const;

  /**
   * @brief Returns the centerline and track limits in Marker Array format
   * in global coordinates.
   */
  visualization_msgs::msg::Marker getPathCenterLine() const;
  /**
   * @brief Returns the centerline and track limits in Marker Array format
   * in global coordinates.
   */
  visualization_msgs::msg::MarkerArray getPathBorderLeft() const;
  /**
   * @brief Returns the centerline and track limits in Marker Array format
   * in global coordinates.
   */
  visualization_msgs::msg::MarkerArray getPathBorderRight() const;
};