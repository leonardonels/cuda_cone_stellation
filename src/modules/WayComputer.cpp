/**
 * @file WayComputer.cpp
 * @author Oriol Gorriz (origovi2000@gmail.com)
 * @brief Contains the WayComputer class member functions implementation
 * @version 1.0
 * @date 2022-10-31
 *
 * @copyright Copyright (c) 2022 BCN eMotorsport
 */

#include "modules/WayComputer.hpp"

#include "modules/search_factory.hpp"

/* ----------------------------- Private Methods ---------------------------- */

void WayComputer::filterTriangulation(TriangleSet &triangulation) const {
  auto it = triangulation.begin();
  while (it != triangulation.end()) {
    bool removeTriangle = false;
    // Look for edges longer than accepted
    for (const Edge &e : it->edges) {
      if (e.len > this->params_.max_triangle_edge_len) {
        removeTriangle = true;
        break;
      }
    }

    // Look for angles smaller than accepted
    if (!removeTriangle) {
      for (const double &angle : it->angles()) {
        if (angle < this->params_.min_triangle_angle) {
          removeTriangle = true;
          break;
        }
      }
    }

    if (removeTriangle)
      it = triangulation.erase(it);
    else
      it++;
  }
}

void WayComputer::filterMidpoints(EdgeSet &edges, const TriangleSet &triangulation) const {
  // Build a k-d tree of all circumcenters so finding matches is O(logn)
  std::vector<Point> circums(triangulation.size());
  std::transform(triangulation.begin(), triangulation.end(), circums.begin(),
                 [](const Triangle &t) -> const Point & { return t.circumCenter(); });
  KDTree circumKDTree(circums);

  // Perform the filtering
  auto it = edges.begin();
  while (it != edges.end()) {
    Point midPoint = it->midPoint();
    KDTData<size_t> nearestCC = circumKDTree.nearest_index(midPoint);
    if (bool(nearestCC) and Point::distSq(circums[*nearestCC], midPoint) > pow(this->params_.max_dist_circum_midPoint, 2)) {
      it = edges.erase(it);
    } else {
      it++;
    }
  }
}

void WayComputer::computeWay(const std::vector<Edge> &edges, const Params::WayComputer::Search &params) {
  ISearch::Result res = this->search_->computeWay(edges, params, this->way_);
  this->isLoopClosed_ = res.loopClosed;
  this->wayToPublish_ = res.wayToPublish;
}

/* ----------------------------- Public Methods ----------------------------- */

WayComputer::WayComputer(const Params::WayComputer &params) : params_(params), search_(createSearch(params)) {
  Way::init(params.way);
  this->generalFailsafe_.initGeneral(this->params_.search, this->params_.general_failsafe_safetyFactor, this->params_.failsafe_max_way_horizon_size);
}
/*
void WayComputer::stateCallback(const nav_msgs::msg::Odometry::SharedPtr &data) {
  geometry_msgs::msg::Pose pose;
  pose.position = data->pose.pose.position;
  tf2::Quaternion qAux;
  qAux.setRPY(0.0, 0.0, tf2::getYaw(data->pose.pose.orientation));
  pose.orientation = tf2::toMsg(qAux);
  this->localTf_ = tf2::transformToEigen(pose);

  this->localTf_ = this->localTf_.inverse();

  this->localTfValid_ = true;
}
*/

void WayComputer::stateCallback(const nav_msgs::msg::Odometry::SharedPtr odom) {
    geometry_msgs::msg::Pose pose;
    pose.position = odom->pose.pose.position;

    // Extract yaw from the existing quaternion orientation
    tf2::Quaternion q_orig;
    tf2::fromMsg(odom->pose.pose.orientation, q_orig);

    double roll, pitch, yaw;
    tf2::Matrix3x3(q_orig).getRPY(roll, pitch, yaw);

    // Recreate the quaternion with zero roll and pitch,
    // preserving only the yaw to mimic the original logic
    tf2::Quaternion qAux;
    qAux.setRPY(0.0, 0.0, yaw);
    pose.orientation = tf2::toMsg(qAux);

    Eigen::Isometry3d eigen_pose;
    tf2::fromMsg(pose, eigen_pose);

    // Invert the transform
    this->localTf_ = eigen_pose.inverse();

    // Mark as valid
    this->localTfValid_ = true;
}


void WayComputer::update(TriangleSet &triangulation, const rclcpp::Time &stamp, bool unlimitedHorizon) {
  if (not this->localTfValid_) {
    RCLCPP_WARN(rclcpp::get_logger("local_planner"), "CarState not being received.");
    return;
  }

  // #0: Update last way (this will be used to calculate the raplan flag).
  //     And update stamp.
  this->lastWay_ = this->way_;
  this->lastStamp_ = stamp;

  // #1: Remove all triangles which we know will not be part of the track.
  this->filterTriangulation(triangulation);

  // #2: Extract all midpoints without repetitions, do that through an EdgeSet
  // so no midpoint is got twice.
  EdgeSet edgeSet;
  for (const Triangle &t : triangulation) {
    for (const Edge &e : t.edges) {
      edgeSet.insert(e);
    }
  }

  // #3: Filter the midpoints. Only the ones having a circumcenter near, will be
  // left.
  this->filterMidpoints(edgeSet, triangulation);

  // Convert this set to a vector
  std::vector<Edge> edgeVec;
  edgeVec.reserve(edgeSet.size());
  for (const Edge &e : edgeSet) {
    edgeVec.push_back(e);
  }

  // #4: Update all local positions (way and edges) with car tf
  this->way_.updateLocal(this->localTf_);
  for (const Edge &e : edgeVec) {
    e.updateLocal(this->localTf_);
  }

  // #5: Perform the search through the midpoints in order to obtain a way
  //     using normal parameters.
  Params::WayComputer::Search searchParams = this->params_.search;
  if (unlimitedHorizon) searchParams.max_way_horizon_size = 0;
  this->computeWay(edgeVec, searchParams);

  // #6: Check failsafe(s)
  if (this->params_.general_failsafe and this->way_.sizeAheadOfCar() < MIN_FAILSAFE_WAY_SIZE and !this->isLoopClosed_) {
    RCLCPP_WARN(rclcpp::get_logger("local_planner"), "GENERAL FAILSAFE ACTIVATED!");
    this->computeWay(edgeVec, this->generalFailsafe_);
  }

  // #7: Visualize
  Visualization::getInstance().setTimestamp(stamp);
  Visualization::getInstance().visualize(edgeSet);
  Visualization::getInstance().visualize(triangulation);
  Visualization::getInstance().visualize(this->wayToPublish_);
}

const bool &WayComputer::isLoopClosed() const {
  return this->isLoopClosed_;
}

void WayComputer::writeWayToFile(const std::string &file_path) const {
  std::ofstream oStreamToWrite(file_path);
  oStreamToWrite << this->wayToPublish_;
  oStreamToWrite.close();
}

const bool &WayComputer::isLocalTfValid() const {
  return this->localTfValid_;
}

 const Eigen::Affine3d &WayComputer::getLocalTf() const {
  return this->localTf_;
}

std::vector<Point> WayComputer::getPath() const {
  return this->wayToPublish_.getPath();
}

Tracklimits WayComputer::getTracklimits() const {
  return this->wayToPublish_.getTracklimits();
}

visualization_msgs::msg::Marker WayComputer::getPathCenterLine() const {
  visualization_msgs::msg::Marker marker;

  // Get the centerline path as a vector of Points.
  std::vector<Point> path = this->wayToPublish_.getPath();
  std::vector<double> point_dist_from_borders = this->wayToPublish_.getPointDistanceFromBorders();

  // Set up header information.
  marker.header.frame_id = "track";
  marker.header.stamp = this->lastStamp_;
  marker.ns = "path_center_line";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  marker.action = visualization_msgs::msg::Marker::ADD;

  // For a LINE_STRIP marker, only scale.x is used (as the width of the line).
  marker.scale.x = 0.3;

  // Set color to red.
  marker.color.r = 1.0;
  marker.color.g = 0.0;
  marker.color.b = 0.0;
  marker.color.a = 1.0;

  // Set the marker's pose to identity. The points are specified relative to the header frame.
  marker.pose.position.x = 0.0;
  marker.pose.position.y = 0.0;
  marker.pose.position.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;

  int index = 0;
  // Add each point in the path to the marker's points vector.
  for (const Point &p : path) {
    geometry_msgs::msg::Point point = p.gmPoint();
    point.z = point_dist_from_borders[index];
    marker.points.push_back(point);
    index++;
  }

  return marker;
}


visualization_msgs::msg::MarkerArray WayComputer::getPathBorderLeft() const {

  visualization_msgs::msg::MarkerArray res;
  
  // Fill Tracklimits
  Tracklimits tracklimits = this->wayToPublish_.getTracklimits();
  // res.marker.stamp = res.stamp;
  // tracklimits.left.reserve(tracklimits.first.size());

  for (const Node &n : tracklimits.first) {
    res.markers.push_back(n.cone());
  }

  // res.tracklimits.replan indicates if the n midpoints in front of the car
  // have varied from last iteration
  // res.tracklimits.replan = this->way_.quinEhLobjetiuDeLaSevaDiresio(this->lastWay_);
 return res;
}
visualization_msgs::msg::MarkerArray WayComputer::getPathBorderRight() const {

  visualization_msgs::msg::MarkerArray res;
  
  // Fill Tracklimits
  Tracklimits tracklimits = this->wayToPublish_.getTracklimits();

  // res.tracklimits.stamp = res.stamp;
  // tracklimits.left.reserve(tracklimits.second.size());

  for (const Node &n : tracklimits.second) {
    res.markers.push_back(n.cone());
  }

  // res.tracklimits.replan indicates if the n midpoints in front of the car
  // have varied from last iteration
  // res.tracklimits.replan = this->way_.quinEhLobjetiuDeLaSevaDiresio(this->lastWay_);
 return res;
}