#include "autocross.h"
#include <iomanip>
#include <sstream>
#include <cstdio>

/// @brief constructor
/// @param nh node handler
/// @param centerLinePub centerline publisher
/// @param centerLineCompletedPub complete centerline publisher
AutocrossPlanner::AutocrossPlanner(const rclcpp::Node::SharedPtr &nh,
								   const rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr &centerLinePub,
								   const rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr &centerLineCompletedPub)
{
	this->nh = nh;

  // Notice how we don't care about the borders
	this->centerLinePub = centerLinePub;
	this->centerLineCompletedPub = centerLineCompletedPub;

	this->currentLap = 0;

	this->params = new Params(nh);
	this->wayComputer = new WayComputer(params->wayComputer);
	Visualization::getInstance().init(nh, params->visualization);

}


/// @brief callback to get the race status from the subscription
/// @param raceStatus custom message
void AutocrossPlanner::raceStatusCallBack(mmr_base::msg::RaceStatus::SharedPtr raceStatus)
{

	this->currentLap = raceStatus->current_lap;
}

void AutocrossPlanner::slamConesCallback(visualization_msgs::msg::Marker::SharedPtr slamCones)
{
	RCLCPP_DEBUG(rclcpp::get_logger(""), "[local_planner] slamConesCallback");

  if(this->idle)
  {
    RCLCPP_DEBUG(rclcpp::get_logger(""), "[local_planner] idle");
    return;
  }

  if (!this->wayComputer->isLocalTfValid())
  {
    RCLCPP_WARN_THROTTLE(rclcpp::get_logger(""), *this->nh->get_clock(), 2000, "[local_planner] CarState not being received.");
    return;
  }
  if (slamCones->points.empty())
  {
    RCLCPP_WARN_THROTTLE(rclcpp::get_logger(""), *this->nh->get_clock(), 2000, "[local_planner] reading empty set of cones.");
    return;
  }

  if (this->params->main.logging) Time::tick("computation"); // Start measuring time

  const bool lapComplete = this->currentLap > 1;

  // Fallback path only: rebuild the FULL track centerline from the whole cone
  // map with an unlimited horizon. This is orders of magnitude more expensive
  // than a racing callback, so it is used only if the complete trajectory was
  // not already captured at loop closure (see below).
  const bool finalPass = lapComplete and not this->fullTrajectoryPublished;

  // Convert to Node vector, move them to local coordinates and crop to a window
  // around the car. The SLAM map is cumulative, so without this crop the cost of
  // the whole pipeline keeps growing with the track length, lap after lap.
  const double maxConeDist = finalPass ? 0.0 : this->params->main.max_cone_distance;
  const double maxConeDistSq = maxConeDist * maxConeDist;
  const Eigen::Affine3d &localTf = this->wayComputer->getLocalTf();

  uint32_t id = 0;
  std::vector<Node> nodes;
  nodes.reserve(slamCones->points.size());
  for (const geometry_msgs::msg::Point &c : slamCones->points)
  {
    // NOTE: id is deliberately incremented for every cone of the map, cropped
    // or not. It is the cone's identity across callbacks: Edge hashes and
    // Way::containsEdge rely on it being stable.
    Node n(c.x, c.y, c.x, c.y, id++);
    n.updateLocal(localTf);
    if (maxConeDist > 0.0 and n.x() * n.x() + n.y() * n.y() > maxConeDistSq) continue;
    nodes.push_back(n);
  }

  RCLCPP_DEBUG(rclcpp::get_logger(""), "[local_planner] %zu cones in map, %zu after crop",
               slamCones->points.size(), nodes.size());

  // Delaunay triangulation
  TriangleSet triangles = DelaunayTri::compute(nodes);

  RCLCPP_DEBUG(rclcpp::get_logger(""), "[local_planner] the size of triangles is %ld", triangles.size());

  // Update the way with the new triangulation
  this->wayComputer->update(triangles, slamCones->header.stamp, finalPass);

  // The Way closes its loop while the car is still short of the finish line.
  // At that instant WayComputer has already run restructureClosure(), so the
  // published Way IS the complete track -- no extra computation needed. It only
  // stays that way for a handful of callbacks: afterwards the Way is trimmed
  // back to the car and re-extended, and the complete trajectory is lost. So
  // capture it here, the first time it appears.
  bool publishedThisCallback = false;
  if (this->params->main.publish_full_trajectory_on_loop_closure and
      not this->fullTrajectoryPublished and this->wayComputer->isLoopClosed())
  {
    this->fullTrajectory = this->wayComputer->getPathCenterLine();
    this->fullTrajectoryPublished = true;
    this->centerLineCompletedPub->publish(this->fullTrajectory); // transient local topic
    publishedThisCallback = true;
    RCLCPP_INFO(rclcpp::get_logger(""),
                "[local_planner] loop closed, complete trajectory (%zu points) published early",
                this->fullTrajectory.points.size());
    // The trajectory is out, so the run's search work is done and its totals
    // are final: this is the moment the summary means something.
    if (this->params->main.debug) this->wayComputer->reportSearchStats();
  }

  // Lap complete: become idle. The complete trajectory is normally already out
  // (published at loop closure above); otherwise fall back to whatever the
  // unlimited-horizon pass just produced.
 if (lapComplete)
	{

		this->idle = true;
		if (not publishedThisCallback)
		{
			this->centerLineCompletedPub->publish(this->fullTrajectoryPublished
			                                          ? this->fullTrajectory
			                                          : this->wayComputer->getPathCenterLine());
			if (this->params->main.debug) this->wayComputer->reportSearchStats();
		}
    return;
	}

  // Publish partial centerline only if it is not empty
  else
  {
    visualization_msgs::msg::Marker current_centerline = this->wayComputer->getPathCenterLine();
    if(current_centerline.points.size() > 0)
    {
      this->centerLinePub->publish(current_centerline);
    }
    else
    {
      RCLCPP_DEBUG(rclcpp::get_logger(""), "[local_planner] current centerline is empty");
    }
  }

  if (this->params->main.logging) Time::tock("computation"); // End measuring time
}


