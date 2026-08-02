/**
 * @file cpu_search.cpp
 * @brief CpuSearch member function implementations.
 */

#include "modules/cpu_search.hpp"

#include <algorithm>
#include <chrono>

#include <rclcpp/rclcpp.hpp>

namespace {

/// The three per-Edge quantities the SoA needs, all in the local frame.
struct EdgeFields {
  CpuSearch::L::Vec2 mid;
  CpuSearch::L::Vec2 normal;
  uint64_t hash;
};

EdgeFields edgeFields(const Edge &e) {
  const Point mid = e.midPoint();
  const Vector normal = e.normal();
  EdgeFields f;
  f.mid = ccs::vec2<CpuSearch::scalar>(static_cast<CpuSearch::scalar>(mid.x), static_cast<CpuSearch::scalar>(mid.y));
  f.normal = ccs::vec2<CpuSearch::scalar>(static_cast<CpuSearch::scalar>(normal.x), static_cast<CpuSearch::scalar>(normal.y));
  f.hash = std::hash<Edge>{}(e);
  return f;
}

}  // namespace

/* ----------------------------- Private Methods ---------------------------- */

CpuSearch::L::SearchConsts CpuSearch::makeConsts(const Params::WayComputer::Search &params) const {
  L::SearchConsts p;
  p.search_radius = static_cast<CpuSearch::scalar>(params.search_radius);
  p.max_angle_diff = static_cast<CpuSearch::scalar>(params.max_angle_diff);
  p.edge_len_diff_factor = static_cast<CpuSearch::scalar>(params.edge_len_diff_factor);
  p.max_next_heuristic = static_cast<CpuSearch::scalar>(params.max_next_heuristic);
  p.heur_dist_ponderation = params.heur_dist_ponderation;  // stays float, see search_types.hpp
  p.allow_intersection = params.allow_intersection;
  p.max_search_options = params.max_search_options;
  p.max_search_tree_height = params.max_search_tree_height;
  if (p.max_search_tree_height > CCS_MAX_TRACE_LEN) {
    RCLCPP_WARN(rclcpp::get_logger("local_planner"),
                "max_search_tree_height (%d) exceeds CCS_MAX_TRACE_LEN (%d), clamping.",
                p.max_search_tree_height, CCS_MAX_TRACE_LEN);
    p.max_search_tree_height = CCS_MAX_TRACE_LEN;
  }
  p.max_dist_loop_closure = static_cast<CpuSearch::scalar>(this->wayParams_.max_dist_loop_closure);
  p.max_angle_diff_loop_closure = static_cast<CpuSearch::scalar>(this->wayParams_.max_angle_diff_loop_closure);
  p.min_loop_size = MIN_LOOP_SIZE;
  return p;
}

void CpuSearch::buildEdgeSoA(const std::vector<Edge> &edges) {
  this->edgeSoa_.clear();
  this->edgeSoa_.reserve(edges.size());

  std::vector<Point> midpoints;
  midpoints.reserve(edges.size());

  for (const Edge &e : edges) {
    const EdgeFields f = edgeFields(e);
    this->edgeSoa_.push(f.mid, f.normal, static_cast<CpuSearch::scalar>(e.len), f.hash);
    // Seed the k-d tree from the SoA midpoints, not from the Edge(s): the query
    // centre comes from the SoA, so the index has to agree with it. At
    // scalar == double these are the same values; at float32 they are not,
    // and using the Edge would make the host see a geometry the device never
    // will.
    midpoints.push_back(Point(f.mid.x, f.mid.y));
  }

  // Build a k-d tree of all midpoints so it is cheaper to find close points.
  this->midpointsKDT_ = KDTree(midpoints);
}

void CpuSearch::resetWaySoA(const Way &way) {
  this->waySoa_.clear();
  for (const Edge &e : way.edges()) {
    const EdgeFields f = edgeFields(e);
    this->waySoa_.push(f.mid, f.normal, f.hash);
  }
  this->waySoa_.setAvgEdgeLen(static_cast<CpuSearch::scalar>(way.getAvgEdgeLen()));
}

void CpuSearch::appendToWaySoA(const Edge &edge, const Way &way) {
  const EdgeFields f = edgeFields(edge);
  this->waySoa_.push(f.mid, f.normal, f.hash);
  this->waySoa_.setAvgEdgeLen(static_cast<CpuSearch::scalar>(way.getAvgEdgeLen()));
}

CpuSearch::L::SearchContext CpuSearch::makeContext(const L::Trace *trace) const {
  const L::EdgeSoA e = this->edgeSoa_.view();
  const L::WaySoA w = this->waySoa_.view();

  L::SearchContext c;
  c.hasActEdge = false;
  c.actEdgeMid = ccs::vec2<scalar>(0, 0);
  c.actEdgeNormal = ccs::vec2<scalar>(0, 0);
  c.actEdgeHash = 0;
  c.actPos = ccs::vec2<scalar>(0, 0);
  c.lastPos = ccs::vec2<scalar>(0, 0);

  // Set actual and last position.
  if (trace and trace->size > 0) {
    const uint32_t i = ccs::traceLast(*trace);
    c.hasActEdge = true;
    c.actEdgeMid = e.mid[i];
    c.actEdgeNormal = e.normal[i];
    c.actEdgeHash = e.hash[i];
    c.actPos = e.mid[i];
    if (trace->size >= 2) c.lastPos = e.mid[ccs::tracePrev(*trace)];
  }
  if (w.size > 0) {
    if (not c.hasActEdge) {
      const uint32_t b = w.size - 1;
      c.hasActEdge = true;
      c.actEdgeMid = w.mid[b];
      c.actEdgeNormal = w.normal[b];
      c.actEdgeHash = w.hash[b];
      c.actPos = w.mid[b];
      if (w.size >= 2) c.lastPos = w.mid[w.size - 2];
    } else if (trace->size < 2) {
      c.lastPos = w.mid[w.size - 1];
    }
  }

  // Set dir vector. The condition avoids setting a vector pointing backwards.
  if (c.hasActEdge and (w.size >= 2 or trace))
    c.dir = ccs::vecFromTo(c.lastPos, c.actPos);
  else
    c.dir = ccs::vec2<scalar>(1, 0);

  c.avgEdgeLen = ccs::combinedAvgEdgeLen(w, trace);
  c.hasTrace = (trace != nullptr);
  c.traceLoopClosed = trace ? trace->loopClosed : false;
  c.sideCheckEnabled = (w.size >= 2 or trace != nullptr);

  return c;
}

void CpuSearch::findNextEdges(std::vector<HeurIdx> &out, const L::Trace *trace, const L::SearchConsts &p) {
  out.clear();

  const L::EdgeSoA e = this->edgeSoa_.view();
  const L::WaySoA w = this->waySoa_.view();
  const L::SearchContext c = this->makeContext(trace);

  // Find all possible edges in a specified radius.
  const indexArr candidates =
      this->midpointsKDT_.neighborhood_indices(Point(c.actPos.x, c.actPos.y), p.search_radius);

  // Discard by specification, then keep only the ones whose heuristic is small
  // enough. This is the whole parallel workload: ~114k candidate evaluations
  // per callback on a 112-cone track.
  ++this->stats_.bfsNodes;
  this->stats_.candidateEvals += candidates.size();

  this->privilegeRunner_.clear();
  this->privilegeRunner_.reserve(candidates.size());
  for (const size_t &idx : candidates) {
    const uint32_t i = static_cast<uint32_t>(idx);
    if (ccs::discardCandidate(c, w, e.mid[i], e.len[i], e.hash[i], p)) continue;
    const scalar h = ccs::heuristic(c.actPos, e.mid[i], c.dir, p);
    if (h <= p.max_next_heuristic) this->privilegeRunner_.emplace_back(h, i);
  }

  // Copy the n best into out. Ordering is by (heuristic, index) and therefore
  // total, so it does not depend on the order the radius query returned.
  out.resize(std::min(this->privilegeRunner_.size(), static_cast<size_t>(p.max_search_options)));
  std::partial_sort_copy(this->privilegeRunner_.begin(), this->privilegeRunner_.end(), out.begin(), out.end());
}

uint32_t CpuSearch::treeSearch(const std::vector<HeurIdx> &seeds, const L::SearchConsts &p, float timeLimit) {
  const L::EdgeSoA e = this->edgeSoa_.view();
  const L::WaySoA w = this->waySoa_.view();
  const L::Vec2 wayBack = ccs::wayBackMid(w);

  // FIFO: the search is level-synchronous BFS, bounded at max_search_tree_height
  // levels and max_search_options branches. head_ is the pop cursor; the vector
  // is never erased from, so no reallocation churn per level.
  this->queue_.clear();
  for (const HeurIdx &seed : seeds) {
    L::Trace t;
    ccs::traceInit(t);
    const bool closesLoop = ccs::wayClosesLoopWith(w, e.mid[seed.second], wayBack, p);
    ccs::traceAppend(t, seed.second, seed.first, e.len[seed.second], closesLoop);
    this->queue_.push_back(t);
  }

  // The provisional best is the front one.
  L::Trace best = this->queue_.front();

  // Get the longest (and best) path, stopping if the time limit is exceeded.
  //
  // steady_clock, not rclcpp::Clock: the original constructed a fresh
  // rclcpp::Clock on every pop, which measured at 0.24 us/node on an Orin --
  // ~6% of the whole search, spent on rcl_clock_init rather than on the search.
  // A steady_clock read is ~25 ns and needs no construction. This is also what
  // makes two backends comparable: the time limit has to cost the same in both,
  // or the comparison measures the bail-out mechanism instead of the search.
  const auto searchBeginTime = std::chrono::steady_clock::now();
  const auto searchTimeLimit = std::chrono::duration<double>(timeLimit);
  size_t head = 0;
  while (head < this->queue_.size()) {
    if (std::chrono::steady_clock::now() - searchBeginTime > searchTimeLimit) {
      RCLCPP_WARN(rclcpp::get_logger("local_planner"), "Tree search time limit exceeded.");
      ++this->stats_.timeLimitHits;
      break;
    }
    const L::Trace t = this->queue_[head++];

    bool trace_at_max_height = false;
    if (t.size >= static_cast<uint32_t>(p.max_search_tree_height))
      trace_at_max_height = true;
    else
      this->findNextEdges(this->childEdges_, &t, p);

    if (trace_at_max_height or this->childEdges_.empty()) {
      // This trace is finished, it is a candidate for "best":
      // 1. the longest trace wins;
      // 2. on a tie, the smallest accumulated heuristic wins.
      if (t.size > best.size or (t.size == best.size and t.sumHeur < best.sumHeur)) best = t;
    } else {
      const L::Vec2 actPos = e.mid[ccs::traceLast(t)];
      for (const HeurIdx &child : this->childEdges_) {
        const bool closesLoop = ccs::wayClosesLoopWith(w, e.mid[child.second], actPos, p);
        L::Trace aux = t;
        ccs::traceAppend(aux, child.second, child.first, e.len[child.second], closesLoop);
        this->queue_.push_back(aux);
      }
    }
  }

  // The next point will be the FIRST point of the best path.
  return ccs::traceFirst(best);
}

/* ----------------------------- Public Methods ----------------------------- */

CpuSearch::CpuSearch(const Params::WayComputer &params) : wayParams_(params.way) {}

ISearch::Result CpuSearch::computeWay(const std::vector<Edge> &edges,
                                      const Params::WayComputer::Search &params,
                                      Way &way) {
  Result res;
  const CallbackTimer timer(this->stats_);

  // Get rid of all edges from closest to car (included) to last.
  way.trimByLocal();

  const L::SearchConsts p = this->makeConsts(params);
  this->buildEdgeSoA(edges);
  this->resetWaySoA(way);

  // Get first set of possible Edges.
  this->findNextEdges(this->nextEdges_, nullptr, p);

  // Main outer loop: irreducibly sequential, every iteration appends exactly
  // one midpoint to the Way and every filter downstream reads that Way.
  while (rclcpp::ok() and not this->nextEdges_.empty() and
         (!params.max_way_horizon_size or
          way.sizeAheadOfCar() <= static_cast<uint32_t>(params.max_way_horizon_size))) {
    ++this->stats_.outerIters;
    this->stats_.waySizeSum += this->waySoa_.size();
    const uint32_t nextEdgeInd = this->treeSearch(this->nextEdges_, p, params.max_treeSearch_time);

    // Append the new Edge to both representations. The Way stays the authority.
    const Edge &nextEdge = edges[nextEdgeInd];
    way.addEdge(nextEdge);
    this->appendToWaySoA(nextEdge, way);

    // Check for loop closure.
    if (way.closesLoop()) {
      res.wayToPublish = way.restructureClosure();
      res.loopClosed = true;
      return res;
    }

    // Get next set of possible edges.
    this->findNextEdges(this->nextEdges_, nullptr, p);
  }

  res.loopClosed = false;
  res.wayToPublish = way;
  return res;
}
