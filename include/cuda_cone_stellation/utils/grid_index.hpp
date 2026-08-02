/**
 * @file grid_index.hpp
 * @brief Uniform-grid radius query over 2-D points. Allocation-free per query.
 *
 * This replaces KDTree for the search's inner radius query. The k-d tree is not
 * a bad algorithm here, but its implementation is the search's dominant cost:
 * KDTree::neighborhood_ returns a std::vector<std::pair<Point,size_t>> BY VALUE
 * at every recursion level and merges with insert(), so one query does heap
 * traffic proportional to the nodes it visits and then copies the results back
 * up the recursion; neighborhood_indices() allocates a third vector to strip
 * the points out again; and the nodes are shared_ptr-linked, so the traversal
 * is pointer chasing with refcount traffic. Measured on an Orin at N=300 and
 * radius 5 m, one query costs 3.2 us against 0.65 us for the entire filter and
 * heuristic workload it feeds -- and a flat brute-force scan over the same
 * points costs 1.2 us, i.e. having no index at all beats that k-d tree.
 *
 * A grid fits this problem better than a tree regardless of implementation
 * quality: the query radius is a CONSTANT (search_radius) known at build time,
 * so cells can be sized to it and every query touches a fixed, small number of
 * them. The tree's advantage -- adapting to arbitrary query radii and to
 * clustered data -- is worth nothing here, and it pays for it with pointer
 * chasing.
 *
 * SET EQUIVALENCE: query() returns exactly the points KDTree::neighborhood_
 * would, including on the boundary. It compares the same quantity, dx*dx+dy*dy,
 * against the same r2 = radius*radius, both in double, from the same
 * coordinates -- so a point at exactly the radius lands the same way in both.
 * The ORDER differs, which is fine and checked: findNextEdges ranks candidates
 * by (heuristic, index), a total order, and takes the best n with
 * partial_sort_copy, so its output does not depend on the order the query
 * returned them in.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ccs {

class GridIndex {
 public:
  /**
   * @brief (Re)builds the index over \a n points. Reuses its storage, so the
   * steady state does not allocate.
   *
   * Coordinates are taken in double regardless of the backend's scalar,
   * because the radius test has to be the one the reference backend performs.
   *
   * @param[in] radius the query radius every query() will use. Cells are sized
   * to it, so a query touches at most a 3x3 block.
   */
  void build(const double *xs, const double *ys, uint32_t n, double radius) {
    this->n_ = n;
    this->r2_ = radius * radius;
    this->xs_.assign(xs, xs + n);
    this->ys_.assign(ys, ys + n);

    if (n == 0) {
      this->nx_ = this->ny_ = 0;
      return;
    }

    double minX = xs[0], maxX = xs[0], minY = ys[0], maxY = ys[0];
    for (uint32_t i = 1; i < n; ++i) {
      minX = std::min(minX, xs[i]);
      maxX = std::max(maxX, xs[i]);
      minY = std::min(minY, ys[i]);
      maxY = std::max(maxY, ys[i]);
    }
    this->minX_ = minX;
    this->minY_ = minY;

    // A degenerate radius would produce an unbounded grid; fall back to a
    // single cell, which makes query() a brute-force scan (still correct).
    this->cell_ = (radius > 0.0) ? radius : std::max(maxX - minX, maxY - minY) + 1.0;
    if (not(this->cell_ > 0.0)) this->cell_ = 1.0;

    this->nx_ = static_cast<int32_t>((maxX - minX) / this->cell_) + 1;
    this->ny_ = static_cast<int32_t>((maxY - minY) / this->cell_) + 1;

    // Counting sort into CSR: cellStart_[c]..cellStart_[c+1] indexes items_.
    const size_t nCells = static_cast<size_t>(this->nx_) * static_cast<size_t>(this->ny_);
    this->cellStart_.assign(nCells + 1, 0);
    this->cellOf_.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t c = this->cellIndex(xs[i], ys[i]);
      this->cellOf_[i] = c;
      ++this->cellStart_[c + 1];
    }
    for (size_t c = 0; c < nCells; ++c) this->cellStart_[c + 1] += this->cellStart_[c];

    this->items_.resize(n);
    this->fill_ = this->cellStart_;  // copy of the offsets, consumed as cursors
    for (uint32_t i = 0; i < n; ++i) this->items_[this->fill_[this->cellOf_[i]]++] = i;
  }

  /**
   * @brief Replaces \a out with the indices of every point within the build
   * radius of (\a qx, \a qy).
   */
  void query(double qx, double qy, std::vector<uint32_t> &out) const {
    out.clear();
    if (this->n_ == 0) return;

    const double radius = std::sqrt(this->r2_);

    // Derive the cell range from the query BOX, not from the query point's
    // cell. Clamping a centre that falls outside the grid would silently drop
    // the cells on the far side of it.
    const int32_t gxLo = this->clampX(static_cast<int32_t>(std::floor((qx - radius - this->minX_) / this->cell_)));
    const int32_t gxHi = this->clampX(static_cast<int32_t>(std::floor((qx + radius - this->minX_) / this->cell_)));
    const int32_t gyLo = this->clampY(static_cast<int32_t>(std::floor((qy - radius - this->minY_) / this->cell_)));
    const int32_t gyHi = this->clampY(static_cast<int32_t>(std::floor((qy + radius - this->minY_) / this->cell_)));

    for (int32_t gy = gyLo; gy <= gyHi; ++gy) {
      const size_t rowBase = static_cast<size_t>(gy) * static_cast<size_t>(this->nx_);
      for (int32_t gx = gxLo; gx <= gxHi; ++gx) {
        const size_t c = rowBase + static_cast<size_t>(gx);
        const uint32_t begin = this->cellStart_[c], end = this->cellStart_[c + 1];
        for (uint32_t k = begin; k < end; ++k) {
          const uint32_t i = this->items_[k];
          const double dx = this->xs_[i] - qx, dy = this->ys_[i] - qy;
          // Same expression, same order, same type as KDTree::dist2 -- see the
          // set-equivalence note at the top of this file.
          if (dx * dx + dy * dy <= this->r2_) out.push_back(i);
        }
      }
    }
  }

 private:
  uint32_t cellIndex(double x, double y) const {
    const int32_t gx = this->clampX(static_cast<int32_t>((x - this->minX_) / this->cell_));
    const int32_t gy = this->clampY(static_cast<int32_t>((y - this->minY_) / this->cell_));
    return static_cast<uint32_t>(static_cast<size_t>(gy) * static_cast<size_t>(this->nx_) +
                                 static_cast<size_t>(gx));
  }

  int32_t clampX(int32_t g) const { return g < 0 ? 0 : (g >= this->nx_ ? this->nx_ - 1 : g); }
  int32_t clampY(int32_t g) const { return g < 0 ? 0 : (g >= this->ny_ ? this->ny_ - 1 : g); }

  uint32_t n_ = 0;
  double minX_ = 0, minY_ = 0, cell_ = 1, r2_ = 0;
  int32_t nx_ = 0, ny_ = 0;
  std::vector<double> xs_, ys_;
  std::vector<uint32_t> cellStart_, items_, fill_, cellOf_;
};

}  // namespace ccs
