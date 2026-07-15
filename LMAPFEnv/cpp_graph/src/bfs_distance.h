#ifndef BFS_DISTANCE_H
#define BFS_DISTANCE_H

#include <cstdint>
#include <pybind11/numpy.h>

namespace py = pybind11;

/**
 * Compute a BFS distance grid from a single target position.
 *
 * Returns a numpy int16 array of shape (height, width) where each cell
 * contains the shortest-path distance to (target_x, target_y), or -1
 * if unreachable.  Respects passable_mask and the shelf-to-shelf
 * movement restriction.
 *
 * passable_mask / shelf_mask: 2D numpy arrays (bool or uint8).
 */
py::array_t<int16_t> bfs_distance_grid(
    int width, int height,
    py::array passable_mask,
    py::array shelf_mask,
    int target_x, int target_y);

#endif // BFS_DISTANCE_H
