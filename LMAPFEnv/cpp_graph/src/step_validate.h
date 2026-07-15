#ifndef STEP_VALIDATE_H
#define STEP_VALIDATE_H

#include <cstdint>
#include <pybind11/numpy.h>

namespace py = pybind11;

/**
 * Batch-validate agent actions for the simulation step.
 *
 * For each agent, check whether its requested action leads to a valid
 * target cell (in-bounds, passable, no shelf-to-shelf move).
 *
 * All inputs are flat int32 arrays of length num_agents.
 *
 * @param width      Map width in cells
 * @param height     Map height in cells
 * @param agv_x      Current X positions   (num_agents,)
 * @param agv_y      Current Y positions   (num_agents,)
 * @param actions    Requested actions 0-4  (num_agents,)
 * @param passable   2-D passability mask   (height, width), bool/uint8
 * @param shelf      2-D shelf mask         (height, width), bool/uint8
 * @return           uint8 array (num_agents,): 1 = feasible, 0 = stay forced
 */
py::array_t<uint8_t> validate_agent_actions(
    int width, int height,
    py::array_t<int32_t> agv_x,
    py::array_t<int32_t> agv_y,
    py::array_t<int32_t> actions,
    py::array passable,
    py::array shelf);

#endif // STEP_VALIDATE_H
