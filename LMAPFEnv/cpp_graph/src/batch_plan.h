#pragma once

#include <vector>
#include <cstdint>
#include <utility>
#include <string>
#include <map>

namespace fast_graph {

/// Per-agent data for batch sequential planning.
struct BatchAgent {
    int start_x, start_y;
    std::vector<std::pair<int, int>> goals;  // ordered goal sequence
};

/// Result of batch sequential planning.
struct BatchPlanResult {
    std::map<int, std::vector<std::pair<int, int>>> paths;  // agent_index → path
    bool success = false;
    int nodes_expanded = 0;
};

/// Batch sequential multi-agent planning kernel.
///
/// Plans agents one by one in the given order, accumulating constraints
/// from already-planned agents. Supports two modes:
///
/// - mode="astar": Space-time A* with vertex/edge constraint sets.
/// - mode="sipp": Safe Interval Path Planning with ReservationTable (CT/CAT).
///
/// Includes conflict detection and force-wait repair after planning.
///
/// @param agents       Per-agent data (start positions and goal sequences).
/// @param order        Planning order (indices into agents vector).
/// @param per_agent_max_time  Per-agent search depth cap (indexed by agent position).
/// @param passable_grid  Flat row-major passability grid.
/// @param shelf_grid     Flat row-major shelf grid.
/// @param width        Grid width.
/// @param height       Grid height.
/// @param mode         "astar" or "sipp".
/// @param horizon_mode A* horizon mode (A* mode only).
/// @param shelf_penalty Shelf traversal penalty.
/// @param planning_window  Conflict-free guarantee window.
/// @param k_robust     K-robust constraint expansion (SIPP mode only).
/// @param initial_ct_flat  Initial CT constraints per agent (SIPP mode only).
/// @param deadline     Wall-clock deadline (seconds since epoch); 0 = no deadline.
BatchPlanResult cxx_batch_sequential_plan(
    const std::vector<BatchAgent>& agents,
    const std::vector<int>& order,
    const std::vector<int>& per_agent_max_time,
    const int8_t* passable_grid,
    const int8_t* shelf_grid,
    int width, int height,
    const std::string& mode,
    bool horizon_mode,
    double shelf_penalty,
    int planning_window,
    int k_robust,
    const std::vector<std::vector<std::tuple<int64_t, int, int>>>& initial_ct_flat,
    double deadline
);

}  // namespace fast_graph
