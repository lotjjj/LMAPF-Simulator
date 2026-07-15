#pragma once

#include <vector>
#include <cstdint>
#include <utility>
#include <tuple>

namespace fast_graph {

// Forward declaration
class ReservationTable;

struct SafeInterval {
    int t_min, t_max, conflicts;
};

struct SIPPGoal {
    int x, y;
    int release_time;
};

struct SIPPResult {
    std::vector<std::pair<int, int>> path;
    int nodes_expanded = 0;
};

/// Single-agent SIPP search.
///
/// @param ct  Pre-built constraint table: (loc_key, t_min, t_max).
///            Vertex loc_key = y * map_width + x.
///            Edge loc_key   = -(to_y * w + to_x) * 10000 - (from_y * w + from_x).
/// @param existing_paths  Already-planned higher-priority agent paths
///                        (each truncated to planning_window+1 steps).
/// @param cat_data  Conflict Avoidance Table: (loc_key, timestep) entries
///                  used as soft constraints.  Paths planned by higher-
///                  priority agents are inserted here so that lower-
///                  priority agents prefer conflict-free routes.
SIPPResult cxx_sipp_search(
    int start_x, int start_y,
    const std::vector<SIPPGoal>& goals,
    const int8_t* passable_grid,
    const int8_t* shelf_grid,
    int width, int height, int map_width,
    int k_robust, int window,
    int max_time, double shelf_penalty,
    bool hold_endpoints,
    const std::vector<std::tuple<int64_t, int, int>>& ct,
    const std::vector<std::vector<std::pair<int, int>>>& existing_paths,
    const std::vector<std::pair<int64_t, int>>& cat_data = {}
);

/// SIPP search using a pre-built ReservationTable.
/// Avoids the overhead of reconstructing the RT from flat arrays.
/// Used by the batch sequential planner for multi-agent SIPP planning.
SIPPResult cxx_sipp_search_with_rt(
    int start_x, int start_y,
    const std::vector<SIPPGoal>& goals,
    const int8_t* passable_grid,
    const int8_t* shelf_grid,
    int width, int height, int map_width,
    int max_time, double shelf_penalty,
    ReservationTable& rt
);

}  // namespace fast_graph
