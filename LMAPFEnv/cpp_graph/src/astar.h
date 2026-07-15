#pragma once

#include <vector>
#include <cstdint>
#include <utility>
#include <functional>
#include <cstddef>
#include <unordered_set>

namespace fast_graph {

// ── Data structures ──────────────────────────────────────────────────────────

struct Position {
    int x, y;
    bool operator==(const Position& o) const noexcept { return x == o.x && y == o.y; }
    bool operator!=(const Position& o) const noexcept { return !(*this == o); }
};

struct PositionHash {
    size_t operator()(const Position& p) const noexcept {
        return (static_cast<size_t>(p.x) << 16) ^ static_cast<size_t>(p.y);
    }
};

/// (position, time) state for space-time A*
struct SpaceTimeState {
    Position pos;
    int t;
    bool operator==(const SpaceTimeState& o) const noexcept {
        return pos == o.pos && t == o.t;
    }
};

struct SpaceTimeStateHash {
    size_t operator()(const SpaceTimeState& s) const noexcept {
        return PositionHash()(s.pos) ^ (static_cast<size_t>(s.t) << 24);
    }
};

/// Edge constraint: cannot move from->to at time t
struct EdgeConstraint {
    Position from, to;
    int t;
    bool operator==(const EdgeConstraint& o) const noexcept {
        return from == o.from && to == o.to && t == o.t;
    }
};

struct EdgeConstraintHash {
    size_t operator()(const EdgeConstraint& e) const noexcept {
        return PositionHash()(e.from) ^ (PositionHash()(e.to) << 8) ^
               (static_cast<size_t>(e.t) << 24);
    }
};

// ── Options ──────────────────────────────────────────────────────────────────

struct AStarOptions {
    int max_time = 500;           // max search depth (time steps)
    bool horizon_mode = false;    // if true: return best-effort path at max_time
    bool use_closed_set = true;   // prune revisited states
    bool tie_breaker_by_depth = true;  // true: prefer deeper (-g); false: prefer closer (h)
    double shelf_penalty = 4.0;   // penalty for traversing shelf cells
};

// ── Constraint containers ────────────────────────────────────────────────────

using VertexConstraintSet =
    std::unordered_set<SpaceTimeState, SpaceTimeStateHash>;
using EdgeConstraintSet =
    std::unordered_set<EdgeConstraint, EdgeConstraintHash>;

// ── Result ───────────────────────────────────────────────────────────────────

struct AStarResult {
    std::vector<std::pair<int, int>> path;  // sequence of (x, y)
    int nodes_expanded = 0;
};

// ── Public API ───────────────────────────────────────────────────────────────

/// Unified space-time A* search with multi-goal support.
///
/// @param start          (x, y) start position
/// @param goals          sequence of (x, y) goal positions; search visits them in order
/// @param passable_grid  flat row-major bool grid (1 = passable)
/// @param shelf_grid     flat row-major bool grid (1 = shelf)
/// @param width          grid width
/// @param height         grid height
/// @param vertex_constraints  (position, time) constraints
/// @param edge_constraints     (from, to, time) constraints
/// @param options        search options
///
/// When goals has > 1 entry, the search tracks a goal_id in its state.
/// Upon reaching goals[goal_id], goal_id advances. The search terminates
/// when all goals are visited. The heuristic estimates cost through the
/// entire remaining goal chain.
AStarResult cxx_astar(
    std::pair<int, int> start,
    const std::vector<std::pair<int, int>>& goals,
    const std::vector<int8_t>& passable_grid,
    const std::vector<int8_t>& shelf_grid,
    int width,
    int height,
    const VertexConstraintSet& vertex_constraints,
    const EdgeConstraintSet& edge_constraints,
    const AStarOptions& options
);

/// Zero-copy A* API: accepts raw pointers for grid data and flat int arrays
/// for constraints, avoiding pybind11 type conversion overhead.
///
/// @param passable_ptr / shelf_ptr  raw int8 pointers to row-major grid data
/// @param vc_flat  flat array [x0, y0, t0, x1, y1, t1, ...] (length = 3*vc_count)
/// @param ec_flat  flat array [fx0, fy0, tx0, ty0, t0, ...] (length = 5*ec_count)
AStarResult cxx_astar_nocopy(
    std::pair<int, int> start,
    const std::vector<std::pair<int, int>>& goals,
    const int8_t* passable_ptr,
    const int8_t* shelf_ptr,
    int width,
    int height,
    const int* vc_flat, int vc_count,
    const int* ec_flat, int ec_count,
    const AStarOptions& options
);

}  // namespace fast_graph
