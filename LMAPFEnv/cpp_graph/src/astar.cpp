#include "astar.h"

#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <cstring>

namespace fast_graph {

namespace {

// ── Helpers ──────────────────────────────────────────────────────────────────

inline int manhattan(int x1, int y1, int x2, int y2) {
    return std::abs(x1 - x2) + std::abs(y1 - y2);
}

inline bool is_passable(const int8_t* grid, int x, int y, int w, int h) {
    if (x < 0 || x >= w || y < 0 || y >= h) return false;
    return grid[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] != 0;
}

inline bool is_shelf(const int8_t* grid, int x, int y, int w, int h) {
    if (x < 0 || x >= w || y < 0 || y >= h) return false;
    return grid[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] != 0;
}

// 5 actions: UP, DOWN, LEFT, RIGHT, STAY
constexpr int DX[5] = {0, 0, -1, 1, 0};
constexpr int DY[5] = {-1, 1, 0, 0, 0};

// ── Chain heuristic: estimate cost from (x,y) through remaining goals ───────

inline int chain_heuristic(int x, int y, int goal_id,
                           const std::vector<std::pair<int, int>>& goals)
{
    if (goal_id >= static_cast<int>(goals.size())) return 0;
    int h = manhattan(x, y, goals[goal_id].first, goals[goal_id].second);
    for (int i = goal_id + 1; i < static_cast<int>(goals.size()); ++i)
        h += manhattan(goals[i - 1].first, goals[i - 1].second,
                       goals[i].first, goals[i].second);
    return h;
}

// ── Search entry for the priority queue ──────────────────────────────────────

struct SearchEntry {
    double f;
    int8_t shelf_bias;
    int tie_breaker;  // -t (depth mode) or h (heuristic mode)
    int counter;
    int x, y, t, goal_id;
};

struct CompareEntry {
    bool operator()(const SearchEntry& a, const SearchEntry& b) const noexcept {
        if (a.f != b.f) return a.f > b.f;
        if (a.shelf_bias != b.shelf_bias) return a.shelf_bias > b.shelf_bias;
        if (a.tie_breaker != b.tie_breaker) return a.tie_breaker > b.tie_breaker;
        return a.counter > b.counter;
    }
};

using OpenList = std::priority_queue<SearchEntry, std::vector<SearchEntry>, CompareEntry>;

// ── Flat state-index for O(1) lookup ────────────────────────────────────────
// Layout: state_idx = ((goal_id * H + y) * W + x) * T + t
// where T = max_time + 1, W = width, H = height

struct FlatState {
    int W, H, T, num_goals;
    size_t total;  // total number of states

    FlatState() : W(0), H(0), T(0), num_goals(0), total(0) {}
    FlatState(int w, int h, int max_time, int ng)
        : W(w), H(h), T(max_time + 1), num_goals(ng),
          total(static_cast<size_t>(ng) * static_cast<size_t>(h) *
                static_cast<size_t>(w) * static_cast<size_t>(T)) {}

    inline int idx(int x, int y, int t, int gid) const noexcept {
        return ((gid * H + y) * W + x) * T + t;
    }

    inline void decode(int idx, int& x, int& y, int& t, int& gid) const noexcept {
        t = idx % T;
        int rem = idx / T;
        x = rem % W;
        rem /= W;
        y = rem % H;
        gid = rem / H;
    }

    inline bool valid(int t) const noexcept {
        return t >= 0 && t < T;
    }
};

// ── Path reconstruction with flat arrays ────────────────────────────────────

std::vector<std::pair<int, int>> reconstruct_path_flat(
    const std::vector<int>& parent_idx,
    const FlatState& fs,
    int goal_idx,
    int sx, int sy)
{
    std::vector<std::pair<int, int>> path;
    int cur = goal_idx;
    int last_t = -1;

    // Collect states in reverse
    std::vector<int> state_chain;
    while (cur >= 0) {
        state_chain.push_back(cur);
        if (cur == parent_idx[cur]) break;  // root (self-referencing)
        cur = parent_idx[cur];
    }

    // Reverse and extract (x, y), skipping zero-duration goal_id transitions
    for (int i = static_cast<int>(state_chain.size()) - 1; i >= 0; --i) {
        int x, y, t, gid;
        fs.decode(state_chain[i], x, y, t, gid);
        if (!path.empty() && t == last_t)
            continue;
        path.emplace_back(x, y);
        last_t = t;
    }
    return path;
}

// ── Fallback: hash-based path reconstruction (for large state spaces) ────────

struct GCostKey {
    int x, y, t, goal_id;
    bool operator==(const GCostKey& o) const noexcept {
        return x == o.x && y == o.y && t == o.t && goal_id == o.goal_id;
    }
};

struct GCostKeyHash {
    size_t operator()(const GCostKey& k) const noexcept {
        // Better mixing hash to reduce collisions
        size_t h = static_cast<size_t>(k.x);
        h ^= static_cast<size_t>(k.y) * 2654435761ULL;
        h ^= static_cast<size_t>(k.t) * 40503ULL;
        h ^= static_cast<size_t>(k.goal_id) * 1234567891ULL;
        return h;
    }
};

std::vector<std::pair<int, int>> reconstruct_path_hash(
    const std::unordered_map<GCostKey, GCostKey, GCostKeyHash>& parent,
    int gx, int gy, int gt, int g_goal_id,
    int sx, int sy)
{
    std::vector<GCostKey> states;
    GCostKey cur{gx, gy, gt, g_goal_id};
    while (true) {
        states.push_back(cur);
        if (cur.x == sx && cur.y == sy && cur.t == 0) break;
        auto it = parent.find(cur);
        if (it == parent.end()) break;
        cur = it->second;
    }
    std::reverse(states.begin(), states.end());

    std::vector<std::pair<int, int>> path;
    int last_t = -1;
    for (const auto& s : states) {
        if (!path.empty() && s.t == last_t)
            continue;
        path.emplace_back(s.x, s.y);
        last_t = s.t;
    }
    return path;
}

}  // anonymous namespace

// ── Shared core: works with raw pointers + pre-built constraint sets ─────────

namespace {

AStarResult astar_core(
    std::pair<int, int> start,
    const std::vector<std::pair<int, int>>& goals,
    const int8_t* passable_grid,
    const int8_t* shelf_grid,
    int width, int height,
    const VertexConstraintSet& vertex_constraints,
    const EdgeConstraintSet& edge_constraints,
    const AStarOptions& options)
{
    AStarResult result;

    if (goals.empty()) return result;

    int sx = start.first;
    int sy = start.second;
    int num_goals = static_cast<int>(goals.size());

    // Trivial case: single goal at start
    if (num_goals == 1 && sx == goals[0].first && sy == goals[0].second) {
        result.path.emplace_back(sx, sy);
        return result;
    }

    // Start must be passable
    if (!is_passable(passable_grid, sx, sy, width, height)) {
        return result;
    }

    // Start must not be vertex-constrained at time 0
    SpaceTimeState start_state{{sx, sy}, 0};
    if (vertex_constraints.count(start_state)) {
        return result;
    }

    // Find the last time the FINAL goal is blocked by a vertex constraint
    auto& final_goal = goals.back();
    int goal_blocked_max_time = -1;
    for (const auto& vc : vertex_constraints) {
        if (vc.pos.x == final_goal.first && vc.pos.y == final_goal.second
            && vc.t > goal_blocked_max_time) {
            goal_blocked_max_time = vc.t;
        }
    }

    int max_time = options.max_time;
    bool use_closed = options.use_closed_set;
    bool horizon_mode = options.horizon_mode;
    bool tie_by_depth = options.tie_breaker_by_depth;
    double shelf_penalty = options.shelf_penalty;

    // ── Decide: flat array (fast) vs hash map (memory-safe) ──────────────
    // Flat array uses ~5 * num_goals * H * W * (max_time+1) bytes.
    // Cap at 32 MB to prevent excessive memory use.
    constexpr size_t FLAT_ARRAY_MAX_BYTES = 32 * 1024 * 1024;
    FlatState fs(width, height, max_time, num_goals);
    bool use_flat = (fs.total * 5 * sizeof(int)) <= FLAT_ARRAY_MAX_BYTES;

    OpenList open_list;
    int counter = 0;

    int h0 = chain_heuristic(sx, sy, 0, goals);
    int8_t start_bias = 0;
    if (shelf_grid && is_shelf(shelf_grid, sx, sy, width, height)) {
        start_bias = 1;
    }
    double f0 = static_cast<double>(h0) + static_cast<double>(start_bias) * shelf_penalty;
    int tb0 = tie_by_depth ? 0 : h0;

    // For horizon mode: best state found at max_time
    int best_h = std::numeric_limits<int>::max();
    int best_x = sx, best_y = sy, best_t = 0, best_gid = 0;

    if (use_flat) {
        // ── FAST PATH: flat arrays for O(1) state lookup ────────────────
        std::vector<int> g_flat(fs.total, -1);        // g-costs (-1 = unvisited)
        std::vector<int> parent_flat(fs.total, -1);    // parent state indices
        std::vector<int8_t> closed_flat;               // closed set (lazy alloc)
        if (use_closed) {
            closed_flat.assign(fs.total, 0);
        }

        int start_idx = fs.idx(sx, sy, 0, 0);
        g_flat[start_idx] = 0;
        parent_flat[start_idx] = start_idx;  // self-referencing root
        open_list.push({f0, start_bias, tb0, counter++, sx, sy, 0, 0});

        while (!open_list.empty()) {
            SearchEntry entry = open_list.top();
            open_list.pop();

            int cx = entry.x;
            int cy = entry.y;
            int ct = entry.t;
            int c_gid = entry.goal_id;

            int cur_idx = fs.idx(cx, cy, ct, c_gid);

            // Skip stale entries
            if (ct > g_flat[cur_idx]) continue;

            // Closed set check
            if (use_closed) {
                if (closed_flat[cur_idx]) continue;
                closed_flat[cur_idx] = 1;
            }

            // ── Goal advancement ────────────────────────────────────────
            int gid = c_gid;
            if (gid < num_goals) {
                auto& g = goals[gid];
                if (cx == g.first && cy == g.second) {
                    if (gid == num_goals - 1) {
                        if (ct > goal_blocked_max_time) {
                            result.path = reconstruct_path_flat(parent_flat, fs, cur_idx, sx, sy);
                            result.nodes_expanded = counter;
                            return result;
                        }
                    } else {
                        ++gid;
                    }
                }
            }

            // If goal_id advanced, store the new state
            if (gid != c_gid) {
                int new_idx = fs.idx(cx, cy, ct, gid);
                if (g_flat[new_idx] < 0 || ct < g_flat[new_idx]) {
                    g_flat[new_idx] = ct;
                    parent_flat[new_idx] = cur_idx;
                    int h = chain_heuristic(cx, cy, gid, goals);
                    int8_t sb = 0;
                    if (shelf_grid && is_shelf(shelf_grid, cx, cy, width, height))
                        sb = 1;
                    double f = static_cast<double>(ct) + static_cast<double>(h) +
                               static_cast<double>(sb) * shelf_penalty;
                    int tb = tie_by_depth ? -ct : h;
                    open_list.push({f, sb, tb, counter++, cx, cy, ct, gid});
                }
                continue;
            }

            // Horizon check
            if (ct >= max_time) {
                if (horizon_mode) {
                    int h_val = chain_heuristic(cx, cy, gid, goals);
                    if (h_val < best_h) {
                        best_h = h_val;
                        best_x = cx; best_y = cy; best_t = ct; best_gid = gid;
                    }
                }
                continue;
            }

            // Expand neighbors
            int gx_cur = goals[gid].first;
            int gy_cur = goals[gid].second;

            for (int i = 0; i < 5; ++i) {
                int nx = cx + DX[i];
                int ny = cy + DY[i];
                int nt = ct + 1;

                // Vertex constraint check
                SpaceTimeState vc_key{{nx, ny}, nt};
                if (vertex_constraints.count(vc_key)) continue;

                // Edge constraint check
                EdgeConstraint ec_key{{cx, cy}, {nx, ny}, ct};
                if (edge_constraints.count(ec_key)) continue;

                // Bounds and passability check
                if (!is_passable(passable_grid, nx, ny, width, height)) continue;

                // Shelf-to-shelf check
                if ((nx != cx || ny != cy) &&
                    shelf_grid &&
                    is_shelf(shelf_grid, cx, cy, width, height) &&
                    is_shelf(shelf_grid, nx, ny, width, height))
                {
                    continue;
                }

                // G-cost check via flat array (O(1))
                int next_idx = fs.idx(nx, ny, nt, gid);
                if (g_flat[next_idx] >= 0 && nt >= g_flat[next_idx]) continue;

                // Closed set check
                if (use_closed && closed_flat[next_idx]) continue;

                g_flat[next_idx] = nt;
                parent_flat[next_idx] = cur_idx;

                int h_val = chain_heuristic(nx, ny, gid, goals);
                int8_t shelf_bias = 0;
                if (!(nx == gx_cur && ny == gy_cur) && shelf_grid &&
                    is_shelf(shelf_grid, nx, ny, width, height)) {
                    shelf_bias = 1;
                }

                double f = static_cast<double>(nt) + static_cast<double>(h_val) +
                           static_cast<double>(shelf_bias) * shelf_penalty;
                int tb = tie_by_depth ? -nt : h_val;

                open_list.push({f, shelf_bias, tb, counter++, nx, ny, nt, gid});
                result.nodes_expanded++;
            }
        }

        // Horizon mode: return best-effort path
        if (horizon_mode && best_h < std::numeric_limits<int>::max()) {
            int best_idx = fs.idx(best_x, best_y, best_t, best_gid);
            result.path = reconstruct_path_flat(parent_flat, fs, best_idx, sx, sy);
        }

    } else {
        // ── FALLBACK: hash maps for large state spaces ──────────────────
        std::unordered_map<GCostKey, int, GCostKeyHash> g_costs;
        std::unordered_map<GCostKey, GCostKey, GCostKeyHash> parent;
        std::unordered_set<GCostKey, GCostKeyHash> closed_set;

        g_costs.reserve(4096);
        parent.reserve(4096);
        if (use_closed) closed_set.reserve(4096);

        g_costs[{sx, sy, 0, 0}] = 0;
        open_list.push({f0, start_bias, tb0, counter++, sx, sy, 0, 0});

        while (!open_list.empty()) {
            SearchEntry entry = open_list.top();
            open_list.pop();

            int cx = entry.x;
            int cy = entry.y;
            int ct = entry.t;
            int c_gid = entry.goal_id;

            GCostKey cur_key{cx, cy, ct, c_gid};
            auto git = g_costs.find(cur_key);
            if (git == g_costs.end() || ct > git->second) continue;

            if (use_closed) {
                if (closed_set.count(cur_key)) continue;
                closed_set.insert(cur_key);
            }

            int gid = c_gid;
            if (gid < num_goals) {
                auto& g = goals[gid];
                if (cx == g.first && cy == g.second) {
                    if (gid == num_goals - 1) {
                        if (ct > goal_blocked_max_time) {
                            result.path = reconstruct_path_hash(parent, cx, cy, ct, gid, sx, sy);
                            result.nodes_expanded = counter;
                            return result;
                        }
                    } else {
                        ++gid;
                    }
                }
            }

            if (gid != c_gid) {
                GCostKey new_key{cx, cy, ct, gid};
                auto ng = g_costs.find(new_key);
                if (ng == g_costs.end() || ct < ng->second) {
                    g_costs[new_key] = ct;
                    parent[new_key] = cur_key;
                    int h = chain_heuristic(cx, cy, gid, goals);
                    int8_t sb = 0;
                    if (shelf_grid && is_shelf(shelf_grid, cx, cy, width, height))
                        sb = 1;
                    double f = static_cast<double>(ct) + static_cast<double>(h) +
                               static_cast<double>(sb) * shelf_penalty;
                    int tb = tie_by_depth ? -ct : h;
                    open_list.push({f, sb, tb, counter++, cx, cy, ct, gid});
                }
                continue;
            }

            if (ct >= max_time) {
                if (horizon_mode) {
                    int h_val = chain_heuristic(cx, cy, gid, goals);
                    if (h_val < best_h) {
                        best_h = h_val;
                        best_x = cx; best_y = cy; best_t = ct; best_gid = gid;
                    }
                }
                continue;
            }

            int gx_cur = goals[gid].first;
            int gy_cur = goals[gid].second;

            for (int i = 0; i < 5; ++i) {
                int nx = cx + DX[i];
                int ny = cy + DY[i];
                int nt = ct + 1;

                SpaceTimeState vc_key{{nx, ny}, nt};
                if (vertex_constraints.count(vc_key)) continue;

                EdgeConstraint ec_key{{cx, cy}, {nx, ny}, ct};
                if (edge_constraints.count(ec_key)) continue;

                if (!is_passable(passable_grid, nx, ny, width, height)) continue;

                if ((nx != cx || ny != cy) &&
                    shelf_grid &&
                    is_shelf(shelf_grid, cx, cy, width, height) &&
                    is_shelf(shelf_grid, nx, ny, width, height))
                {
                    continue;
                }

                GCostKey next_key{nx, ny, nt, gid};
                auto g_it = g_costs.find(next_key);
                if (g_it != g_costs.end() && nt >= g_it->second) continue;

                if (use_closed && closed_set.count(next_key)) continue;

                g_costs[next_key] = nt;
                parent[next_key] = cur_key;

                int h_val = chain_heuristic(nx, ny, gid, goals);
                int8_t shelf_bias = 0;
                if (!(nx == gx_cur && ny == gy_cur) && shelf_grid &&
                    is_shelf(shelf_grid, nx, ny, width, height)) {
                    shelf_bias = 1;
                }

                double f = static_cast<double>(nt) + static_cast<double>(h_val) +
                           static_cast<double>(shelf_bias) * shelf_penalty;
                int tb = tie_by_depth ? -nt : h_val;

                open_list.push({f, shelf_bias, tb, counter++, nx, ny, nt, gid});
                result.nodes_expanded++;
            }
        }

        if (horizon_mode && best_h < std::numeric_limits<int>::max()) {
            result.path = reconstruct_path_hash(parent, best_x, best_y, best_t, best_gid, sx, sy);
        }
    }

    return result;
}

}  // anonymous namespace

// ── Public API: vector-based (backward compatible) ────────────────────────────

AStarResult cxx_astar(
    std::pair<int, int> start,
    const std::vector<std::pair<int, int>>& goals,
    const std::vector<int8_t>& passable_grid,
    const std::vector<int8_t>& shelf_grid,
    int width, int height,
    const VertexConstraintSet& vertex_constraints,
    const EdgeConstraintSet& edge_constraints,
    const AStarOptions& options)
{
    return astar_core(start, goals,
                      passable_grid.data(), shelf_grid.data(),
                      width, height,
                      vertex_constraints, edge_constraints, options);
}

// ── Public API: zero-copy (raw pointers + flat int arrays) ────────────────────

AStarResult cxx_astar_nocopy(
    std::pair<int, int> start,
    const std::vector<std::pair<int, int>>& goals,
    const int8_t* passable_ptr,
    const int8_t* shelf_ptr,
    int width, int height,
    const int* vc_flat, int vc_count,
    const int* ec_flat, int ec_count,
    const AStarOptions& options)
{
    // Build constraint sets from flat int arrays (avoids pybind11 tuple conversion)
    VertexConstraintSet vertex_constraints;
    vertex_constraints.reserve(vc_count);
    for (int i = 0; i < vc_count; ++i) {
        vertex_constraints.insert({{vc_flat[i * 3], vc_flat[i * 3 + 1]}, vc_flat[i * 3 + 2]});
    }
    EdgeConstraintSet edge_constraints;
    edge_constraints.reserve(ec_count);
    for (int i = 0; i < ec_count; ++i) {
        edge_constraints.insert({{ec_flat[i * 5], ec_flat[i * 5 + 1]},
                                 {ec_flat[i * 5 + 2], ec_flat[i * 5 + 3]},
                                 ec_flat[i * 5 + 4]});
    }
    return astar_core(start, goals,
                      passable_ptr, shelf_ptr,
                      width, height,
                      vertex_constraints, edge_constraints, options);
}

}  // namespace fast_graph

