#include "sipp.h"
#include "reservation_table.h"

#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <climits>
#include <tuple>

namespace fast_graph {

namespace {

// INTERVAL_MAX is now defined in reservation_table.h

// ── Heuristic ────────────────────────────────────────────────────────────────

inline int goal_heuristic(int x, int y, int goal_id,
                          const std::vector<SIPPGoal>& goals)
{
    if (goal_id >= static_cast<int>(goals.size())) return 0;
    int h = std::abs(x - goals[goal_id].x) + std::abs(y - goals[goal_id].y);
    for (int i = goal_id + 1; i < static_cast<int>(goals.size()); ++i)
        h += std::abs(goals[i - 1].x - goals[i].x) +
             std::abs(goals[i - 1].y - goals[i].y);
    return h;
}

// ── Search node ──────────────────────────────────────────────────────────────

struct Node {
    int x, y, timestep;
    int si_min, si_max, si_conflicts;
    double g_val;
    int h_val, goal_id, conflicts;
    int parent;  // index in nodes vector, -1 = root
};

// ── Closed set ───────────────────────────────────────────────────────────────

struct CSKey {
    int x, y, si_min, goal_id;
    bool operator==(const CSKey& o) const noexcept {
        return x == o.x && y == o.y && si_min == o.si_min && goal_id == o.goal_id;
    }
};
struct CSHash {
    size_t operator()(const CSKey& k) const noexcept {
        return (static_cast<size_t>(k.x) << 20) ^
               (static_cast<size_t>(k.y) << 10) ^
               static_cast<size_t>(k.si_min) ^
               (static_cast<size_t>(k.goal_id) << 28);
    }
};

// ── Path reconstruction with wait-step filling ───────────────────────────────

std::vector<std::pair<int, int>> reconstruct(
    const std::vector<Node>& nodes, int idx,
    const int8_t* passable_grid,
    const int8_t* shelf_grid,
    int width, int height, int map_width)
{
    std::vector<int> chain;
    while (idx >= 0) { chain.push_back(idx); idx = nodes[idx].parent; }
    std::reverse(chain.begin(), chain.end());

    constexpr int DX[4] = {0, 0, -1, 1};
    constexpr int DY[4] = {-1, 1, 0, 0};

    std::vector<std::pair<int, int>> path;
    for (size_t i = 0; i < chain.size(); ++i) {
        auto& nd = nodes[chain[i]];
        if (i > 0) {
            auto& prev = nodes[chain[i - 1]];
            int gap = nd.timestep - prev.timestep - 1;
            if (gap > 0 && prev.x == nd.x && prev.y == nd.y
                && nd.timestep == nd.si_min
                && prev.timestep < prev.si_max
                && nd.si_min >= prev.si_max) {
                // Cross-interval gap: pick a passable neighbour
                // Prefer non-shelf neighbour; fallback to any passable neighbour
                int alt_x = prev.x, alt_y = prev.y;
                bool found = false;
                // First pass: passable non-shelf neighbour
                for (int d = 0; d < 4; ++d) {
                    int nx = prev.x + DX[d], ny = prev.y + DY[d];
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                    if (!passable_grid[ny * map_width + nx]) continue;
                    if (shelf_grid != nullptr
                        && shelf_grid[prev.y * map_width + prev.x]
                        && shelf_grid[ny * map_width + nx]) continue;
                    alt_x = nx; alt_y = ny;
                    found = true;
                    break;
                }
                // Second pass: any passable neighbour (even shelf)
                if (!found) {
                    for (int d = 0; d < 4; ++d) {
                        int nx = prev.x + DX[d], ny = prev.y + DY[d];
                        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                        if (!passable_grid[ny * map_width + nx]) continue;
                        alt_x = nx; alt_y = ny;
                        found = true;
                        break;
                    }
                }
                for (int g = 0; g < gap; ++g)
                    path.emplace_back(alt_x, alt_y);
            } else {
                for (int g = 0; g < gap; ++g)
                    path.emplace_back(prev.x, prev.y);
            }
        } else if (nd.timestep > 0) {
            for (int g = 0; g < nd.timestep; ++g)
                path.emplace_back(nd.x, nd.y);
        }
        path.emplace_back(nd.x, nd.y);
    }
    return path;
}

// ── Internal search using a pre-built ReservationTable ────────────────────

static SIPPResult sipp_search_impl(
    int start_x, int start_y,
    const std::vector<SIPPGoal>& goals,
    const int8_t* passable_grid,
    const int8_t* shelf_grid,
    int width, int height, int map_width,
    int max_time, double shelf_penalty,
    ReservationTable& rt)
{
    SIPPResult result;
    if (goals.empty()) return result;

    // ── Precompute endpoint holding time ─────────────────────────────────
    int earliest_holding = 0;
    if (rt.hold_endpoints && !goals.empty()) {
        auto& last = goals.back();
        int64_t last_loc = ReservationTable::loc_key(last.x, last.y, map_width);
        earliest_holding = rt.get_first_safe_interval(last_loc).t_min;
    }

    // ── Start state ──────────────────────────────────────────────────────
    int64_t start_loc = ReservationTable::loc_key(start_x, start_y, map_width);
    auto first_si = rt.get_first_safe_interval(start_loc);
    if (first_si.t_min > max_time) return result;

    int h0 = goal_heuristic(start_x, start_y, 0, goals);
    if (h0 > max_time + 1) return result;

    // ── Open list: (f, conflicts, counter, node_index) ───────────────────
    using OEntry = std::tuple<double, int, int, int>;
    auto cmp = [](const OEntry& a, const OEntry& b) {
        if (std::get<0>(a) != std::get<0>(b))
            return std::get<0>(a) > std::get<0>(b);
        if (std::get<1>(a) != std::get<1>(b))
            return std::get<1>(a) > std::get<1>(b);
        return std::get<2>(a) > std::get<2>(b);
    };
    std::priority_queue<OEntry, std::vector<OEntry>, decltype(cmp)> open(cmp);

    std::vector<Node> nodes;
    std::unordered_set<CSKey, CSHash> closed;
    int counter = 0;

    int start_t = first_si.t_min;
    double start_g = static_cast<double>(start_t) * 0.0;
    nodes.push_back({start_x, start_y, start_t,
                     first_si.t_min, first_si.t_max, first_si.conflicts,
                     start_g, h0, 0, first_si.conflicts, -1});
    open.push({static_cast<double>(h0) + start_g, first_si.conflicts, counter++, 0});

    constexpr int DX[4] = {0, 0, -1, 1};
    constexpr int DY[4] = {-1, 1, 0, 0};

    while (!open.empty()) {
        auto [f, conf, cnt, nidx] = open.top();
        open.pop();

        const Node cur = nodes[nidx];
        CSKey skey{cur.x, cur.y, cur.si_min, cur.goal_id};
        if (closed.count(skey)) continue;
        closed.insert(skey);

        int gid = cur.goal_id;
        if (gid < static_cast<int>(goals.size())) {
            auto& g = goals[gid];
            if (cur.x == g.x && cur.y == g.y && cur.timestep >= g.release_time) {
                ++gid;
                if (gid == static_cast<int>(goals.size()) &&
                    earliest_holding > cur.timestep)
                    --gid;
            }
        }
        if (gid == static_cast<int>(goals.size())) {
            result.path = reconstruct(nodes, nidx, passable_grid, shelf_grid, width, height, map_width);
            result.nodes_expanded = counter;
            return result;
        }

        int min_t = cur.timestep + 1;
        int max_t = cur.si_max + 1;

        for (int d = 0; d < 4; ++d) {
            int nx = cur.x + DX[d];
            int ny = cur.y + DY[d];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            if (!passable_grid[ny * width + nx]) continue;
            if (shelf_grid != nullptr &&
                shelf_grid[cur.y * width + cur.x] &&
                shelf_grid[ny * width + nx])
                continue;

            int h = goal_heuristic(nx, ny, gid, goals);
            if (h > max_time + 1) continue;

            int64_t to_loc = ReservationTable::loc_key(nx, ny, map_width);
            int64_t ekey = ReservationTable::edge_key(
                cur.x, cur.y, nx, ny, map_width);

            auto intervals = rt.get_safe_intervals_for_edge(
                to_loc, ekey, min_t, max_t);

            for (auto& si : intervals) {
                int arrive = std::max(si.t_min, min_t);
                int wait = arrive - cur.timestep - 1;
                double g = cur.g_val + wait * shelf_penalty + 1.0;
                int nc = cur.conflicts + si.conflicts;

                CSKey nk{nx, ny, si.t_min, gid};
                if (closed.count(nk)) continue;

                int ni = static_cast<int>(nodes.size());
                nodes.push_back({nx, ny, arrive,
                                 si.t_min, si.t_max, si.conflicts,
                                 g, h, gid, nc, nidx});
                open.push({g + h, nc, counter++, ni});
            }
        }

        if (cur.si_max < max_time) {
            int64_t cur_loc = ReservationTable::loc_key(cur.x, cur.y, map_width);
            auto next_si = rt.get_safe_intervals(cur_loc, cur.si_max, max_time + 1);

            if (!next_si.empty()) {
                auto& si = next_si[0];
                int h = goal_heuristic(cur.x, cur.y, gid, goals);
                int wait = si.t_min - cur.timestep - 1;
                double g = cur.g_val + wait * shelf_penalty;
                int nc = cur.conflicts + si.conflicts;

                CSKey nk{cur.x, cur.y, si.t_min, gid};
                if (!closed.count(nk)) {
                    int ni = static_cast<int>(nodes.size());
                    nodes.push_back({cur.x, cur.y, si.t_min,
                                     si.t_min, si.t_max, si.conflicts,
                                     g, h, gid, nc, nidx});
                    open.push({g + h, nc, counter++, ni});
                }
            }
        }

        if (cur.timestep + 1 < cur.si_max && cur.timestep + 1 <= max_time) {
            int next_t = cur.timestep + 1;
            CSKey nk{cur.x, cur.y, cur.si_min, gid};
            if (!closed.count(nk)) {
                int h = goal_heuristic(cur.x, cur.y, gid, goals);
                double g = cur.g_val + shelf_penalty;
                int nc = cur.conflicts + cur.si_conflicts;
                int ni = static_cast<int>(nodes.size());
                nodes.push_back({cur.x, cur.y, next_t,
                                 cur.si_min, cur.si_max, cur.si_conflicts,
                                 g, h, gid, nc, nidx});
                open.push({g + h, nc, counter++, ni});
            }
        }
    }

    return result;  // no path found
}

}  // anonymous namespace

// ══════════════════════════════════════════════════════════════════════════════
// Public API
// ══════════════════════════════════════════════════════════════════════════════

SIPPResult cxx_sipp_search(
    int start_x, int start_y,
    const std::vector<SIPPGoal>& goals,
    const int8_t* passable_grid,
    const int8_t* shelf_grid,
    int width, int height, int map_width,
    int k_robust, int window,
    int max_time, double shelf_penalty,
    bool hold_endpoints,
    const std::vector<std::tuple<int64_t, int, int>>& ct_data,
    const std::vector<std::vector<std::pair<int, int>>>& existing_paths,
    const std::vector<std::pair<int64_t, int>>& cat_data)
{
    SIPPResult result;
    if (goals.empty()) return result;

    // ── Build reservation table ───────────────────────────────────────────
    ReservationTable rt(map_width, k_robust, window, hold_endpoints, max_time);

    for (auto& [loc, t_min, t_max] : ct_data)
        rt.ct[loc].push_back({t_min, t_max});

    for (auto& path : existing_paths)
        rt.insert_path_constraints(path);

    for (auto& [loc_key, t] : cat_data) {
        if (t >= 0 && t <= window + k_robust)
            rt.cat[t].insert(loc_key);
    }

    return sipp_search_impl(start_x, start_y, goals,
                            passable_grid, shelf_grid,
                            width, height, map_width,
                            max_time, shelf_penalty, rt);
}

SIPPResult cxx_sipp_search_with_rt(
    int start_x, int start_y,
    const std::vector<SIPPGoal>& goals,
    const int8_t* passable_grid,
    const int8_t* shelf_grid,
    int width, int height, int map_width,
    int max_time, double shelf_penalty,
    ReservationTable& rt)
{
    return sipp_search_impl(start_x, start_y, goals,
                            passable_grid, shelf_grid,
                            width, height, map_width,
                            max_time, shelf_penalty, rt);
}

}  // namespace fast_graph
