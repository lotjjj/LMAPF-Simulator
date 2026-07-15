#include "batch_plan.h"
#include "astar.h"
#include "sipp.h"
#include "reservation_table.h"

#include <chrono>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <cstring>

namespace fast_graph {

namespace {

// ── Helpers ──────────────────────────────────────────────────────────────────

inline double wall_clock() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}

inline int chain_distance(int sx, int sy,
                          const std::vector<std::pair<int, int>>& goals)
{
    int dist = 0;
    int px = sx, py = sy;
    for (auto [gx, gy] : goals) {
        dist += std::abs(px - gx) + std::abs(py - gy);
        px = gx; py = gy;
    }
    return dist;
}

inline std::pair<int,int> get_pos_at(
    const std::vector<std::pair<int,int>>& path, int t)
{
    if (path.empty()) return {-1, -1};
    if (t < static_cast<int>(path.size())) return path[t];
    return path.back();
}

void force_wait(std::vector<std::pair<int,int>>& path, int t) {
    if (path.empty()) return;
    if (t >= static_cast<int>(path.size())) {
        auto last = path.back();
        path.resize(t + 1, last);
    } else if (t > 0) {
        path[t] = path[t - 1];
    }
}

// ── Conflict detection ──────────────────────────────────────────────────────

struct Conflict {
    int type;  // 0 = vertex, 1 = edge
    int agent_a, agent_b;
    int time;
};

Conflict find_first_conflict(
    const std::map<int, std::vector<std::pair<int,int>>>& solution,
    int conflict_horizon)
{
    if (solution.size() < 2) return {-1, -1, -1, -1};

    // Collect agent ids in sorted order
    std::vector<int> ids;
    ids.reserve(solution.size());
    for (auto& [id, _] : solution) ids.push_back(id);

    int max_len = 0;
    for (auto& [_, p] : solution)
        max_len = std::max(max_len, static_cast<int>(p.size()));
    max_len = std::min(max_len, conflict_horizon + 1);

    for (int t = 0; t < max_len; ++t) {
        // Vertex conflicts
        std::map<std::pair<int,int>, int> occupied;
        for (int aid : ids) {
            auto it = solution.find(aid);
            if (it == solution.end()) continue;
            auto pos = get_pos_at(it->second, t);
            if (pos.first < 0) continue;
            auto oit = occupied.find(pos);
            if (oit != occupied.end())
                return {0, oit->second, aid, t};
            occupied[pos] = aid;
        }

        // Edge conflicts
        if (t + 1 < max_len) {
            std::map<std::tuple<int,int,int,int>, int> moves;
            for (int aid : ids) {
                auto it = solution.find(aid);
                if (it == solution.end()) continue;
                auto pos = get_pos_at(it->second, t);
                auto nxt = get_pos_at(it->second, t + 1);
                if (pos.first < 0 || nxt.first < 0 || pos == nxt) continue;
                auto rev = std::make_tuple(nxt.first, nxt.second, pos.first, pos.second);
                auto oit = moves.find(rev);
                if (oit != moves.end())
                    return {1, oit->second, aid, t};
                moves[std::make_tuple(pos.first, pos.second, nxt.first, nxt.second)] = aid;
            }
        }
    }
    return {-1, -1, -1, -1};
}

// ── A* mode: sequential planning with vertex/edge sets ──────────────────────

std::vector<std::pair<int,int>> astar_search_single(
    int sx, int sy,
    const std::vector<std::pair<int,int>>& goals,
    const int8_t* passable_ptr, const int8_t* shelf_ptr,
    int width, int height,
    const int* vc_flat, int vc_count,
    const int* ec_flat, int ec_count,
    int max_time, bool horizon_mode, double shelf_penalty)
{
    AStarOptions opts;
    opts.max_time = max_time;
    opts.horizon_mode = horizon_mode;
    opts.use_closed_set = false;
    opts.tie_breaker_by_depth = true;
    opts.shelf_penalty = shelf_penalty;

    auto res = cxx_astar_nocopy(
        {sx, sy}, goals,
        passable_ptr, shelf_ptr,
        width, height,
        vc_flat, vc_count,
        ec_flat, ec_count,
        opts);
    return res.path;
}

bool plan_astar_mode(
    const std::vector<BatchAgent>& agents,
    const std::vector<int>& order,
    const std::vector<int>& per_agent_max_time,
    const int8_t* passable_ptr, const int8_t* shelf_ptr,
    int width, int height,
    bool horizon_mode, double shelf_penalty,
    int planning_window, double deadline,
    BatchPlanResult& result)
{
    // Global reservations
    // vc: (x, y, t)  ec: (fx, fy, tx, ty, t)
    std::vector<int> res_v;  // flat: x,y,t per entry
    std::vector<int> res_e;  // flat: fx,fy,tx,ty,t per entry

    for (int idx : order) {
        if (deadline > 0 && wall_clock() >= deadline) return false;

        auto& agent = agents[idx];
        int max_time = per_agent_max_time[idx];
        max_time = std::max(max_time, planning_window);

        // Build constraint arrays: vertex = all reservations
        int vc_count = static_cast<int>(res_v.size() / 3);

        // Build edge constraints: original + reverse
        int orig_ec = static_cast<int>(res_e.size() / 5);
        std::vector<int> ec_arr;
        ec_arr.reserve(res_e.size() * 2);
        for (int i = 0; i < orig_ec; ++i) {
            int fx = res_e[i*5], fy = res_e[i*5+1];
            int tx = res_e[i*5+2], ty = res_e[i*5+3];
            int t  = res_e[i*5+4];
            // Original
            ec_arr.push_back(fx); ec_arr.push_back(fy);
            ec_arr.push_back(tx); ec_arr.push_back(ty);
            ec_arr.push_back(t);
            // Reverse
            ec_arr.push_back(tx); ec_arr.push_back(ty);
            ec_arr.push_back(fx); ec_arr.push_back(fy);
            ec_arr.push_back(t);
        }
        int ec_count = static_cast<int>(ec_arr.size() / 5);

        // Multi-goal search
        auto path = astar_search_single(
            agent.start_x, agent.start_y, agent.goals,
            passable_ptr, shelf_ptr, width, height,
            res_v.data(), vc_count,
            ec_arr.data(), ec_count,
            max_time, horizon_mode, shelf_penalty);

        // Fallback: single goal
        if (path.empty() && agent.goals.size() > 1) {
            std::vector<std::pair<int,int>> single_goal = {agent.goals.front()};
            path = astar_search_single(
                agent.start_x, agent.start_y, single_goal,
                passable_ptr, shelf_ptr, width, height,
                res_v.data(), vc_count,
                ec_arr.data(), ec_count,
                max_time, horizon_mode, shelf_penalty);
        }

        if (path.empty()) return false;

        // Pad to planning_window + 1
        int pw = planning_window;
        while (static_cast<int>(path.size()) < pw + 1)
            path.push_back(path.back());

        // Add to reservations (only within planning_window)
        for (int t = 0; t <= pw; ++t) {
            res_v.push_back(path[t].first);
            res_v.push_back(path[t].second);
            res_v.push_back(t);
        }
        for (int t = 0; t < pw; ++t) {
            auto& p0 = path[t];
            auto& p1 = path[t + 1];
            if (p0 != p1) {
                res_e.push_back(p0.first); res_e.push_back(p0.second);
                res_e.push_back(p1.first); res_e.push_back(p1.second);
                res_e.push_back(t);
            }
        }

        result.paths[idx] = std::move(path);
    }
    return true;
}

// ── SIPP mode: sequential planning with ReservationTable ────────────────────

bool plan_sipp_mode(
    const std::vector<BatchAgent>& agents,
    const std::vector<int>& order,
    const std::vector<int>& per_agent_max_time,
    const int8_t* passable_grid,
    const int8_t* shelf_grid,
    int width, int height,
    double shelf_penalty,
    int planning_window, int k_robust,
    const std::vector<std::vector<std::tuple<int64_t, int, int>>>& initial_ct_flat,
    double deadline,
    BatchPlanResult& result)
{
    int k_robust_ct = 0;  // CT uses k_robust=0 (matching Python behavior)
    ReservationTable rt(
        std::max(width, height), k_robust, planning_window,
        false /* hold_endpoints */, 500 /* max_time for SIT */);

    for (int idx : order) {
        if (deadline > 0 && wall_clock() >= deadline) return false;

        auto& agent = agents[idx];
        int max_time = per_agent_max_time[idx];

        // Add initial constraints
        std::vector<std::tuple<int64_t, std::pair<int,int>>> ic_keys;
        if (idx < static_cast<int>(initial_ct_flat.size())) {
            for (auto& [loc, t_min, t_max] : initial_ct_flat[idx]) {
                int adj_min = std::max(0, t_min - k_robust_ct);
                int adj_max = t_max + k_robust_ct;
                rt.ct[loc].push_back({adj_min, adj_max});
                ic_keys.push_back({loc, {adj_min, adj_max}});
            }
        }

        // Build goals
        std::vector<SIPPGoal> goals;
        goals.reserve(agent.goals.size());
        for (auto [gx, gy] : agent.goals)
            goals.push_back({gx, gy, 0});

        // Multi-goal SIPP search
        auto sipp_res = cxx_sipp_search_with_rt(
            agent.start_x, agent.start_y, goals,
            passable_grid, shelf_grid,
            width, height, width,
            max_time, shelf_penalty, rt);

        auto path = sipp_res.path;
        result.nodes_expanded += sipp_res.nodes_expanded;

        // Fallback: single goal
        if (path.empty() && goals.size() > 1) {
            std::vector<SIPPGoal> single = {goals.front()};
            sipp_res = cxx_sipp_search_with_rt(
                agent.start_x, agent.start_y, single,
                passable_grid, shelf_grid,
                width, height, width,
                max_time, shelf_penalty, rt);
            path = sipp_res.path;
            result.nodes_expanded += sipp_res.nodes_expanded;
        }

        if (path.empty()) return false;

        // Pad
        int pw = planning_window;
        while (static_cast<int>(path.size()) < pw + 1)
            path.push_back(path.back());

        result.paths[idx] = path;

        // Insert constraints (only within planning_window)
        std::vector<std::pair<int,int>> res_path(path.begin(), path.begin() + pw + 1);
        rt.insert_path_constraints(res_path);
        rt.insert_path_to_cat(res_path);

        // Remove initial constraints
        for (auto& [loc, interval] : ic_keys) {
            auto it = rt.ct.find(loc);
            if (it != rt.ct.end()) {
                auto& vec = it->second;
                vec.erase(std::remove(vec.begin(), vec.end(), interval), vec.end());
            }
        }
    }
    return true;
}

// ── Conflict repair ─────────────────────────────────────────────────────────

bool repair_conflicts(
    std::map<int, std::vector<std::pair<int,int>>>& solution,
    int planning_window,
    int max_rounds = 10)
{
    for (int round = 0; round < max_rounds; ++round) {
        auto conflict = find_first_conflict(solution, planning_window);
        if (conflict.type < 0) return true;  // no conflicts

        int a1 = conflict.agent_a, a2 = conflict.agent_b;
        int t = conflict.time;
        bool repaired = false;

        for (int aid : {a2, a1}) {
            auto it = solution.find(aid);
            if (it == solution.end()) continue;
            auto& path = it->second;
            if (t > 0 && t < static_cast<int>(path.size()) && path[t] != path[t-1]) {
                force_wait(path, t);
                repaired = true;
                break;
            }
        }
        if (!repaired) return false;
    }
    // Final check
    return find_first_conflict(solution, planning_window).type < 0;
}

}  // anonymous namespace

// ══════════════════════════════════════════════════════════════════════════════
// Public API
// ══════════════════════════════════════════════════════════════════════════════

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
    double deadline)
{
    BatchPlanResult result;

    bool ok = false;
    if (mode == "astar") {
        ok = plan_astar_mode(
            agents, order, per_agent_max_time,
            passable_grid, shelf_grid, width, height,
            horizon_mode, shelf_penalty, planning_window,
            deadline, result);
    } else if (mode == "sipp") {
        ok = plan_sipp_mode(
            agents, order, per_agent_max_time,
            passable_grid, shelf_grid, width, height,
            shelf_penalty, planning_window, k_robust,
            initial_ct_flat, deadline, result);
    }

    if (!ok) {
        result.success = false;
        result.paths.clear();
        return result;
    }

    // Conflict detection + repair
    if (!repair_conflicts(result.paths, planning_window)) {
        result.success = false;
        result.paths.clear();
        return result;
    }

    result.success = true;
    return result;
}

}  // namespace fast_graph
