#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <utility>

#include "sipp.h"  // SafeInterval

namespace fast_graph {

constexpr int INTERVAL_MAX = 1000000000;

/// Interval-based reservation table with Safe Interval Table (SIT).
///
/// Following the official RHCR design (Jiaoyang Li et al., AAAI 2021):
/// - CT: location_key -> list of (t_min, t_max) forbidden intervals
/// - SIT: lazily computed safe intervals per location
/// - CAT: timestep -> set of location_keys (soft constraints)
/// - Supports k-robust constraints (expand forbidden intervals by k steps)
class ReservationTable {
public:
    int map_width;
    int k_robust;
    int window;
    int max_time;   // search depth upper bound for SIT construction
    bool hold_endpoints;

    std::unordered_map<int64_t, std::vector<std::pair<int, int>>> ct;
    std::unordered_map<int, std::unordered_set<int64_t>> cat;

    ReservationTable(int w, int k, int win, bool hold, int mt = 0)
        : map_width(w), k_robust(k), window(win), max_time(mt > 0 ? mt : win),
          hold_endpoints(hold) {}

    static int64_t loc_key(int x, int y, int w) {
        return static_cast<int64_t>(y) * w + x;
    }

    static int64_t edge_key(int fx, int fy, int tx, int ty, int w) {
        int64_t to_vk = static_cast<int64_t>(ty) * w + tx;
        int64_t from_vk = static_cast<int64_t>(fy) * w + fx;
        return -(to_vk * 10000 + from_vk);
    }

    void insert_path_constraints(const std::vector<std::pair<int, int>>& path) {
        if (path.empty()) return;
        for (int t = 0; t < static_cast<int>(path.size()); ++t) {
            auto [px, py] = path[t];
            int64_t loc = loc_key(px, py, map_width);
            int t_min = std::max(0, t - k_robust);
            int t_max = t + 1 + k_robust;
            ct[loc].push_back({t_min, t_max});

            if (t > 0 && path[t - 1] != path[t]) {
                auto [ppx, ppy] = path[t - 1];
                int64_t ekey = edge_key(ppx, ppy, px, py, map_width);
                ct[ekey].push_back({t - 1 - k_robust, t + 1 + k_robust});
                int64_t rekey = edge_key(px, py, ppx, ppy, map_width);
                ct[rekey].push_back({t - 1 - k_robust, t + 1 + k_robust});
            }
        }
        if (hold_endpoints) {
            auto [lx, ly] = path.back();
            int64_t last_loc = loc_key(lx, ly, map_width);
            ct[last_loc].push_back({static_cast<int>(path.size()), INTERVAL_MAX});
        }
    }

    void insert_path_to_cat(const std::vector<std::pair<int, int>>& path) {
        if (path.empty()) return;
        for (int t = 0; t < static_cast<int>(path.size()); ++t) {
            if (t > window + k_robust) break;
            auto [px, py] = path[t];
            int64_t loc = loc_key(px, py, map_width);
            int dt_lo = std::max(-k_robust, -t);
            int dt_hi = std::min(k_robust, window + k_robust - t);
            for (int dt = dt_lo; dt <= dt_hi; ++dt) {
                int ct_t = t + dt;
                if (ct_t >= 0 && ct_t <= window + k_robust)
                    cat[ct_t].insert(loc);
            }
            if (t > 0 && path[t - 1] != path[t]) {
                auto [ppx, ppy] = path[t - 1];
                int64_t ekey = edge_key(ppx, ppy, px, py, map_width);
                for (int dt = dt_lo; dt <= dt_hi; ++dt) {
                    int ct_t = t + dt;
                    if (ct_t >= 0 && ct_t <= window + k_robust)
                        cat[ct_t].insert(ekey);
                }
            }
        }
    }

    // ── SIT query ────────────────────────────────────────────────────────

    std::vector<SafeInterval> get_safe_intervals(int64_t loc, int t_min, int t_max) {
        auto built = build_sit(loc);
        std::vector<SafeInterval> result;
        for (auto& si : built) {
            if (si.t_min >= t_max) break;
            if (si.t_max <= t_min) continue;
            result.push_back({std::max(si.t_min, t_min),
                              std::min(si.t_max, t_max),
                              si.conflicts});
        }
        return result;
    }

    SafeInterval get_first_safe_interval(int64_t loc) {
        auto built = build_sit(loc);
        return built.empty() ? SafeInterval{0, INTERVAL_MAX, 0} : built[0];
    }

    std::vector<SafeInterval> get_safe_intervals_for_edge(
        int64_t to_loc, int64_t ekey, int t_min, int t_max)
    {
        auto vi = get_safe_intervals(to_loc, t_min, t_max);
        auto ei = get_safe_intervals(ekey, t_min, t_max);

        std::vector<SafeInterval> result;
        size_t i = 0, j = 0;
        while (i < vi.size() && j < ei.size()) {
            int lo = std::max(vi[i].t_min, ei[j].t_min);
            int hi = std::min(vi[i].t_max, ei[j].t_max);
            if (lo < hi)
                result.push_back({lo, hi, vi[i].conflicts + ei[j].conflicts});
            if (vi[i].t_max <= ei[j].t_max) ++i;
            else ++j;
        }
        return result;
    }

private:
    std::vector<SafeInterval> build_sit(int64_t loc) {
        auto it = ct.find(loc);
        if (it == ct.end() || it->second.empty())
            return {{0, INTERVAL_MAX, 0}};

        auto sorted_cons = it->second;
        std::sort(sorted_cons.begin(), sorted_cons.end());

        int sit_boundary = max_time + 1;

        std::vector<SafeInterval> intervals;
        int cur = 0;
        for (auto [c_min, c_max] : sorted_cons) {
            c_max = std::min(c_max, sit_boundary);
            if (c_min > cur) {
                int conf = count_cat_conflicts(loc, cur, c_min);
                intervals.push_back({cur, c_min, conf});
            }
            cur = std::max(cur, c_max);
        }
        if (cur < INTERVAL_MAX) {
            int conf = count_cat_conflicts(loc, cur, INTERVAL_MAX);
            intervals.push_back({cur, INTERVAL_MAX, conf});
        }
        if (intervals.empty())
            intervals.push_back({0, INTERVAL_MAX, 0});
        return intervals;
    }

    int count_cat_conflicts(int64_t loc, int t_min, int t_max) {
        int count = 0;
        int limit = std::min(t_max, window + k_robust + 1);
        for (int t = t_min; t < limit; ++t) {
            auto it2 = cat.find(t);
            if (it2 != cat.end() && it2->second.count(loc))
                ++count;
        }
        return count;
    }
};

}  // namespace fast_graph
