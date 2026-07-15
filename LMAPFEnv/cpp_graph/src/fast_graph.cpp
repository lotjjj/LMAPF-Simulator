#include "fast_graph.h"
#include <algorithm>
#include <cstring>
#include <stack>
#include <queue>

FastGraph::FastGraph(int max_nodes)
    : max_nodes_(max_nodes),
      n_(0),
      adj_(max_nodes),
      radj_(max_nodes),
      visit_token_(max_nodes, 0),
      current_token_(1),
      stack_(max_nodes),
      color_(max_nodes, 0),
      parent_(max_nodes, -1),
      in_subset_(max_nodes, 0) {}

void FastGraph::reset(int n) {
    n_ = n;
    // Clear adjacency lists — only touch first n entries
    for (int i = 0; i < n; ++i) {
        adj_[i].clear();
        radj_[i].clear();
    }
    // Bump token to invalidate previous visited state
    ++current_token_;
    if (current_token_ <= 0) {  // overflow guard
        std::fill(visit_token_.begin(), visit_token_.end(), 0);
        current_token_ = 1;
    }
}

void FastGraph::add_edge(int u, int v) {
    adj_[u].push_back(v);
    radj_[v].push_back(u);
}

// ── Weakly Connected Components ─────────────────────────────────────────────

void FastGraph::bfs_component(int start, std::vector<int>& out) {
    std::queue<int> q;
    q.push(start);
    visit_token_[start] = current_token_;
    out.push_back(start);

    while (!q.empty()) {
        int u = q.front(); q.pop();

        // forward neighbours
        for (int v : adj_[u]) {
            if (visit_token_[v] != current_token_) {
                visit_token_[v] = current_token_;
                out.push_back(v);
                q.push(v);
            }
        }
        // backward neighbours (undirected connectivity)
        for (int v : radj_[u]) {
            if (visit_token_[v] != current_token_) {
                visit_token_[v] = current_token_;
                out.push_back(v);
                q.push(v);
            }
        }
    }
}

std::vector<std::vector<int>> FastGraph::components() {
    std::vector<std::vector<int>> result;
    for (int i = 0; i < n_; ++i) {
        if (visit_token_[i] != current_token_) {
            std::vector<int> comp;
            bfs_component(i, comp);
            result.push_back(std::move(comp));
        }
    }
    return result;
}

// ── Directed Cycle Detection ────────────────────────────────────────────────

bool FastGraph::cycle_dfs(int u, std::vector<int>& cycle_scratch) {
    color_[u] = 1;  // gray

    for (int v : adj_[u]) {
        if (!in_subset_[v]) continue;  // v not in current subgraph

        if (color_[v] == 1) {  // back edge: v is an ancestor → cycle found
            // Reconstruct the cycle: u → ... → v
            // We store edges from v back to u, following parent pointers
            int cur = u;
            while (cur != v) {
                int p = parent_[cur];
                if (p < 0) break;
                cycle_scratch.push_back(p);
                cycle_scratch.push_back(cur);
                cur = p;
            }
            // Add the closing edge: u → v
            cycle_scratch.push_back(u);
            cycle_scratch.push_back(v);
            return true;
        }

        if (color_[v] == 0) {  // white — not yet visited
            parent_[v] = u;
            if (cycle_dfs(v, cycle_scratch)) {
                return true;
            }
        }
    }

    color_[u] = 2;  // black
    return false;
}

std::vector<std::pair<int, int>> FastGraph::find_cycle(const std::vector<int>& nodes) {
    // Build subset bitmap
    for (int v : nodes) {
        in_subset_[v] = 1;
    }

    // Reset colors for the subset
    for (int v : nodes) {
        color_[v] = 0;     // white
        parent_[v] = -1;
    }

    std::vector<std::pair<int, int>> result;

    for (int v : nodes) {
        if (color_[v] == 0) {  // white
            std::vector<int> scratch;
            if (cycle_dfs(v, scratch)) {
                // Convert flat scratch to edge pairs
                // scratch = [p1, c1, p2, c2, ..., pn, cn]  where cn is the back edge target
                // The last pair (pn, cn) is the back edge that closes the cycle
                // We need to format as [(p1,c1), (p2,c2), ..., (pn,cn)]
                for (size_t i = 0; i + 1 < scratch.size(); i += 2) {
                    result.emplace_back(scratch[i], scratch[i + 1]);
                }
                break;
            }
        }
    }

    // Clear subset bitmap
    for (int v : nodes) {
        in_subset_[v] = 0;
    }

    return result;
}

// ── DAG Longest Path ────────────────────────────────────────────────────────

std::vector<int> FastGraph::dag_longest_path(const std::vector<int>& nodes) {
    if (nodes.empty()) return {};

    // Build in-degree for topological sort within the subset
    for (int v : nodes) {
        in_subset_[v] = 1;
        color_[v] = 0;  // will be reused as in-degree
    }
    for (int u : nodes) {
        for (int v : adj_[u]) {
            if (in_subset_[v]) {
                color_[v]++;  // in-degree
            }
        }
    }

    // Kahn's algorithm for topological sort
    std::vector<int> topo;
    topo.reserve(nodes.size());
    std::queue<int> q;

    for (int v : nodes) {
        if (color_[v] == 0) {  // in-degree 0
            q.push(v);
        }
    }

    while (!q.empty()) {
        int u = q.front(); q.pop();
        topo.push_back(u);

        for (int v : adj_[u]) {
            if (in_subset_[v]) {
                color_[v]--;
                if (color_[v] == 0) {
                    q.push(v);
                }
            }
        }
    }

    // DP: longest path ending at each node
    // Reuse parent_[] for DP predecessor tracking
    // Reuse stack_[] for DP distance
    for (int v : nodes) {
        stack_[v] = 1;      // distance (at least itself)
        parent_[v] = -1;
    }

    int best_dist = 1;
    int best_node = nodes[0];

    for (int u : topo) {
        for (int v : adj_[u]) {
            if (!in_subset_[v]) continue;
            int cand = stack_[u] + 1;
            if (cand > stack_[v]) {
                stack_[v] = cand;
                parent_[v] = u;
                if (cand > best_dist) {
                    best_dist = cand;
                    best_node = v;
                }
            }
        }
    }

    // Reconstruct path
    std::vector<int> path;
    int cur = best_node;
    while (cur >= 0) {
        path.push_back(cur);
        cur = parent_[cur];
    }
    std::reverse(path.begin(), path.end());

    // Cleanup subset bitmap
    for (int v : nodes) {
        in_subset_[v] = 0;
    }

    return path;
}
