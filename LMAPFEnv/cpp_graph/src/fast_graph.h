#ifndef FAST_GRAPH_H
#define FAST_GRAPH_H

#include <vector>
#include <utility>
#include <cstdint>

/**
 * Persistent directed-graph engine for agent conflict resolution.
 *
 * Design principles:
 *  - Pre-allocate all storage once (max_nodes) — no heap allocations in hot path.
 *  - Visit-token trick for O(1) per-query visited-array reset.
 *  - Subset bitmaps instead of hash-sets for O(1) "is node in component" checks.
 *  - All public methods are thread-unsafe by design (single-threaded use).
 */
class FastGraph {
public:
    explicit FastGraph(int max_nodes);

    // --- lifecycle (called once per env step) -------------------------------
    void reset(int n);

    // --- graph construction -------------------------------------------------
    void add_edge(int u, int v);

    // --- queries ------------------------------------------------------------
    std::vector<std::vector<int>> components();

    // Find a directed cycle within the subgraph induced by `nodes`.
    // Returns edges (u,v) of the first cycle found, or empty if acyclic.
    std::vector<std::pair<int, int>> find_cycle(const std::vector<int>& nodes);

    // Longest path in a DAG (requires subgraph to be acyclic).
    // Returns node IDs on the longest path.
    std::vector<int> dag_longest_path(const std::vector<int>& nodes);

private:
    int max_nodes_;
    int n_;                       // active node count this step

    // adjacency storage — pre-allocated, cleared via size reset
    std::vector<std::vector<int>> adj_;      // forward:  u -> [v, ...]
    std::vector<std::vector<int>> radj_;     // reverse:  u -> [v, ...]  (for undirected BFS)

    // visit-token trick: visited_[i] == visit_token_  ⇔  visited
    std::vector<int> visit_token_;
    int current_token_;

    // per-query scratch buffers (reused to avoid reallocation)
    std::vector<int> stack_;
    std::vector<int> color_;      // 0=white 1=gray 2=black
    std::vector<int> parent_;     // for cycle reconstruction
    std::vector<uint8_t> in_subset_;  // bitmap: 1 = node is in current subgraph

    void bfs_component(int start, std::vector<int>& out);

    // DFS returning true if a cycle is found (populates cycle_edges_scratch).
    bool cycle_dfs(int u,
                   std::vector<int>& cycle_scratch);
};

#endif // FAST_GRAPH_H
