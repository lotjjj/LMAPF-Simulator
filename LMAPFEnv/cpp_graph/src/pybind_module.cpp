#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include "fast_graph.h"
#include "astar.h"
#include "sipp.h"
#include "batch_plan.h"
#include "bfs_distance.h"
#include "step_validate.h"

namespace py = pybind11;
namespace fg = fast_graph;

// ── Module ───────────────────────────────────────────────────────────────────

PYBIND11_MODULE(fast_graph, m) {
    m.doc() = "C++ FastGraph engine for LMAPF-Simulator conflict resolution";

    // ── FastGraph (existing) ───────────────────────────────────────────────
    py::class_<FastGraph>(m, "FastGraph")
        .def(py::init<int>(), py::arg("max_nodes"),
             "Create a persistent graph engine pre-allocated for at most max_nodes.")
        .def("reset", &FastGraph::reset, py::arg("n"),
             "Reset for a new step with n active nodes.")
        .def("add_edge", &FastGraph::add_edge, py::arg("u"), py::arg("v"),
             "Add a directed edge u -> v.")
        .def("components", &FastGraph::components,
             "Return weakly connected components as list of node-ID lists.")
        .def("find_cycle", &FastGraph::find_cycle, py::arg("nodes"),
             "Find a directed cycle in the subgraph induced by `nodes`. "
             "Returns list of (u,v) edge pairs, or empty if acyclic.")
        .def("dag_longest_path", &FastGraph::dag_longest_path, py::arg("nodes"),
             "Return the longest path (list of node IDs) in the DAG subgraph.");

    // ── A* engine (multi-goal) ─────────────────────────────────────────────
    m.def("cxx_astar",
          [](std::pair<int, int> start,
             const std::vector<std::pair<int, int>>& goals,
             py::array_t<int8_t, py::array::c_style | py::array::forcecast> passable_grid,
             py::array_t<int8_t, py::array::c_style | py::array::forcecast> shelf_grid,
             const std::vector<std::tuple<int, int, int>>& vertex_constraints,
             const std::vector<std::tuple<int, int, int, int, int>>& edge_constraints,
             int max_time,
             bool horizon_mode,
             bool use_closed_set,
             bool tie_breaker_by_depth,
             double shelf_penalty) -> std::vector<std::pair<int, int>>
          {
              auto buf = passable_grid.request();
              int width = static_cast<int>(buf.shape[1]);
              int height = static_cast<int>(buf.shape[0]);
              const int8_t* passable_ptr = static_cast<const int8_t*>(buf.ptr);

              auto shelf_buf = shelf_grid.request();
              const int8_t* shelf_ptr = static_cast<const int8_t*>(shelf_buf.ptr);

              std::vector<int> vc_flat;
              vc_flat.reserve(vertex_constraints.size() * 3);
              for (const auto& t : vertex_constraints) {
                  vc_flat.push_back(std::get<0>(t));
                  vc_flat.push_back(std::get<1>(t));
                  vc_flat.push_back(std::get<2>(t));
              }

              std::vector<int> ec_flat;
              ec_flat.reserve(edge_constraints.size() * 5);
              for (const auto& t : edge_constraints) {
                  ec_flat.push_back(std::get<0>(t));
                  ec_flat.push_back(std::get<1>(t));
                  ec_flat.push_back(std::get<2>(t));
                  ec_flat.push_back(std::get<3>(t));
                  ec_flat.push_back(std::get<4>(t));
              }

              fg::AStarOptions opts;
              opts.max_time = max_time;
              opts.horizon_mode = horizon_mode;
              opts.use_closed_set = use_closed_set;
              opts.tie_breaker_by_depth = tie_breaker_by_depth;
              opts.shelf_penalty = shelf_penalty;

              fg::AStarResult res;
              {
                  py::gil_scoped_release release;
                  res = fg::cxx_astar_nocopy(
                      start, goals,
                      passable_ptr, shelf_ptr,
                      width, height,
                      vc_flat.data(), static_cast<int>(vc_flat.size() / 3),
                      ec_flat.data(), static_cast<int>(ec_flat.size() / 5),
                      opts);
              }

              return res.path;
          },
          py::arg("start"), py::arg("goals"),
          py::arg("passable_grid"), py::arg("shelf_grid"),
          py::arg("vertex_constraints"), py::arg("edge_constraints"),
          py::arg("max_time") = 500,
          py::arg("horizon_mode") = false,
          py::arg("use_closed_set") = true,
          py::arg("tie_breaker_by_depth") = true,
          py::arg("shelf_penalty") = 4.0,
          "Unified space-time A* search with multi-goal support.\n\n"
          "goals: list of (x, y) positions; search visits them in order.\n"
          "passable_grid / shelf_grid: 2D numpy int8 arrays.\n"
          "vertex_constraints: list of (x, y, t) tuples.\n"
          "edge_constraints: list of (x1, y1, x2, y2, t) tuples.\n\n"
          "Returns a list of (x, y) positions from start through all goals,\n"
          "or empty list if no path found.");

    // ── A* engine (multi-goal, zero-copy) ──────────────────────────────
    m.def("cxx_astar_nocopy",
          [](std::pair<int, int> start,
             const std::vector<std::pair<int, int>>& goals,
             py::array_t<int8_t, py::array::c_style | py::array::forcecast> passable_grid,
             py::array_t<int8_t, py::array::c_style | py::array::forcecast> shelf_grid,
             py::array_t<int, py::array::c_style | py::array::forcecast> vc_flat,
             py::array_t<int, py::array::c_style | py::array::forcecast> ec_flat,
             int max_time,
             bool horizon_mode,
             bool use_closed_set,
             bool tie_breaker_by_depth,
             double shelf_penalty) -> std::vector<std::pair<int, int>>
          {
              auto p_buf = passable_grid.request();
              int width = static_cast<int>(p_buf.shape[1]);
              int height = static_cast<int>(p_buf.shape[0]);
              const int8_t* passable_ptr = static_cast<const int8_t*>(p_buf.ptr);

              auto s_buf = shelf_grid.request();
              const int8_t* shelf_ptr = static_cast<const int8_t*>(s_buf.ptr);

              auto vc_buf = vc_flat.request();
              int vc_count = static_cast<int>(vc_buf.size / 3);
              const int* vc_ptr = static_cast<const int*>(vc_buf.ptr);

              auto ec_buf = ec_flat.request();
              int ec_count = static_cast<int>(ec_buf.size / 5);
              const int* ec_ptr = static_cast<const int*>(ec_buf.ptr);

              fg::AStarOptions opts;
              opts.max_time = max_time;
              opts.horizon_mode = horizon_mode;
              opts.use_closed_set = use_closed_set;
              opts.tie_breaker_by_depth = tie_breaker_by_depth;
              opts.shelf_penalty = shelf_penalty;

              fg::AStarResult res;
              {
                  py::gil_scoped_release release;
                  res = fg::cxx_astar_nocopy(
                      start, goals,
                      passable_ptr, shelf_ptr,
                      width, height,
                      vc_ptr, vc_count,
                      ec_ptr, ec_count,
                      opts);
              }

              return res.path;
          },
          py::arg("start"), py::arg("goals"),
          py::arg("passable_grid"), py::arg("shelf_grid"),
          py::arg("vc_flat"), py::arg("ec_flat"),
          py::arg("max_time") = 500,
          py::arg("horizon_mode") = false,
          py::arg("use_closed_set") = false,
          py::arg("tie_breaker_by_depth") = true,
          py::arg("shelf_penalty") = 3.0,
          "Zero-copy A* search: accepts raw numpy buffers + flat int arrays.\n\n"
          "vc_flat: 1D int32 array [x0,y0,t0, x1,y1,t1, ...] (length = 3*vc_count).\n"
          "ec_flat: 1D int32 array [fx0,fy0,tx0,ty0,t0, ...] (length = 5*ec_count).\n"
          "Grids are passed as raw pointers (no copy).\n\n"
          "Returns a list of (x, y) positions, or empty list if no path found.");

    // ── SIPP engine ─────────────────────────────────────────────────────────
    m.def("cxx_sipp_search",
          [](int start_x, int start_y,
             const std::vector<std::tuple<int,int,int>>& goals,  // (x, y, release_time)
             py::array_t<int8_t, py::array::c_style | py::array::forcecast> passable_grid,
             py::array_t<int8_t, py::array::c_style | py::array::forcecast> shelf_grid,
             int map_width,
             int k_robust,
             int window,
             int max_time,
             double shelf_penalty,
             bool hold_endpoints,
             // ct_data: list of (loc_key, t_min, t_max)
             const std::vector<std::tuple<int64_t,int,int>>& ct_data,
             // existing_paths: list of list of (x, y)
             const std::vector<std::vector<std::pair<int,int>>>& existing_paths,
             // cat_data: list of (loc_key, timestep) for soft conflict avoidance
             const std::vector<std::pair<int64_t,int>>& cat_data
             ) -> std::vector<std::pair<int,int>>
          {
              auto passable_buf = passable_grid.request();
              int height = static_cast<int>(passable_buf.shape[0]);
              int width  = static_cast<int>(passable_buf.shape[1]);
              const int8_t* passable_ptr = static_cast<const int8_t*>(passable_buf.ptr);

              auto shelf_buf = shelf_grid.request();
              const int8_t* shelf_ptr = static_cast<const int8_t*>(shelf_buf.ptr);

              std::vector<fg::SIPPGoal> cpp_goals;
              cpp_goals.reserve(goals.size());
              for (auto& [gx, gy, rt] : goals)
                  cpp_goals.push_back({gx, gy, rt});

              fg::SIPPResult res;
              {
                  py::gil_scoped_release release;
                  res = fg::cxx_sipp_search(
                      start_x, start_y,
                      cpp_goals,
                      passable_ptr, shelf_ptr,
                      width, height, map_width,
                      k_robust, window,
                      max_time, shelf_penalty,
                      hold_endpoints,
                      ct_data,
                      existing_paths,
                      cat_data);
              }

              return res.path;
          },
          py::arg("start_x"), py::arg("start_y"),
          py::arg("goals"),
          py::arg("passable_grid"), py::arg("shelf_grid"),
          py::arg("map_width"),
          py::arg("k_robust"), py::arg("window"),
          py::arg("max_time"),
          py::arg("shelf_penalty"),
          py::arg("hold_endpoints"),
          py::arg("ct_data"),
          py::arg("existing_paths"),
          py::arg("cat_data") = std::vector<std::pair<int64_t,int>>(),
          "C++ SIPP single-agent search.\n\n"
          "goals: list of (x, y, release_time).\n"
          "ct_data: list of (loc_key, t_min, t_max) pre-built constraints.\n"
          "existing_paths: already-planned higher-priority paths.\n"
          "cat_data: list of (loc_key, timestep) soft conflict avoidance entries.\n"
          "Returns list of (x, y) positions, or empty if no path found.");

    // ── BFS distance grid ──────────────────────────────────────────────
    m.def("bfs_distance_grid",
          &bfs_distance_grid,
          py::arg("width"), py::arg("height"),
          py::arg("passable_mask"), py::arg("shelf_mask"),
          py::arg("target_x"), py::arg("target_y"),
          "Compute BFS distance grid from target position.\n\n"
          "Returns int16 numpy array (height, width) with shortest-path\n"
          "distances to (target_x, target_y), or -1 if unreachable.\n"
          "passable_mask / shelf_mask: 2D numpy arrays (bool or uint8).");

    // ── Step validation (C1: batch action validation) ─────────────────
    m.def("validate_agent_actions",
          &validate_agent_actions,
          py::arg("width"), py::arg("height"),
          py::arg("agv_x"), py::arg("agv_y"), py::arg("actions"),
          py::arg("passable"), py::arg("shelf"),
          "Batch-validate agent actions for simulation step.\n\n"
          "Returns uint8 array (num_agents,): 1 = feasible, 0 = stay forced.");

    // ── Batch sequential planner ───────────────────────────────────────
    py::class_<fg::BatchAgent>(m, "BatchAgent")
        .def(py::init<>())
        .def_readwrite("start_x", &fg::BatchAgent::start_x)
        .def_readwrite("start_y", &fg::BatchAgent::start_y)
        .def_readwrite("goals", &fg::BatchAgent::goals);

    m.def("cxx_batch_sequential_plan",
          [](const std::vector<fg::BatchAgent>& agents,
             const std::vector<int>& order,
             const std::vector<int>& per_agent_max_time,
             py::array_t<int8_t, py::array::c_style | py::array::forcecast> passable_grid,
             py::array_t<int8_t, py::array::c_style | py::array::forcecast> shelf_grid,
             const std::string& mode,
             bool horizon_mode,
             double shelf_penalty,
             int planning_window,
             int k_robust,
             const std::vector<std::vector<std::tuple<int64_t,int,int>>>& initial_ct_flat,
             double deadline) -> py::dict
          {
              auto p_buf = passable_grid.request();
              int height = static_cast<int>(p_buf.shape[0]);
              int width  = static_cast<int>(p_buf.shape[1]);
              const int8_t* passable_ptr = static_cast<const int8_t*>(p_buf.ptr);

              auto s_buf = shelf_grid.request();
              const int8_t* shelf_ptr = static_cast<const int8_t*>(s_buf.ptr);

              fg::BatchPlanResult res;
              {
                  py::gil_scoped_release release;
                  res = fg::cxx_batch_sequential_plan(
                      agents, order, per_agent_max_time,
                      passable_ptr, shelf_ptr,
                      width, height,
                      mode, horizon_mode, shelf_penalty,
                      planning_window, k_robust,
                      initial_ct_flat, deadline);
              }

              // Convert to Python dict: {agent_idx: [(x,y), ...]}
              py::dict paths_dict;
              for (auto& [idx, path] : res.paths)
                  paths_dict[py::int_(idx)] = path;

              py::dict result;
              result["paths"] = paths_dict;
              result["success"] = res.success;
              result["nodes_expanded"] = res.nodes_expanded;
              return result;
          },
          py::arg("agents"),
          py::arg("order"),
          py::arg("per_agent_max_time"),
          py::arg("passable_grid"), py::arg("shelf_grid"),
          py::arg("mode"),
          py::arg("horizon_mode") = false,
          py::arg("shelf_penalty") = 3.0,
          py::arg("planning_window") = 10,
          py::arg("k_robust") = 0,
          py::arg("initial_ct_flat") = std::vector<std::vector<std::tuple<int64_t,int,int>>>(),
          py::arg("deadline") = 0.0,
          "Batch sequential multi-agent planning kernel.\n\n"
          "Plans agents sequentially in given order with constraint accumulation.\n"
          "mode: 'astar' or 'sipp'.\n"
          "Returns dict with 'paths', 'success', 'nodes_expanded'.");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "0.1.0";
#endif
}
