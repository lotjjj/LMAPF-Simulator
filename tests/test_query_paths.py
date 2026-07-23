from __future__ import annotations

import copy
import pickle
import unittest

from LMAPFEnv import PathQueryResult, WarehouseEnv


def _positions(paths):
    return {
        aid: tuple((pos.x, pos.y) for pos in path)
        for aid, path in paths.items()
    }


def _planner_state(planner):
    if planner is None:
        return None
    state = {
        "paths": _positions(planner.get_paths()),
        "heads": planner.get_path_heads(),
        "goals": {
            aid: tuple((pos.x, pos.y) for pos in goals)
            for aid, goals in planner._goal_sequences.items()
        },
        "success": planner.is_last_plan_successful(),
        "last_timing": copy.deepcopy(planner.last_timing),
    }
    if hasattr(planner, "_initial_constraints"):
        state["initial_constraints"] = {
            aid: tuple(((pos.x, pos.y), time_step) for pos, time_step in constraints)
            for aid, constraints in planner._initial_constraints.items()
        }
    if hasattr(planner, "_rng"):
        state["rng"] = planner._rng.getstate()
    return state


def _task_state(base_env):
    return {
        aid: tuple(
            ((task.target_pos.x, task.target_pos.y), task.status, task.assigned_agent)
            for task in tasks
        )
        for aid, tasks in base_env.task_manager.snapshot_all().items()
    }


class QueryPathsTests(unittest.TestCase):
    def test_requires_reset_and_validates_arguments(self):
        env = WarehouseEnv(num_agvs=1, map_size="small", path_planner="AStar")
        self.addCleanup(env.close)

        with self.assertRaises(RuntimeError):
            env.query_paths("RHCR")

        env.reset(seed=7)
        with self.assertRaises(ValueError):
            env.query_paths("NOT_A_PLANNER")
        with self.assertRaises(ValueError):
            env.query_paths("RHCR", {"planning_widnow": 10})
        with self.assertRaises(ValueError):
            env.query_paths("RHCR", timeout=0)

    def test_query_result_is_detached_and_does_not_advance_environment(self):
        env = WarehouseEnv(num_agvs=2, map_size="small", path_planner="AStar")
        self.addCleanup(env.close)
        env.reset(seed=11)
        env.step({agent: 4 for agent in env.agents})
        base = env._env

        positions_before = {name: agv.position for name, agv in env.agvs.items()}
        tasks_before = _task_state(base)
        main_before = _planner_state(base.path_planner)
        rng_before = pickle.dumps(base.np_random.bit_generator.state)

        result = env.query_paths(
            "RHCR",
            {"planning_window": 8, "horizon": 4, "max_low_level_steps": 100},
            timeout=2.0,
        )

        self.assertIsInstance(result, PathQueryResult)
        self.assertTrue(result.success, result.failure_reason)
        self.assertEqual(set(result.paths), set(env.agents))
        self.assertEqual(result.current_step, 1)
        self.assertEqual(base._current_step, 1)
        self.assertEqual(
            {name: agv.position for name, agv in env.agvs.items()}, positions_before
        )
        self.assertEqual(_task_state(base), tasks_before)
        self.assertEqual(_planner_state(base.path_planner), main_before)
        self.assertEqual(pickle.dumps(base.np_random.bit_generator.state), rng_before)

        first_agent = next(iter(result.paths))
        result.paths[first_agent].append((-1, -1))
        self.assertEqual(_planner_state(base.path_planner), main_before)

    def test_query_does_not_change_internal_rhcr_or_observation_rhcr(self):
        env = WarehouseEnv(
            num_agvs=2,
            map_size="small",
            path_planner="RHCR",
            planner_args={
                "planning_window": 8,
                "horizon": 4,
                "max_low_level_steps": 100,
                "obs_planner_type": "RHCR",
                "obs_planner_args": {
                    "planning_window": 6,
                    "horizon": 3,
                    "max_low_level_steps": 100,
                },
            },
        )
        self.addCleanup(env.close)
        env.reset(seed=19)
        base = env._env

        main_before = _planner_state(base.path_planner)
        obs_before = _planner_state(base.obs_path_planner)
        cache_before = copy.deepcopy(base._planner_paths_info_cache)
        meta_before = pickle.dumps(base._info_cache)

        result = env.query_paths(
            "RHCR",
            {
                "planning_window": 8,
                "horizon": 4,
                "max_low_level_steps": 100,
                "k_robust": 1,
                "use_sipp": True,
            },
            timeout=2.0,
            use_current_constraints=True,
        )

        self.assertTrue(result.success, result.failure_reason)
        self.assertEqual(_planner_state(base.path_planner), main_before)
        self.assertEqual(_planner_state(base.obs_path_planner), obs_before)
        self.assertEqual(pickle.dumps(base._info_cache), meta_before)
        for agent_name, cached in cache_before.items():
            current = base._planner_paths_info_cache[agent_name]
            self.assertEqual(cached.keys(), current.keys())
            self.assertTrue((cached["path_abs"] == current["path_abs"]).all())
            self.assertEqual(cached["alive"], current["alive"])
            self.assertEqual(cached["has_path"], current["has_path"])

    def test_astar_path_sufficiency_uses_partial_replan_and_reports_info(self):
        env = WarehouseEnv(num_agvs=2, map_size="small", path_planner="AStar")
        self.addCleanup(env.close)
        env.reset(seed=23)
        base = env._env
        planner = base.path_planner

        target_agent = env.agents[0]
        target_id = base.agvs[target_agent].id
        pw = base.path_window

        for agent_name in env.agents:
            agv = base.agvs[agent_name]
            goals = base.task_manager.goal_sequence(agv.id)
            current = agv.position
            if agent_name == target_agent:
                planner.set_path(agv.id, [current])
                base._planner_goal_sequences_snapshot[agv.id] = tuple()
                continue

            final_goal = goals[-1] if goals else current
            planner.set_path(agv.id, [current] * (pw + 1) + [final_goal])
            planner.set_path_head(agv.id, 0)
            base._planner_goal_sequences_snapshot[agv.id] = tuple(goals)

        replan_flag, obs_replan_flag, replan_agents = base._evaluate_replan_triggers(set())

        self.assertTrue(replan_flag)
        self.assertFalse(obs_replan_flag)
        self.assertEqual(replan_agents, {target_agent})

        base._begin_step_planner_meta()
        base._plan_paths(agent_names=replan_agents)
        info = base.get_info(target_agent)
        planner_meta = info["planner_meta"]

        self.assertTrue(planner_meta["replanned"])
        self.assertTrue(planner_meta["partial_replan"])
        self.assertEqual(planner_meta["replanned_agents"], [target_agent])
        self.assertFalse(planner_meta["skipped"])


if __name__ == "__main__":
    unittest.main()
