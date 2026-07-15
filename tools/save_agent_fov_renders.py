from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LMAPFEnv.utils import normalize_map_sizes, parse_planner_args


def _cell_base_rgb(fov_cell: np.ndarray) -> Tuple[float, float, float]:
    if fov_cell[1] > 0.5:
        return 0.05, 0.05, 0.05
    if fov_cell[2] > 0.5:
        return 0.65, 0.48, 0.28
    return 1.0, 1.0, 1.0


def _render_fov_image(fov: np.ndarray) -> np.ndarray:
    _, h, w = fov.shape
    img = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            img[y, x] = np.array(_cell_base_rgb(fov[:, y, x]), dtype=np.float32)
    return img


def _unsharp_mask_rgb(img: np.ndarray, amount: float = 0.8) -> np.ndarray:
    if amount <= 0:
        return np.clip(img, 0.0, 1.0)

    padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    blur = (
        padded[0:-2, 0:-2] + padded[0:-2, 1:-1] + padded[0:-2, 2:] +
        padded[1:-1, 0:-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
        padded[2:, 0:-2] + padded[2:, 1:-1] + padded[2:, 2:]
    ) / 9.0
    sharp = img + amount * (img - blur)
    return np.clip(sharp, 0.0, 1.0)


def _enhance_image(img: np.ndarray, contrast: float = 1.15, sharpness: float = 0.8) -> np.ndarray:
    out = np.clip(img, 0.0, 1.0)
    out = np.clip((out - 0.5) * float(contrast) + 0.5, 0.0, 1.0)
    out = _unsharp_mask_rgb(out, amount=float(sharpness))
    return out


def _path_points_in_fov(
    path_abs: np.ndarray,
    self_pos: Tuple[int, int],
    fov_h: int,
    fov_w: int,
) -> list[Tuple[int, int, int]]:
    r_y = fov_h // 2
    r_x = fov_w // 2
    self_x, self_y = self_pos
    points: list[Tuple[int, int, int]] = []

    arr = np.asarray(path_abs, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] <= 1:
        return points

    for step_idx, (x_raw, y_raw) in enumerate(arr[1:], start=1):
        x = int(round(float(x_raw)))
        y = int(round(float(y_raw)))
        fx = r_x + (x - self_x)
        fy = r_y + (y - self_y)
        if 0 <= fx < fov_w and 0 <= fy < fov_h:
            points.append((fx, fy, step_idx))
    return points


def _overlay_planner_paths(
    ax,
    planner_paths: dict[str, dict[str, Any]],
    self_agent: str,
    self_pos: Tuple[int, int],
    fov_h: int,
    fov_w: int,
    channel_base: int = 1,
    corner_offset: bool = False,
) -> None:
    cell_to_steps: dict[Tuple[int, int], list[int]] = {}
    for agent_name, path_info in planner_paths.items():
        if agent_name == self_agent:
            continue
        if not bool(path_info.get("alive", False)):
            continue
        for fx, fy, step_idx in _path_points_in_fov(path_info.get("path_abs"), self_pos, fov_h, fov_w):
            cell_to_steps.setdefault((fy, fx), []).append(step_idx + channel_base - 1)

    for (y, x), steps in cell_to_steps.items():
        txt = ",".join(str(step) for step in steps)
        tx = x - 0.24 if corner_offset else x
        ty = y + 0.24 if corner_offset else y
        ax.text(
            tx,
            ty,
            txt,
            ha="center",
            va="center",
            fontsize=7 if corner_offset else 8,
            color="crimson",
            fontweight="bold",
            bbox=dict(facecolor=(1, 1, 1, 0.72), edgecolor="none", boxstyle="round,pad=0.08"),
        )


def _overlay_markers(ax, fov: np.ndarray) -> None:
    _, h, w = fov.shape
    r_y = h // 2
    r_x = w // 2
    ax.scatter([r_x], [r_y], s=80, c=["dodgerblue"], marker="o", edgecolors="k", linewidths=0.8)
    for y in range(h):
        for x in range(w):
            if fov[0, y, x] > 0.5:
                ax.scatter([x], [y], s=90, c=["limegreen"], marker="D", edgecolors="k", linewidths=0.8)
            if fov[3, y, x] > 0.5:
                ax.scatter([x], [y], s=50, c=["orange"], marker="s", edgecolors="k", linewidths=0.6)


def _draw_grid(ax, h: int, w: int) -> None:
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color=(0, 0, 0, 0.25), linewidth=0.8)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)


def _overlay_other_agent_tracks(
    ax,
    planner_paths: dict[str, dict[str, Any]],
    self_agent: str,
    self_pos: Tuple[int, int],
    fov_h: int,
    fov_w: int,
    channel_base: int = 1,
    draw_step_labels: bool = True,
    draw_agent_labels: bool = False,
    linewidth: float = 2.0,
    marker_size: float = 30.0,
) -> None:
    import matplotlib.pyplot as plt

    if self_agent not in planner_paths:
        return

    other_names = sorted([name for name in planner_paths.keys() if name != self_agent])
    if not other_names:
        return

    cmap = plt.get_cmap("tab20")

    for idx, other_name in enumerate(other_names):
        path_info = planner_paths.get(other_name, {})
        if not bool(path_info.get("alive", False)):
            continue

        points = []
        for fx, fy, step_idx in _path_points_in_fov(path_info.get("path_abs"), self_pos, fov_h, fov_w):
            points.append((fx, fy, step_idx - 1))

        if not points:
            continue

        color = cmap(idx % 20)
        seg_x: list[int] = []
        seg_y: list[int] = []
        prev_k: Optional[int] = None

        def flush() -> None:
            if len(seg_x) >= 2:
                ax.plot(seg_x, seg_y, color=color, linewidth=float(linewidth), alpha=0.9)

        for fx, fy, k in points:
            if prev_k is None or k == prev_k + 1:
                seg_x.append(fx)
                seg_y.append(fy)
            else:
                flush()
                seg_x = [fx]
                seg_y = [fy]
            prev_k = k
        flush()

        ax.scatter(
            [p[0] for p in points],
            [p[1] for p in points],
            s=float(marker_size),
            c=[color],
            marker="o",
            edgecolors="k",
            linewidths=0.5,
        )

        if draw_step_labels:
            for fx, fy, k in points:
                ax.text(
                    fx,
                    fy,
                    str(int(k) + int(channel_base)),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                    fontweight="bold",
                )

        if draw_agent_labels:
            fx0, fy0, _ = points[0]
            ax.text(
                fx0,
                fy0 - 0.35,
                other_name,
                ha="center",
                va="bottom",
                fontsize=7,
                color=color,
                fontweight="bold",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="renders/fov")
    parser.add_argument("--map-sizes", nargs="*", default=None)
    parser.add_argument("--num-agvs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent", type=str, default="agv_0")
    parser.add_argument("--path-planner", type=str, default="RHCR")
    parser.add_argument("--planner-args", type=str, default=None)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--channel-base", type=int, default=1)
    parser.add_argument("--no-planner-paths", action="store_true")
    parser.add_argument("--per-agent-tracks", action="store_true")
    parser.add_argument("--no-track-step-labels", action="store_true")
    parser.add_argument("--track-agent-labels", action="store_true")
    parser.add_argument("--track-linewidth", type=float, default=2.0)
    parser.add_argument("--track-marker-size", type=float, default=30.0)
    parser.add_argument("--contrast", type=float, default=1.15)
    parser.add_argument("--sharpness", type=float, default=0.8)
    args = parser.parse_args()

    from LMAPFEnv.algorithms.path_planners import PlannerPolicy
    from LMAPFEnv.envs import WarehouseEnv
    import matplotlib.pyplot as plt

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    planner_args = parse_planner_args(args.planner_args)
    map_sizes = normalize_map_sizes(args.map_sizes)

    for map_size in map_sizes:
        env = WarehouseEnv(
            num_agvs=int(args.num_agvs),
            map_size=map_size,
            render_mode=None,
            path_planner=args.path_planner,
            planner_args=planner_args or None,
        )

        obs, infos = env.reset(seed=int(args.seed))
        policy = PlannerPolicy(env.path_planner)

        for _ in range(max(0, int(args.steps))):
            actions = policy.select_actions(env.agvs, env.agents)
            obs, _, _, _, infos = env.step(actions)
            if not env.agents:
                break

        if args.agent not in obs:
            raise ValueError(f"Unknown agent '{args.agent}', available: {list(obs.keys())}")

        agent_obs = obs[args.agent]
        if "fov" not in agent_obs:
            raise ValueError("observation does not contain 'fov'")

        fov = np.array(agent_obs["fov"], dtype=np.float32)
        agent_info = infos[args.agent]
        planner_paths = agent_info.get("planner_paths", {})
        self_pos_arr = np.array(planner_paths.get(args.agent, {}).get("path_abs", []), dtype=np.float32)
        if self_pos_arr.ndim != 2 or self_pos_arr.shape[1] != 2 or self_pos_arr.shape[0] == 0:
            self_pos = (
                int(env.agvs[args.agent].position.x),
                int(env.agvs[args.agent].position.y),
            )
        else:
            self_pos = (
                int(round(float(self_pos_arr[0, 0]))),
                int(round(float(self_pos_arr[0, 1]))),
            )
        planner_horizon = 0
        for path_info in planner_paths.values():
            path_abs = np.asarray(path_info.get("path_abs", []), dtype=np.float32)
            if path_abs.ndim == 2 and path_abs.shape[1] == 2 and path_abs.shape[0] > 0:
                planner_horizon = max(planner_horizon, int(path_abs.shape[0] - 1))

        img = _render_fov_image(fov)
        img = _enhance_image(img, contrast=float(args.contrast), sharpness=float(args.sharpness))
        h, w, _ = img.shape

        fig_w = max(2.0, w * 0.5)
        fig_h = max(2.0, h * 0.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=int(args.dpi))
        ax.imshow(img, origin="upper", interpolation="nearest")
        _draw_grid(ax, h, w)
        _overlay_markers(ax, fov)
        if not args.no_planner_paths:
            _overlay_planner_paths(
                ax,
                planner_paths,
                self_agent=args.agent,
                self_pos=self_pos,
                fov_h=int(h),
                fov_w=int(w),
                channel_base=int(args.channel_base),
                corner_offset=bool(args.per_agent_tracks),
            )
        if args.per_agent_tracks:
            _overlay_other_agent_tracks(
                ax,
                planner_paths,
                self_agent=args.agent,
                self_pos=self_pos,
                fov_h=int(h),
                fov_w=int(w),
                channel_base=int(args.channel_base),
                draw_step_labels=not bool(args.no_track_step_labels),
                draw_agent_labels=bool(args.track_agent_labels),
                linewidth=float(args.track_linewidth),
                marker_size=float(args.track_marker_size),
            )

        filename = f"fov_{map_size}_{args.agent}_step{env.current_step}_K{planner_horizon}.png"
        path = outdir / filename
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

        env.close()


if __name__ == "__main__":
    main()
