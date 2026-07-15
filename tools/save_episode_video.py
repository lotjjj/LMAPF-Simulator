from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LMAPFEnv.algorithms.path_planners import PlannerPolicy
from LMAPFEnv.envs import WarehouseEnv
from LMAPFEnv.utils import parse_planner_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record one planner-driven warehouse episode as a compact, README-friendly MP4."
    )
    parser.add_argument("--out", type=str, default="renders/episode_demo.mp4")
    parser.add_argument("--num-agvs", type=int, default=100)
    parser.add_argument("--map-size", type=str, default="medium")
    parser.add_argument("--path-planner", type=str, default="RHCR")
    parser.add_argument("--planner-args", type=str, default="planning_window=10,horizon=5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=0.2, help="Video seconds to hold each environment step.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--crf", type=int, default=24)
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--hide-trajectories", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _require_imageio() -> Any:
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing video dependency. Install project dependencies with "
            "`pip install -e .` or install `imageio[ffmpeg]`."
        ) from exc
    return imageio


def _fit_even_size(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale to a README-friendly width and H.264-compatible even size."""
    import pygame

    h, w = frame.shape[:2]
    target_w = int(w)
    target_h = int(h)
    if max_width > 0 and w > max_width:
        scale = float(max_width) / float(w)
        target_w = int(round(w * scale))
        target_h = int(round(h * scale))

    target_w = max(2, target_w - (target_w % 2))
    target_h = max(2, target_h - (target_h % 2))
    if target_w == w and target_h == h:
        return frame

    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    scaled = pygame.transform.smoothscale(surface, (target_w, target_h))
    return np.transpose(pygame.surfarray.array3d(scaled), (1, 0, 2))


def _capture_frame(env: WarehouseEnv, *, max_width: int, hide_trajectories: bool) -> np.ndarray:
    frame = _capture_progress_frame(
        env,
        progress=1.0,
        max_width=max_width,
        hide_trajectories=hide_trajectories,
    )
    return frame


def _capture_progress_frame(
    env: WarehouseEnv,
    *,
    progress: float,
    max_width: int,
    hide_trajectories: bool,
) -> np.ndarray:
    if not hasattr(env, "_rgb_widget") or env._rgb_widget is None:
        frame = env.render(if_continuous=False)
        if frame is None:
            raise RuntimeError("render_mode='rgb_array' did not return a frame")

    widget = env._rgb_widget
    if hide_trajectories and hasattr(env, "_rgb_widget") and env._rgb_widget is not None:
        widget.set_trajectory_visibility(False)

    widget._poll_events()
    widget._render_frame(progress=progress)
    frame = np.transpose(widget.get_rgb_array(), (1, 0, 2))
    frame = np.asarray(frame, dtype=np.uint8)
    return _fit_even_size(frame, max_width=max_width)


def _append_frame(writer: Any, frame: np.ndarray, repeat: int) -> int:
    repeat = max(1, int(repeat))
    for _ in range(repeat):
        writer.append_data(frame)
    return repeat


def _append_step_frames(
    writer: Any,
    env: WarehouseEnv,
    *,
    frame_count: int,
    max_width: int,
    hide_trajectories: bool,
) -> int:
    frame_count = max(1, int(frame_count))
    for frame_idx in range(1, frame_count + 1):
        progress = frame_idx / frame_count
        writer.append_data(_capture_progress_frame(
            env,
            progress=progress,
            max_width=max_width,
            hide_trajectories=hide_trajectories,
        ))
    return frame_count


def main() -> None:
    args = parse_args()
    imageio = _require_imageio()
    fps = max(1, int(args.fps))
    step_frame_repeat = max(1, int(round(max(0.0, float(args.sleep)) * fps)))

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = WarehouseEnv(
        num_agvs=int(args.num_agvs),
        map_size=args.map_size,
        path_planner=args.path_planner,
        planner_args=parse_planner_args(args.planner_args) or None,
        render_mode="rgb_array",
        if_continuous=False,
        max_episode_steps=max(1, int(args.steps)),
    )

    frames_written = 0
    completions = 0
    conflicts = 0

    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        macro_block_size=2,
        ffmpeg_params=[
            "-crf",
            str(max(0, min(51, int(args.crf)))),
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ],
    )

    try:
        _, infos = env.reset(seed=int(args.seed))
        policy = PlannerPolicy(env.path_planner)
        policy.update_info(infos)

        frames_written += _append_frame(writer, _capture_frame(
            env,
            max_width=int(args.max_width),
            hide_trajectories=bool(args.hide_trajectories),
        ), repeat=1)

        for step_idx in range(max(0, int(args.steps))):
            actions = policy.select_actions(env.agvs, env.agents, infos)
            _, _, _, _, infos = env.step(actions)
            policy.update_info(infos)

            for agent_info in infos.values():
                completions += int(bool(agent_info.get("task_completed", False)))
                conflicts += int(bool(agent_info.get("conflicted", False)))

            frames_written += _append_step_frames(
                writer,
                env,
                frame_count=step_frame_repeat,
                max_width=int(args.max_width),
                hide_trajectories=bool(args.hide_trajectories),
            )

            if args.verbose:
                print(f"recorded step {step_idx + 1}/{args.steps} ({step_frame_repeat} video frames)")
            if not env.agents:
                break
    finally:
        writer.close()
        env.close()

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(
        f"Saved {frames_written} frames to {out_path} "
        f"({frames_written / fps:.2f} s, {size_mb:.2f} MB, "
        f"completions={completions}, conflicts={conflicts})"
    )


if __name__ == "__main__":
    main()
