from __future__ import annotations

from dataclasses import MISSING, dataclass, fields, replace
from typing import Any, Dict, Literal, Optional, Type

PlannerType = Literal['CBS', 'ECBS', 'PBS', 'AStar', 'EnhancedAStar', 'RHCR', 'RHCR_CBS', 'RHCR_ECBS', 'RHCR_PBS']


@dataclass(frozen=True)
class LegacyRewardConfig:
    each_step_reward: float = -0.002
    invalid_action_penalty: float = -0.05
    conflict_penalty: float = -0.6
    progress_shaping_weight: float = 0.01
    task_completion_reward: float = 2.0


DEFAULT_LEGACY_REWARD_CONFIG = LegacyRewardConfig()


def get_legacy_reward_config() -> LegacyRewardConfig:
    return DEFAULT_LEGACY_REWARD_CONFIG


@dataclass(frozen=True)
class PlannerParamSpec:
    key: str
    label: str
    default: Any
    options: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class PlannerSpec:
    planner_type: PlannerType
    config_cls: Type["PlannerConfigBase"]
    params: tuple[PlannerParamSpec, ...]

    def default_config(self) -> "PlannerConfigBase":
        return self.config_cls()


@dataclass(frozen=True)
class PlannerConfigBase:
    planner_type: PlannerType
    shelf_penalty: Optional[float] = None

    def to_planner_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.shelf_penalty is not None:
            kwargs["shelf_penalty"] = float(self.shelf_penalty)
        return kwargs

    def with_overrides(self, overrides: Optional[Dict[str, Any]]) -> "PlannerConfigBase":
        if not overrides:
            return self
        valid = {k: v for k, v in overrides.items() if hasattr(self, k)}
        if not valid:
            return self
        return replace(self, **valid)


@dataclass(frozen=True)
class AStarConfig(PlannerConfigBase):
    planner_type: PlannerType = 'AStar'


@dataclass(frozen=True)
class EnhancedAStarConfig(PlannerConfigBase):
    planner_type: PlannerType = 'EnhancedAStar'
    visible_agv_penalty: float = 5.0

    def to_planner_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_planner_kwargs()
        kwargs["visible_agv_penalty"] = float(self.visible_agv_penalty)
        return kwargs


@dataclass(frozen=True)
class CBSConfig(PlannerConfigBase):
    planner_type: PlannerType = 'CBS'
    max_cbs_nodes: int = 200000
    max_low_level_steps: int = 500
    max_planning_time: float = 5.0
    conflict_horizon: int = 200

    def to_planner_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_planner_kwargs()
        kwargs["max_cbs_nodes"] = int(self.max_cbs_nodes)
        kwargs["max_low_level_steps"] = int(self.max_low_level_steps)
        kwargs["max_planning_time"] = float(self.max_planning_time)
        kwargs["conflict_horizon"] = int(self.conflict_horizon)
        return kwargs


@dataclass(frozen=True)
class ECBSConfig(PlannerConfigBase):
    planner_type: PlannerType = 'ECBS'
    w: float = 1.5
    max_cbs_nodes: int = 200000
    max_low_level_steps: int = 500
    max_planning_time: float = 5.0
    conflict_horizon: int = 200

    def to_planner_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_planner_kwargs()
        kwargs["w"] = float(self.w)
        kwargs["max_cbs_nodes"] = int(self.max_cbs_nodes)
        kwargs["max_low_level_steps"] = int(self.max_low_level_steps)
        kwargs["max_planning_time"] = float(self.max_planning_time)
        kwargs["conflict_horizon"] = int(self.conflict_horizon)
        return kwargs


@dataclass(frozen=True)
class PBSConfig(PlannerConfigBase):
    planner_type: PlannerType = 'PBS'
    max_pbs_nodes: int = 200000
    max_low_level_steps: int = 500
    max_planning_time: float = 5.0
    conflict_horizon: int = 200

    def to_planner_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_planner_kwargs()
        kwargs["max_pbs_nodes"] = int(self.max_pbs_nodes)
        kwargs["max_low_level_steps"] = int(self.max_low_level_steps)
        kwargs["max_planning_time"] = float(self.max_planning_time)
        kwargs["conflict_horizon"] = int(self.conflict_horizon)
        return kwargs


@dataclass(frozen=True)
class RHCRConfig(PlannerConfigBase):
    planner_type: PlannerType = 'RHCR'
    planning_window: int = 10
    horizon: int = 5
    max_low_level_steps: int = 500
    random_seed: Optional[int] = 42
    use_sipp: bool = False
    k_robust: int = 0
    suboptimal_bound: float = 1.0

    def to_planner_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_planner_kwargs()
        kwargs["planning_window"] = int(self.planning_window)
        kwargs["horizon"] = int(self.horizon)
        kwargs["max_low_level_steps"] = int(self.max_low_level_steps)
        kwargs["random_seed"] = None if self.random_seed is None else int(self.random_seed)
        kwargs["use_sipp"] = bool(self.use_sipp)
        kwargs["k_robust"] = int(self.k_robust)
        kwargs["suboptimal_bound"] = float(self.suboptimal_bound)
        return kwargs



@dataclass(frozen=True)
class RHCRCBSConfig(PlannerConfigBase):
    planner_type: PlannerType = 'RHCR_CBS'
    planning_window: int = 10
    horizon: int = 5
    max_cbs_nodes: int = 50000
    max_low_level_steps: int = 100
    max_planning_time: float = 5.0
    use_sipp: bool = False
    k_robust: int = 0
    suboptimal_bound: float = 1.0

    def to_planner_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_planner_kwargs()
        kwargs["planning_window"] = int(self.planning_window)
        kwargs["horizon"] = int(self.horizon)
        kwargs["max_cbs_nodes"] = int(self.max_cbs_nodes)
        kwargs["max_low_level_steps"] = int(self.max_low_level_steps)
        kwargs["max_planning_time"] = float(self.max_planning_time)
        kwargs["use_sipp"] = bool(self.use_sipp)
        kwargs["k_robust"] = int(self.k_robust)
        kwargs["suboptimal_bound"] = float(self.suboptimal_bound)
        return kwargs


@dataclass(frozen=True)
class RHCRECBSConfig(PlannerConfigBase):
    planner_type: PlannerType = 'RHCR_ECBS'
    planning_window: int = 10
    horizon: int = 5
    w: float = 1.5
    max_cbs_nodes: int = 50000
    max_low_level_steps: int = 100
    max_planning_time: float = 5.0
    use_sipp: bool = False
    k_robust: int = 0
    suboptimal_bound: float = 1.5

    def to_planner_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_planner_kwargs()
        kwargs["planning_window"] = int(self.planning_window)
        kwargs["horizon"] = int(self.horizon)
        kwargs["w"] = float(self.w)
        kwargs["max_cbs_nodes"] = int(self.max_cbs_nodes)
        kwargs["max_low_level_steps"] = int(self.max_low_level_steps)
        kwargs["max_planning_time"] = float(self.max_planning_time)
        kwargs["use_sipp"] = bool(self.use_sipp)
        kwargs["k_robust"] = int(self.k_robust)
        kwargs["suboptimal_bound"] = float(self.suboptimal_bound)
        return kwargs


@dataclass(frozen=True)
class RHCRPBSConfig(PlannerConfigBase):
    planner_type: PlannerType = 'RHCR_PBS'
    planning_window: int = 10
    horizon: int = 5
    max_pbs_nodes: int = 50000
    max_low_level_steps: int = 100
    max_planning_time: float = 5.0
    use_sipp: bool = False
    k_robust: int = 0
    suboptimal_bound: float = 1.0

    def to_planner_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_planner_kwargs()
        kwargs["planning_window"] = int(self.planning_window)
        kwargs["horizon"] = int(self.horizon)
        kwargs["max_pbs_nodes"] = int(self.max_pbs_nodes)
        kwargs["max_low_level_steps"] = int(self.max_low_level_steps)
        kwargs["max_planning_time"] = float(self.max_planning_time)
        kwargs["use_sipp"] = bool(self.use_sipp)
        kwargs["k_robust"] = int(self.k_robust)
        kwargs["suboptimal_bound"] = float(self.suboptimal_bound)
        return kwargs


FIELD_LABELS: Dict[str, str] = {
    "shelf_penalty": "Shelf Penalty",
    "visible_agv_penalty": "Visible AGV Penalty",
    "max_low_level_steps": "Max Low-Level Steps",
    "random_seed": "Random Seed",
    "priority_strategy": "Priority Strategy",
    "max_cbs_nodes": "Max CBS Nodes",
    "max_pbs_nodes": "Max PBS Nodes",
    "max_planning_time": "Max Planning Time",
    "planning_window": "Planning Window",
    "horizon": "Horizon",
    "w": "Suboptimal Weight (w)",
    "use_sipp": "Use SIPP Search",
    "k_robust": "K-Robust Constraint",
    "suboptimal_bound": "SIPP Suboptimal Bound",
}

FIELD_OPTIONS: Dict[tuple[str, str], tuple[str, ...]] = {
}


def _build_param_specs(config_cls: Type[PlannerConfigBase]) -> tuple[PlannerParamSpec, ...]:
    cfg = config_cls()
    specs: list[PlannerParamSpec] = []
    for cfg_field in fields(cfg):
        if cfg_field.name == "planner_type":
            continue
        if cfg_field.default is not MISSING:
            default_value = cfg_field.default
        elif cfg_field.default_factory is not MISSING:  # type: ignore[attr-defined]
            default_value = cfg_field.default_factory()  # type: ignore[misc]
        else:
            default_value = getattr(cfg, cfg_field.name)
        specs.append(
            PlannerParamSpec(
                key=cfg_field.name,
                label=FIELD_LABELS.get(cfg_field.name, cfg_field.name.replace("_", " ").title()),
                default=default_value,
                options=FIELD_OPTIONS.get((cfg.planner_type, cfg_field.name), ()),
            )
        )
    return tuple(specs)


PLANNER_REGISTRY: Dict[PlannerType, PlannerSpec] = {
    "AStar": PlannerSpec("AStar", AStarConfig, _build_param_specs(AStarConfig)),
    "EnhancedAStar": PlannerSpec("EnhancedAStar", EnhancedAStarConfig, _build_param_specs(EnhancedAStarConfig)),
    "CBS": PlannerSpec("CBS", CBSConfig, _build_param_specs(CBSConfig)),
    "ECBS": PlannerSpec("ECBS", ECBSConfig, _build_param_specs(ECBSConfig)),
    "PBS": PlannerSpec("PBS", PBSConfig, _build_param_specs(PBSConfig)),
    "RHCR": PlannerSpec("RHCR", RHCRConfig, _build_param_specs(RHCRConfig)),
    "RHCR_CBS": PlannerSpec("RHCR_CBS", RHCRCBSConfig, _build_param_specs(RHCRCBSConfig)),
    "RHCR_ECBS": PlannerSpec("RHCR_ECBS", RHCRECBSConfig, _build_param_specs(RHCRECBSConfig)),
    "RHCR_PBS": PlannerSpec("RHCR_PBS", RHCRPBSConfig, _build_param_specs(RHCRPBSConfig)),
}


def get_default_planner_config(planner_type: Optional[PlannerType]) -> Optional[PlannerConfigBase]:
    if planner_type is None:
        return None
    spec = PLANNER_REGISTRY.get(planner_type)
    if spec is not None:
        return spec.default_config()
    raise ValueError(f"Unknown planner_type: {planner_type}")
