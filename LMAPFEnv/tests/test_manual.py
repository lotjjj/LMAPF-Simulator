from LMAPFEnv import MapConfig,WarehouseEnv
from LMAPFEnv.envs.entities import Direction

map_config = MapConfig(
    shelf_cols=1,
    shelf_rows=0,
    shelf_width=1,
    shelf_height=2,
    corridor_width=1,
    corridor_out_width=2
)

# 创建环境（开启渲染）
print("\n正在初始化环境...")
env = WarehouseEnv(
    num_agvs=5,
    fov_size=5,
    render_mode="human",
    enable_battery=False,
    map_config=map_config,
    max_episode_steps=5000
)

env.reset()

# 使用 teleport_agv 方法安全地设置 AGV 位置
env.teleport_agv('agv_0', 2, 2)
env.agvs['agv_0'].direction = Direction.RIGHT

env.teleport_agv('agv_1', 3, 2)
env.agvs['agv_1'].direction = Direction.DOWN

env.teleport_agv('agv_2', 3, 3)
env.agvs['agv_2'].direction = Direction.LEFT

env.teleport_agv('agv_3', 2, 3)
env.agvs['agv_3'].status = 'MOVING'
env.agvs['agv_3'].direction = Direction.UP

env.teleport_agv('agv_4', 1, 2)
env.agvs['agv_4'].status = 'CHARGING'
env.agvs['agv_4'].direction = Direction.RIGHT

while True:
    actions = { agent: 0 for agent in env.agents }
    env.step(actions)

    actions = {agent: 2 for agent in env.agents}
    actions['agv_4'] = 3
    env.step(actions)
