"""
任务系统测试脚本 - 测试 AGV 任务完成和奖励机制
"""
import time

from LMAPFEnv import MapConfig, WarehouseEnv, Tasks
from LMAPFEnv.envs.entities import Direction

# 创建环境配置
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
    num_agvs=3,
    fov_size=5,
    render_mode="human",
    enable_battery=False,
    map_config=map_config,
    max_episode_steps=500
)

env.reset()

# 手动设置 AGV 位置和方向（使用 teleport_agv）
env.teleport_agv('agv_0', 2, 2)
env.agvs['agv_0'].direction = Direction.RIGHT

env.teleport_agv('agv_1', 4, 2)
env.agvs['agv_1'].direction = Direction.LEFT

env.teleport_agv('agv_2', 3, 4)
env.agvs['agv_2'].direction = Direction.UP

print("\n初始 AGV 位置和目标:")
tasks = Tasks()
for agent_name, agv in env.agvs.items():
    task = tasks.get_task(agv.id)
    target_pos = task.target_pos if task else None
    print(f"  {agent_name}: 位置=({agv.x}, {agv.y}), 方向={agv.direction.name}, 目标={target_pos}")

print("\n开始任务测试循环...")
print("-" * 60)

try:
    step_count = 0
    total_completed = 0

    for step in range(env.max_episode_steps):
        step_count += 1

        # 随机动作
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }

        # 执行一步
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # 渲染
        env.render()

        # 检查任务完成情况
        for agent_name in env.agents:
            agv = env.agvs[agent_name]
            task = tasks.get_task(agv.id)
            if task and task.status.value == 1:  # TaskStatus.COMPLETED
                total_completed += 1

        # 每10步输出统计信息
        if (step + 1) % 10 == 0:
            print(f"\nStep {step + 1}:")
            print(f"  当前完成任务数: {total_completed}")
            print(f"  当前 AGV 数量: {env.num_agents}")

            for agent_name, agv in env.agvs.items():
                task = tasks.get_task(agv.id)
                target_pos = task.target_pos if task else None
                status = task.status.name if task else "None"
                dist = abs(agv.x - target_pos[0]) + abs(agv.y - target_pos[1]) if target_pos else 0
                print(f"    {agent_name}: 位置=({agv.x}, {agv.y}), 目标={target_pos}, 状态={status}, 距离={dist}")

        # 检查是否所有agent都完成
        if all(terminations.values()) or all(truncations.values()):
            print(f"\n所有agent在步骤 {step + 1} 完成")
            break

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\n测试被用户中断")
except Exception as e:
    print(f"\n\n测试过程中发生错误: {e}")
    import traceback
    traceback.print_exc()
finally:
    # 输出最终统计信息
    print("\n" + "=" * 60)
    print("任务测试结果")
    print("=" * 60)
    print(f"总步数: {step_count}")
    print(f"完成任务总数: {total_completed}")

    # 输出每个 AGV 的任务统计
    print("\n各 AGV 任务统计:")
    for agent_name, agv in env.agvs.items():
        task = tasks.get_task(agv.id)
        if task:
            print(f"  {agent_name}: 最后目标={task.target_pos}, 状态={task.status.name}")
        else:
            print(f"  {agent_name}: 无任务")

    print("=" * 60)

    # 关闭环境
    env.close()
    print("\n环境已关闭")
