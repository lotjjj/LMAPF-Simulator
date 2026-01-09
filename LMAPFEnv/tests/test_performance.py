
import time

import numpy as np

from LMAPFEnv.envs.MAEnv import WarehouseEnv, BatteryConfig, MapConfig


def test_performance():
    print("=" * 60)
    print("开始性能测试 - 8 AGVs with Rendering")
    print("=" * 60)

    battery_config = BatteryConfig(
        energy_decay=0.005,
        energy_decay_noise=0.002,
        min_battery_level=0.0,
        max_battery_level=1.0,
        charging_rate=0.1,
        low_battery_threshold=0.1
    )
    
    map_config = MapConfig(
        shelf_cols=1,
        shelf_rows=2,
        shelf_width=1,
        shelf_height=2,
        corridor_width=1,
        corridor_out_width=2
    )
    
    # 创建环境（开启渲染）
    print("\n正在初始化环境...")
    env = WarehouseEnv(
        num_agvs=8,
        fov_size=5,
        render_mode="human",
        enable_battery=True,
        battery_config=battery_config,
        map_config=map_config,
        max_episode_steps=5000
    )
    
    print(f"环境初始化完成")
    print(f"地图大小: {env.width} x {env.height}")
    print(f"AGV数量: {env.num_agvs}")
    print(f"最大步数: {env.max_episode_steps}")
    
    # 重置环境
    print("\n正在重置环境...")
    observations, infos = env.reset(seed=42)
    print("环境重置完成")
    
    # 性能统计
    total_steps = 0
    step_times = []
    episode_start_time = time.time()
    
    print("\n开始运行测试...")
    print("-" * 60)
    
    try:
        for step in range(env.max_episode_steps):
            step_start = time.time()
            
            # 随机动作
            actions = {
                agent: env.action_space(agent).sample()
                for agent in env.agents
            }
            
            # 执行一步
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # 渲染
            env.render()
            
            step_end = time.time()
            step_time = step_end - step_start
            step_times.append(step_time)
            total_steps += 1
            
            # 每50步输出一次统计信息
            if (step + 1) % 50 == 0:
                avg_time = np.mean(step_times[-50:])
                fps = 1.0 / avg_time if avg_time > 0 else 0
                print(f"Step {step + 1:4d} | "
                      f"当前步耗时: {step_time*1000:6.2f}ms | "
                      f"平均耗时: {avg_time*1000:6.2f}ms | "
                      f"FPS: {fps:5.2f}")
            
            # 检查是否所有agent都完成
            if all(terminations.values()) or all(truncations.values()):
                print(f"\n所有agent在步骤 {step + 1} 完成")
                break
    
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    
    except Exception as e:
        print(f"\n\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        episode_end_time = time.time()
        total_time = episode_end_time - episode_start_time
        
        # 输出最终统计信息
        print("\n" + "=" * 60)
        print("性能测试结果")
        print("=" * 60)
        print(f"总步数: {total_steps}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均每步耗时: {np.mean(step_times)*1000:.2f}ms")
        print(f"最快步耗时: {np.min(step_times)*1000:.2f}ms")
        print(f"最慢步耗时: {np.max(step_times)*1000:.2f}ms")
        print(f"平均FPS: {total_steps/total_time:.2f}")
        print("=" * 60)
        
        # 关闭环境
        env.close()
        print("\n环境已关闭")


