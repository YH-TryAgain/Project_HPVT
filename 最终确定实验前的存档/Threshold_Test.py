# -*- coding: utf-8 -*-
"""
HPVTAdaptiveAlgo 算法参数调优验证脚本

本脚本旨在通过实验找到 'ALOHA_THRESHOLD' 的最优值。
它会测试一系列不同的阈值，并对每个阈值下的算法性能
（特别是总耗时和首次发现时间）进行测量和对比，
最终通过表格和图表展示结果，以帮助确定最佳参数。
"""
import contextlib
import multiprocessing
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import os

# ==============================================================================
# 1. 导入依赖
# ==============================================================================
# 假设您的项目结构如下：
# - Framework.py (包含 run_missing_tag_simulation, print_results 等)
# - HPVT_Adaptive.py (包含 HPVTAdaptiveAlgo)
# 请根据您的实际文件名修改导入语句
from Framework import *
from HPVT_Adaptive import HPVTAdaptiveAlgo

# ==============================================================================
# 2. 核心工作函数 (与您的性能测试框架类似)
# ==============================================================================
def run_single_tuning_task(task_config: Dict) -> Dict:
    """在子进程中执行单次参数下的多次仿真，并返回平均结果。"""
    alg_class = task_config['alg_class']
    global_cfg = task_config['global_config']
    alg_specific_cfg = task_config['alg_specific_config']
    num_runs = task_config['num_runs']

    accumulated_results: Dict[str, float] = {}
    valid_runs = 0

    # 在子进程中，我们通常不希望看到详细的仿真日志
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        for _ in range(num_runs):
            result = run_missing_tag_simulation(global_cfg, alg_class, alg_specific_cfg)
            if result:
                valid_runs += 1
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        accumulated_results[key] = accumulated_results.get(key, 0.0) + value
    
    if valid_runs > 0:
        averaged_result = {key: val / valid_runs for key, val in accumulated_results.items()}
        averaged_result['threshold'] = alg_specific_cfg.get('ALOHA_THRESHOLD')
        return averaged_result
    return {}

# ==============================================================================
# 3. 结果展示函数
# ==============================================================================
def display_tuning_results(results: List[Dict]):
    """以表格和图表的形式展示参数调优的结果。"""
    if not results:
        print("没有有效的调优结果可供展示。")
        return

    # 按阈值大小对结果进行排序
    results.sort(key=lambda x: x['threshold'])

    # --- 打印对比表格 ---
    print("\n" + "="*80)
    print(f"{'HPVTAdaptiveAlgo 参数调优结果':^80}")
    print("="*80)
    header = f"{'ALOHA_THRESHOLD':<18} | {'总耗时 (秒)':<18} | {'首次发现时间 (ms)':<22} | {'总能耗 (Kbits)':<18}"
    print(header)
    print("-" * len(header))

    for res in results:
        threshold = res['threshold']
        total_time_s = res.get('total_protocol_time_us', 0) / 1e6
        first_missing_ms = res.get('time_to_first_missing', -1) / 1e3
        total_bits_kb = (res.get('total_reader_bits', 0) + res.get('total_tag_bits', 0)) / 1024

        first_missing_str = f"{first_missing_ms:.3f}" if first_missing_ms >= 0 else "N/A"

        print(f"{threshold:<18} | {total_time_s:<18.4f} | {first_missing_str:<22} | {total_bits_kb:<18.2f}")
    print("="*80)

    # --- 绘制性能权衡图 ---
    thresholds = [r['threshold'] for r in results]
    total_times = [r.get('total_protocol_time_us', 0) / 1e6 for r in results]
    first_missing_times = [r.get('time_to_first_missing', 0) / 1e3 for r in results]

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"\n警告: 设置中文字体失败，图表可能无法正常显示中文: {e}")

    # 绘制总耗时曲线
    color = 'tab:red'
    ax1.set_xlabel('ALOHA 启动阈值 (ALOHA_THRESHOLD)', fontsize=14)
    ax1.set_ylabel('总耗时 (秒)', color=color, fontsize=14)
    ax1.plot(thresholds, total_times, color=color, marker='o', linestyle='-', label='总耗时')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 创建第二个Y轴，共享X轴
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('首次发现时间 (毫秒)', color=color, fontsize=14)
    ax2.plot(thresholds, first_missing_times, color=color, marker='s', linestyle='--', label='首次发现时间')
    ax2.tick_params(axis='y', labelcolor=color)

    # 设置图表标题和图例
    plt.title('HPVTAdaptiveAlgo 性能权衡分析\n(随 ALOHA_THRESHOLD 变化)', fontsize=16)
    fig.tight_layout()
    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    # 保存图表
    folder = "tuning_results"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, "hpvt_threshold_tuning.png")
    plt.savefig(filename, bbox_inches='tight')
    print(f"\n✅ 性能权衡图已保存到: {filename}")
    plt.show()


# ==============================================================================
#  主程序入口
# ==============================================================================
if __name__ == '__main__':
    multiprocessing.freeze_support()
    print("启动 HPVTAdaptiveAlgo 参数调优脚本...")

    # 1. 定义要测试的阈值范围
    # 从一个较小的值开始，以指数方式增加，以探索大范围内的性能变化
    # thresholds_to_test = [8, 16, 32, 64, 128, 256, 512]
    thresholds_to_test = []
    i = 100
    while i <= 150:
        thresholds_to_test.append(i)
        i+=10

    # 2. 定义固定的全局仿真配置
    # 建议使用一个中等偏高的丢失率（如0.5），这样能更清晰地看到性能差异
    base_global_config = {
        'TOTAL_TAGS': 10000,
        'MISSING_RATE': 0.1, # 使用50%丢失率进行测试
        'BINARY_LENGTH': 96,
        'DATA_RATE_BPS': 160000.0,
        'BITS_PER_MICROSECOND': 160000.0 / 1.0e6,
        'T1_RTcal': 25.0,
        'T2_TRcal': 25.0,
        'MINIMAL_GUARD_TIME_US': 50.0,
        'READER_CMD_BASE_BITS': 37,
        'TAG_SHORT_RESP_BITS': 1,
    }

    # 3. 准备所有仿真任务
    num_runs_per_threshold = 5  # 运行5次取平均值，结果更可靠
    tasks = []
    for threshold in thresholds_to_test:
        alg_specific_config = {'ALOHA_THRESHOLD': threshold}
        tasks.append({
            'alg_class': HPVTAdaptiveAlgo,
            'global_config': base_global_config,
            'alg_specific_config': alg_specific_config,
            'num_runs': num_runs_per_threshold
        })

    # 4. 使用多进程执行所有任务
    print(f"\n准备对 {len(tasks)} 个不同的阈值进行测试 (每个阈值运行 {num_runs_per_threshold} 次)...")
    num_processes = min(multiprocessing.cpu_count() - 1, 8) if multiprocessing.cpu_count() > 1 else 1
    all_results = []
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        with tqdm(total=len(tasks), desc="执行参数调优", unit="task") as pbar:
            async_results = [pool.apply_async(run_single_tuning_task, (task,)) for task in tasks]
            for async_res in async_results:
                try:
                    task_result = async_res.get(timeout=600) # 10分钟超时
                    if task_result:
                        all_results.append(task_result)
                except Exception as e:
                    print(f"\n错误: 一个调优任务执行失败: {e}")
                pbar.update(1)

    # 5. 处理并展示结果
    if all_results:
        display_tuning_results(all_results)
    else:
        print("\n所有调优任务均未返回有效结果。")

    print("\n参数调优脚本执行完毕。")
