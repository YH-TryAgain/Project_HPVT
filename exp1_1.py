# -*- coding: utf-8 -*-
"""
【多维度分析版】实验 1.1: 性能与缺失率的关系

================================================================================
本实验的核心目标 (Objective):
1. 在之前的基础上，从多个维度深入分析缺失率 p_m 对 HPVT 性能的影响，包括：
   - a. 总执行时间 (时间效率)
   - b. 总通信比特数 (能量消耗)
   - c. 剪枝事件数 (核心机制效率)
   - d. 并行验证启动次数 (模式切换行为)
2. 通过一个 2x2 的组合图表，直观地展示这些指标如何相互关联，
   从而从机理上解释 HPVT “缺失率越高，性能越好” 的原因。
================================================================================
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import multiprocessing
from itertools import product
from tqdm import tqdm

# 导入您已升级的、支持多维度指标追踪的框架和算法
from framework import run_missing_tag_simulation
from hpvt_algo import HPVT_AggregatedAlgo

# ==============================================================================
# 1. 实验配置区 (Configuration Area)
# ==============================================================================
print("正在加载实验 1.1 (多维度分析版) 的配置...")

# --- 图表与结果文件配置 ---
EXPERIMENT_NAME = r"Experiment 1.1: Comprehensive Analysis vs. Missing Rate ($p_m$)"
OUTPUT_DIR = "plots"
RESULTS_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_PLOT_FILE = os.path.join(OUTPUT_DIR, "fig1.1_multimetric_vs_missing_rate.png")
OUTPUT_CSV_FILE = os.path.join(RESULTS_DIR, "exp1.1_multimetric_vs_missing_rate_summary.csv")

# --- 仿真场景固定参数 ---
SCENARIO_CONFIG = {
    'TOTAL_TAGS': 20000,
    'BINARY_LENGTH': 96,
}

# --- 仿真可变参数 ---
PM_RANGE = np.linspace(0.1, 0.9, 9)
NUM_RUNS = 20

# --- 待测算法列表 ---
ALGORITHMS_TO_TEST = [
    {
        "class": HPVT_AggregatedAlgo,
        "name": "HPVT", # 名字简化，因为只测试一个算法
        "config": {"aloha_threshold": 128, "binary_length": 96}
    },
]

# ==============================================================================
# 2. 并行任务函数 (Worker Function for Parallel Execution)
# ==============================================================================
def run_single_task(task_params: tuple):
    """
    执行单次仿真任务，并返回一个包含多维度指标的记录。
    """
    algo_info, pm, run_id, scenario_config_base = task_params

    current_scenario_config = scenario_config_base.copy()
    current_scenario_config['MISSING_RATE'] = pm

    # 调用已升级的仿真框架，它会返回包含新指标的result
    result = run_missing_tag_simulation(
        scenario_config=current_scenario_config,
        algorithm_class=algo_info['class'],
        algorithm_specific_config=algo_info['config']
    )

    # 在记录中加入所有需要的指标
    record = {
        "algorithm": algo_info['name'],
        "missing_rate": pm,
        "run_id": run_id,
        "total_time_ms": result.get('total_protocol_time_us', 0) / 1000.0,
        "total_bits": result.get('total_reader_bits', 0) + result.get('total_tag_bits', 0),
        "pruning_events": result.get('pruning_events', 0),
        "aloha_frames": result.get('aloha_frames_initiated', 0),
    }
    return record

# ==============================================================================
# 3. 主执行逻辑 (Main Execution Logic)
# ==============================================================================
def main():
    """
    主函数，负责生成任务、创建进程池并执行仿真，最后处理并展示结果。
    """
    print("正在准备并行仿真任务 (多维度)...")
    tasks_to_run = list(product(ALGORITHMS_TO_TEST, PM_RANGE, range(NUM_RUNS)))
    tasks_with_config = [(task[0], task[1], task[2], SCENARIO_CONFIG) for task in tasks_to_run]
    total_simulations = len(tasks_with_config)
    print(f"任务总数: {total_simulations}")

    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"将使用 {num_processes} 个CPU核心并行执行...")

    results_list = []; start_time = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(run_single_task, tasks_with_config)
        for result in tqdm(results_iterator, total=total_simulations, desc="执行仿真中"):
            results_list.append(result)
    end_time = time.time()
    print(f"并行实验执行完毕。总耗时: {end_time - start_time:.2f} 秒")

    print("正在处理和分析数据...")
    df_raw = pd.DataFrame(results_list)

    # 按缺失率分组，聚合所有需要的指标
    df_summary = df_raw.groupby(['missing_rate']).agg(
        mean_time=('total_time_ms', 'mean'), std_time=('total_time_ms', 'std'),
        mean_bits=('total_bits', 'mean'), std_bits=('total_bits', 'std'),
        mean_pruning=('pruning_events', 'mean'), std_pruning=('pruning_events', 'std'),
        mean_aloha=('aloha_frames', 'mean'), std_aloha=('aloha_frames', 'std')
    ).reset_index()

    print("\n--- 实验结果摘要 (多维度) ---")
    print(df_summary)

    # --- 4. 多图表可视化与保存 ---
    print("\n正在生成 2x2 组合图表...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(EXPERIMENT_NAME, fontsize=20, weight='bold')

    # --- 绘图函数，避免代码重复 ---
    def plot_metric(ax, x_values, y_values, y_std, title, ylabel):
        ax.errorbar(
            x_values, y_values, yerr=y_std,
            fmt='-o', capsize=5, elinewidth=1.5, markeredgewidth=1.5,
            label="HPVT" # 在这个实验中，我们只画一条线
        )
        ax.set_title(title, fontsize=14, weight='bold')
        ax.set_xlabel(r"Missing Rate ($p_m$)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(PM_RANGE)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- 依次绘制四个子图 ---
    plot_metric(axes[0, 0], df_summary['missing_rate'], df_summary['mean_time'], df_summary['std_time'], 
                'a) Time Efficiency', 'Total Execution Time (ms)')
    
    plot_metric(axes[0, 1], df_summary['missing_rate'], df_summary['mean_bits'], df_summary['std_bits'], 
                'b) Energy Consumption', 'Total Communication Bits')
    
    plot_metric(axes[1, 0], df_summary['missing_rate'], df_summary['mean_pruning'], df_summary['std_pruning'], 
                'c) Pruning Mechanism', 'Number of Pruning Events')
    
    plot_metric(axes[1, 1], df_summary['missing_rate'], df_summary['mean_aloha'], df_summary['std_aloha'],
                'd) ALOHA Mechanism', 'Number of ALOHA Frames Initiated')
    
    # 调整Y轴，使其对数显示，更能看出变化趋势
    axes[0, 1].set_yscale('log')
    axes[1, 1].set_yscale('log')


    # 在图表右上角注明关键环境参数
    algo_config = ALGORITHMS_TO_TEST[0]['config']
    env_params_text = (f"Environment:\n"
                       f"  N = {SCENARIO_CONFIG['TOTAL_TAGS']}\n"
                       f"  τ_A = {algo_config['aloha_threshold']}")
    fig.text(0.95, 0.97, env_params_text, ha='right', va='top', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_PLOT_FILE, dpi=300)
    print(f"2x2 组合图表已保存到: {OUTPUT_PLOT_FILE}")
    df_summary.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"聚合数据已保存到: {OUTPUT_CSV_FILE}")
    plt.show()

if __name__ == '__main__':
    main()
