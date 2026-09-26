# -*- coding: utf-8 -*-
"""
【多维度分析版】实验 1.2: 切换阈值 τA 对性能的全面影响

================================================================================
本实验的核心目标 (Objective):
1. 在之前的基础上，从多个维度深入分析 τA 的影响，包括：
   - a. 总执行时间 (时间效率)
   - b. 总通信比特数 (能量消耗)
   - c. 剪枝事件数 (核心机制效率)
   - d. 并行验证启动次数 (模式切换行为)
2. 通过一个 2x2 的组合图表，直观地展示这些指标如何相互关联，
   从而解释性能拐点出现的原因。
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

# 假设仿真框架和算法已存在
from framework import run_missing_tag_simulation
from hpvt_algo import HPVT_AggregatedAlgo

# ==============================================================================
# 1. 实验配置区 (与之前相同)
# ==============================================================================
print("正在加载实验 1.2 (多维度分析版) 的配置...")
EXPERIMENT_NAME = r"Experiment 1.2: Comprehensive Analysis of Threshold $\tau_A$"
OUTPUT_DIR, RESULTS_DIR = "plots", "results"
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_PLOT_FILE = os.path.join(OUTPUT_DIR, "fig1.2_multimetric_threshold_sensitivity.png")
# 【修改】移除了旧的单一CSV文件名，将会在下面为每个指标单独定义
# OUTPUT_CSV_FILE = os.path.join(RESULTS_DIR, "exp1.2_multimetric_threshold_sensitivity_summary.csv")

SCENARIO_CONFIG = {'TOTAL_TAGS': 2000, 'BINARY_LENGTH': 96}
PM_LEVELS = [0.2, 0.5, 0.8]
THRESHOLD_RANGE = [16, 32, 64, 128, 256, 512]
NUM_RUNS = 10
ALGORITHMS_TO_TEST = [{"class": HPVT_AggregatedAlgo, "name": "HPVT"}]

# ==============================================================================
# 2. 并行任务函数 (Worker Function for Parallel Execution)
# ==============================================================================
def run_single_task(task_params: tuple):
    """
    执行单次仿真任务，现在会返回更丰富的指标。
    """
    algo_info, pm, threshold, run_id, scenario_config_base = task_params
    current_scenario_config = scenario_config_base.copy()
    current_scenario_config['MISSING_RATE'] = pm
    current_algo_config = {"aloha_threshold": threshold, "binary_length": scenario_config_base['BINARY_LENGTH']}

    # 调用【已升级】的仿真框架，它会返回包含新指标的result
    result = run_missing_tag_simulation(
        scenario_config=current_scenario_config,
        algorithm_class=algo_info['class'],
        algorithm_specific_config=current_algo_config
    )

    # 【新】在记录中加入所有需要的指标
    record = {
        "algorithm": algo_info['name'],
        "missing_rate": pm,
        "aloha_threshold": threshold,
        "run_id": run_id,
        "total_time_ms": result.get('total_protocol_time_us', 0) / 1000.0,
        "total_bits": result.get('total_reader_bits', 0) + result.get('total_tag_bits', 0),
        "pruning_events": result.get('pruning_events', 0),
        "aloha_frames": result.get('aloha_frames_initiated', 0),
        "collision_slots": result.get('collision_slots', 0)
    }
    return record

# ==============================================================================
# 3. 主执行逻辑 (Main Execution Logic)
# ==============================================================================
def main():
    print("正在准备并行仿真任务 (多维度)...")
    tasks_to_run = list(product(ALGORITHMS_TO_TEST, PM_LEVELS, THRESHOLD_RANGE, range(NUM_RUNS)))
    tasks_with_config = [(task[0], task[1], task[2], task[3], SCENARIO_CONFIG) for task in tasks_to_run]
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
    
    # 【新】聚合所有需要的指标
    df_summary = df_raw.groupby(['missing_rate', 'aloha_threshold']).agg(
        mean_time=('total_time_ms', 'mean'),
        std_time=('total_time_ms', 'std'),
        mean_bits=('total_bits', 'mean'),
        std_bits=('total_bits', 'std'),
        mean_pruning=('pruning_events', 'mean'),
        std_pruning=('pruning_events', 'std'),
        mean_aloha=('aloha_frames', 'mean'),
        std_aloha=('aloha_frames', 'std')
    ).reset_index()

    print("\n--- 实验结果摘要 (长格式) ---")
    print(df_summary)

    # ==========================================================
    # 4. 【新】多图表可视化与【新】独立CSV文件保存
    # ==========================================================

    # --- 数据转换与独立保存函数 ---
    def save_metric_to_csv(metric_name, mean_col, std_col):
        """
        将指定指标的数据转换为宽格式并保存到独立的CSV文件。
        宽格式：index是threshold, columns是不同的pm下的均值和标准差。
        """
        # --- 提取均值数据并透视 ---
        df_mean = df_summary.pivot(
            index='aloha_threshold',
            columns='missing_rate',
            values=mean_col
        )
        # --- 提取标准差数据并透视 ---
        df_std = df_summary.pivot(
            index='aloha_threshold',
            columns='missing_rate',
            values=std_col
        )
        # --- 重命名列名以区分均值和标准差 ---
        df_mean.columns = [f"pm_{col}_mean" for col in df_mean.columns]
        df_std.columns = [f"pm_{col}_std" for col in df_std.columns]
        
        # --- 合并均值和标准差为一个DataFrame ---
        df_plot_friendly = pd.concat([df_mean, df_std], axis=1)
        
        # --- 定义文件名并保存 ---
        file_path = os.path.join(RESULTS_DIR, f"exp1.2_{metric_name}.csv")
        df_plot_friendly.to_csv(file_path)
        print(f"绘图友好的数据已保存到: {file_path}")

    # --- 为四个指标分别调用保存函数 ---
    print("\n正在转换数据格式并分别保存到CSV文件...")
    save_metric_to_csv('time_efficiency', 'mean_time', 'std_time')
    save_metric_to_csv('energy_consumption', 'mean_bits', 'std_bits')
    save_metric_to_csv('pruning_mechanism', 'mean_pruning', 'std_pruning')
    save_metric_to_csv('aloha_mechanism', 'mean_aloha', 'std_aloha')

    print("\n正在生成 2x2 组合图表...")
    plt.style.use('seaborn-v0_8-whitegrid')
    # 创建一个 2x2 的子图网格
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(EXPERIMENT_NAME, fontsize=20, weight='bold')
    
    markers = ['o', 's', '^'] # 为不同曲线设置不同标记

    # --- 绘图函数，避免代码重复 (此部分无需修改) ---
    def plot_metric(ax, metric_mean, metric_std, title, ylabel):
        for i, pm in enumerate(PM_LEVELS):
            group = df_summary[df_summary['missing_rate'] == pm]
            ax.errorbar(
                group['aloha_threshold'], group[metric_mean], yerr=group[metric_std],
                label=f"$p_m = {pm}$", fmt=f'-{markers[i]}', capsize=5,
                elinewidth=1.5, markeredgewidth=1.5
            )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(r"ALOHA Threshold ($\tau_A$)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xscale('log', base=2)
        ax.set_xticks(THRESHOLD_RANGE)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- 依次绘制四个子图 ---
    plot_metric(axes[0, 0], 'mean_time', 'std_time', 'a) Time Efficiency', 'Total Execution Time (ms)')
    plot_metric(axes[0, 1], 'mean_bits', 'std_bits', 'b) Energy Consumption', 'Total Communication Bits')
    plot_metric(axes[1, 0], 'mean_pruning', 'std_pruning', 'c) Pruning Mechanism', 'Number of Pruning Events')
    plot_metric(axes[1, 1], 'mean_aloha', 'std_aloha', 'd) ALOHA Mechanism', 'Number of ALOHA Frames Initiated')

    # 在图表右上角注明关键环境参数
    env_params_text = f"Environment: N = {SCENARIO_CONFIG['TOTAL_TAGS']}"
    fig.text(0.95, 0.97, env_params_text, ha='right', va='top', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    # 优化整体布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存与显示
    plt.savefig(OUTPUT_PLOT_FILE, dpi=300)
    print(f"\n2x2 组合图表已保存到: {OUTPUT_PLOT_FILE}")
    plt.show()

if __name__ == '__main__':
    main()
