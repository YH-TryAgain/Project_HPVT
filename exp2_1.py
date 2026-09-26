# -*- coding: utf-8 -*-
"""
【最终版】实验 2.1: 性能与标签丢失率 (pm) 的关系

================================================================================
本实验的核心目标 (Objective):
1.  固定标签总数(N)，通过改变标签丢失率(pm)，全面评估各协议的性能表现。
2.  对比的指标包括 (时间单位统一为 µs)：
    - a) 总执行时间 vs. 丢失率
    - b) 总通信比特数 vs. 丢失率
    - c) 首次发现所需时间 vs. 丢失率
    - d) 【特殊图】发现过程时间线 (在高丢失率场景下，如 pm=0.8)

本脚本同样采用多进程并行计算 (multiprocessing) 来加速仿真。
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

# 导入您的仿真框架和所有需要对比的算法
# 请确保这些模块在您的项目中路径正确
from framework import run_missing_tag_simulation
from hpvt_algo import HPVT_AggregatedAlgo
from cpt_algo import CPT_Algo
from iip_algo import IIP_Algo
from crmti_algo import CR_MTI_Algo
from ctmti_algo import CTMTIAlgo
from ecumi_algo import CUMIAlgo, ECUMIAlgo

# ==============================================================================
# 1. 实验配置区 (Configuration Area)
# ==============================================================================
print("正在加载实验 2.1 (性能与丢失率关系) 的配置...")

EXPERIMENT_NAME = "Experiment 2.1: Performance Analysis vs. Missing Rate (pm)"
OUTPUT_DIR, RESULTS_DIR = "plots", "results"
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_PLOT_FILE = os.path.join(OUTPUT_DIR, "fig2.1_multimetric_missing_rate_analysis.png")
# --- 总览摘要文件的名称保持不变 ---
OVERALL_SUMMARY_CSV_FILE = os.path.join(RESULTS_DIR, "exp2.1_multimetric_missing_rate_analysis_summary.csv")

# --- 仿真固定参数 ---
N_FIXED = 2000  # 固定标签总数，与论文图7保持一致
TIMELINE_PLOT_PM = 0.8 # 为发现过程时间线图指定一个固定的高丢失率

# --- 仿真可变参数 ---
PM_RANGE = np.linspace(0.1, 0.9, 9) # 丢失率从10%到90%
NUM_RUNS = 10 # 为保证统计显著性，每个配置点运行10次

# ======================================================================
# 所有算法的“注册中心” (与 exp2.2 保持一致)
# ======================================================================
ALGORITHM_BASE_CONFIG = {
    "HPVT (Ours)": {
        "class": HPVT_AggregatedAlgo, 
        "config": {"aloha_threshold": 128, "binary_length": 96}
    },
    "CPT": {
        "class": CPT_Algo, 
        "config": {}
    },
    "IIP": {
        "class": IIP_Algo, 
        "config": {}
    },
    "CR_MTI": {
        "class": CR_MTI_Algo, 
        "config": {"lambda_opt": 15.0, "w": 34}
    },
    "CTMTI": {
        "class": CTMTIAlgo, 
        "config": {"alpha": 0.54, "B": 2}
    }, 
    "CUMI": {
        "class": CUMIAlgo, 
        "config": {'x': 3}
    },      
    "ECUMI": {
        "class": ECUMIAlgo, 
        "config": {'x_prime': 10}
    },
}

# ======================================================================
# 选择本次实验要运行的算法
# ======================================================================
ALGORITHMS_TO_TEST = [
    "HPVT (Ours)",
    "CPT",
    "IIP",
    "CR_MTI",
    "CTMTI",
    "CUMI",
    "ECUMI"
]

# ==============================================================================
# 2. 并行任务函数 (Worker Function for Parallel Execution)
# ==============================================================================
def run_single_task(task_params: tuple):
    """ 
    执行单次仿真任务并返回所有需要的指标。
    【新增】为了方便后续处理，将发现过程时间线(list)展开为独立的列。
    """
    algo_info, n_tags, pm, run_id = task_params
    
    current_scenario_config = {'TOTAL_TAGS': int(n_tags), 'BINARY_LENGTH': 96, 'MISSING_RATE': pm}
    
    current_algo_config = algo_info['config'].copy()
    if 'binary_length' in current_algo_config:
        current_algo_config['binary_length'] = current_scenario_config['BINARY_LENGTH']

    result = run_missing_tag_simulation(
        scenario_config=current_scenario_config,
        algorithm_class=algo_info['class'],
        algorithm_specific_config=current_algo_config
    )

    record = {
        "algorithm": algo_info['name'],
        "total_tags": n_tags,
        "missing_rate": pm,
        "run_id": run_id,
        "total_time_us": result.get('total_protocol_time_us', 0),
        "total_bits": result.get('total_reader_bits', 0) + result.get('total_tag_bits', 0),
        "time_to_first_us": result.get('time_to_first_missing_us', -1.0),
    }

    # 【关键修改】将时间线列表展开，便于后续聚合
    timeline = result.get('discovery_timeline_us', [-1.0] * 11)
    for i, p in enumerate(np.linspace(0, 100, 11, dtype=int)):
        record[f'time_to_{p}p_us'] = timeline[i]
        
    return record

# ==============================================================================
# 3. 主执行逻辑 (Main Execution Logic)
# ==============================================================================
def main():
    """ 主函数，负责执行整个实验 """
    print("正在准备并行仿真任务 (性能与丢失率关系)...")
    
    tasks_to_run_params = list(product(ALGORITHMS_TO_TEST, [N_FIXED], PM_RANGE, range(NUM_RUNS)))
    
    tasks_with_config = []
    for algo_name, n_tags, pm, run_id in tasks_to_run_params:
        if algo_name not in ALGORITHM_BASE_CONFIG:
            print(f"警告: 在 ALGORITHM_BASE_CONFIG 中找不到算法 '{algo_name}'，已跳过。")
            continue
        
        algo_info = ALGORITHM_BASE_CONFIG[algo_name].copy()
        algo_info['name'] = algo_name
        tasks_with_config.append((algo_info, n_tags, pm, run_id))

    total_simulations = len(tasks_with_config)
    print(f"任务总数: {total_simulations}")

    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"将使用 {num_processes} 个CPU核心并行执行...")

    results_list = []; start_time = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(run_single_task, tasks_with_config)
        for result in tqdm(results_iterator, total=total_simulations, desc="执行丢失率影响仿真"):
            results_list.append(result)
    end_time = time.time()
    print(f"\n并行实验执行完毕。总耗时: {end_time - start_time:.2f} 秒")

    print("正在处理和分析数据...")
    df_raw = pd.DataFrame(results_list)
    
    # --- 【核心修改】为所有指标（包括时间线）定义聚合规则 ---
    timeline_cols = [f'time_to_{p}p_us' for p in np.linspace(0, 100, 11, dtype=int)]
    agg_dict = {
        'total_time_us': ['mean', 'std'],
        'total_bits': ['mean', 'std'],
        'time_to_first_us': ['mean', 'std'],
    }
    for col in timeline_cols:
        agg_dict[col] = ['mean', 'std']

    df_summary = df_raw.groupby(['algorithm', 'missing_rate']).agg(agg_dict).reset_index()
    # 扁平化多级列索引
    df_summary.columns = ['_'.join(col).strip('_') for col in df_summary.columns.values]
    
    cleaned_algo_list = [name.replace(' (Ours)', '') for name in ALGORITHMS_TO_TEST]
    df_summary['algorithm'] = df_summary['algorithm'].str.replace(' (Ours)', '', regex=False)

    print("\n--- 实验结果摘要 ---")
    print(df_summary)
    df_summary.to_csv(OVERALL_SUMMARY_CSV_FILE, index=False)
    print(f"聚合数据总览已保存到: {OVERALL_SUMMARY_CSV_FILE}")

    # ==============================================================================
    # 4. 【新增功能】将各指标数据分别保存到独立的、便于绘图的CSV文件中
    # ==============================================================================
    print("\n正在导出各指标的详细对比数据到独立的CSV文件...")

    # --- 导出前三个指标 (vs. Missing Rate) ---
    metrics_to_export = {
        "total_time": "total_time_us",
        "total_bits": "total_bits",
        "time_to_first": "time_to_first_us",
    }

    for metric_name, base_col_name in metrics_to_export.items():
        # 提取均值和标准差数据
        df_mean = df_summary.pivot(index='missing_rate', columns='algorithm', values=f'{base_col_name}_mean')
        df_std = df_summary.pivot(index='missing_rate', columns='algorithm', values=f'{base_col_name}_std')
        
        # 为标准差列添加_std后缀
        df_std = df_std.rename(columns=lambda x: f"{x}_std")
        
        # 合并均值和标准差
        df_final = pd.concat([df_mean, df_std], axis=1)
        
        # 重新排序，使每个算法的均值和标准差相邻
        final_cols_order = []
        for algo in cleaned_algo_list:
            final_cols_order.append(algo)
            final_cols_order.append(f"{algo}_std")
        df_final = df_final[final_cols_order]
        
        # 保存到CSV
        output_filename = os.path.join(RESULTS_DIR, f"exp2.1_data_{metric_name}.csv")
        df_final.to_csv(output_filename)
        print(f" -> 已成功保存 {metric_name} 数据到: {output_filename}")

    # --- 导出第四个指标 (Discovery Timeline) ---
    df_timeline_filtered = df_summary[df_summary['missing_rate'] == TIMELINE_PLOT_PM].set_index('algorithm')
    
    discovery_percentages = np.linspace(0, 100, 11, dtype=int)
    timeline_data_for_csv = {'Discovery_Percent': discovery_percentages}
    
    for algo_name in cleaned_algo_list:
        mean_values = [df_timeline_filtered.loc[algo_name, f'time_to_{p}p_us_mean'] for p in discovery_percentages]
        std_values = [df_timeline_filtered.loc[algo_name, f'time_to_{p}p_us_std'] for p in discovery_percentages]
        timeline_data_for_csv[algo_name] = mean_values
        timeline_data_for_csv[f"{algo_name}_std"] = std_values
        
    df_final_timeline = pd.DataFrame(timeline_data_for_csv)
    output_filename_timeline = os.path.join(RESULTS_DIR, "exp2.1_data_discovery_timeline.csv")
    df_final_timeline.to_csv(output_filename_timeline, index=False)
    print(f" -> 已成功保存 discovery_timeline 数据到: {output_filename_timeline}")


    # =====================================================================
    # 5. 可视化 (Visualization) - 此部分保持不变
    # =====================================================================
    print("\n正在生成 2x2 性能与丢失率对比图表...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.suptitle(f"{EXPERIMENT_NAME} (N={N_FIXED})", fontsize=22, weight='bold')

    # --- 自动样式配置 ---
    style_config = {}
    sota_algorithms = [name for name in cleaned_algo_list if "HPVT" not in name]
    sota_markers = ['^', 's', 'D', 'v', 'p', '*']
    sota_colors = plt.get_cmap('tab10').colors
    
    for i, algo_name in enumerate(sota_algorithms):
        style_config[algo_name] = {
            'color': sota_colors[i % len(sota_colors)],
            'marker': sota_markers[i % len(sota_markers)],
            'zorder': 5
        }
    style_config["HPVT"] = {'color': 'red', 'marker': 'o', 'zorder': 10}

    # --- 绘图函数 (绘制前3个图) ---
    def plot_metric_vs_pm(ax, y_col, y_std, title, ylabel):
        ax.set_title(title, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xlabel('Missing Rate (pm)', fontsize=14)
        
        for algo_name in cleaned_algo_list:
            df_algo = df_summary[df_summary['algorithm'] == algo_name]
            if df_algo.empty: continue
            
            style = style_config[algo_name]
            ax.errorbar(df_algo['missing_rate'], df_algo[f'{y_col}_mean'], yerr=df_algo[f'{y_col}_std'],
                        label=algo_name, marker=style['marker'], color=style['color'],
                        capsize=4, zorder=style['zorder'])
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- 绘制发现过程时间线图 (第4个图) ---
    def plot_discovery_timeline(ax, title, ylabel):
        ax.set_title(title, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xlabel('Discovery Percent (%)', fontsize=14)
        
        df_plot = pd.read_csv(output_filename_timeline)

        for algo_name in cleaned_algo_list:
            style = style_config[algo_name]
            ax.plot(df_plot['Discovery_Percent'], df_plot[algo_name],
                    label=algo_name, marker=style['marker'], color=style['color'],
                    zorder=style['zorder'])
                    
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- 依次绘制四个子图 ---
    plot_metric_vs_pm(axes[0, 0], 'total_time_us', 'total_time_us_std', 'a) Total Execution Time vs. Missing Rate', 'Time (µs)')
    plot_metric_vs_pm(axes[0, 1], 'total_bits', 'total_bits_std', 'b) Total Transmitted Bits vs. Missing Rate', 'Total Bits')
    plot_metric_vs_pm(axes[1, 0], 'time_to_first_us', 'time_to_first_us_std', 'c) Time to First Missing vs. Missing Rate', 'Time (µs)')
    plot_discovery_timeline(axes[1, 1], f'd) Discovery Timeline (pm={TIMELINE_PLOT_PM})', 'Time (µs)')
    
    for row in axes:
        for ax in row:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(OUTPUT_PLOT_FILE, dpi=300, bbox_inches='tight')
    print(f"性能与丢失率对比图表已保存到: {OUTPUT_PLOT_FILE}")
    plt.show()

if __name__ == '__main__':
    main()
