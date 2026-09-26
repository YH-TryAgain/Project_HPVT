# -*- coding: utf-8 -*-
"""
【最终版】实验 2.3: 鲁棒性测试 — 性能与ID分布的关系

================================================================================
本实验的核心目标 (Objective):
1.  评估 HPVT 和 SOTA 协议在处理不同类型ID分布时的性能鲁棒性。
2.  【核心修改】对比三种分布场景：
    - Uniform: 完全随机的ID。
    - Prefix-Clustered: 前48位相同的ID，模拟同一批次产品。
    - Suffix-Clustered: 后48位相同的ID，作为一种极限压力测试。
3.  使用分组柱状图 (Grouped Bar Chart) 进行可视化对比。

本脚本采用多进程并行计算 (multiprocessing) 来加速仿真过程。
================================================================================
"""

import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import multiprocessing
from itertools import product
from tqdm import tqdm

# 导入您的仿真框架和所有需要对比的算法
from framework import run_missing_tag_simulation, Tag
from hpvt_algo import HPVT_AggregatedAlgo
from cpt_algo import CPT_Algo
from iip_algo import IIP_Algo
from crmti_algo import CR_MTI_Algo
from ctmti_algo import CTMTIAlgo
from ecumi_algo import CUMIAlgo, ECUMIAlgo


# ==============================================================================
# 1. 实验配置区 (Configuration Area)
# ==============================================================================
print("正在加载实验 2.3 (鲁棒性测试 - 新分布) 的配置...")

EXPERIMENT_NAME = "Experiment 2.3: Robustness Analysis vs. ID Distribution Type"
OUTPUT_DIR, RESULTS_DIR = "plots", "results"
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_PLOT_FILE = os.path.join(OUTPUT_DIR, "fig2.3_robustness_analysis_new_dists.png")
OVERALL_SUMMARY_CSV_FILE = os.path.join(RESULTS_DIR, "exp2.3_robustness_analysis_new_dists_summary.csv")

# --- 仿真固定参数 ---
N_FIXED = 3000
PM_FIXED = 0.8
# --- 核心修改: 更新为三种新的ID分布类型 (修正了拼写错误) ---
ID_DISTRIBUTIONS = ['Uniform', 'Prefix-Clustered', 'Suffix-Clustered']
NUM_RUNS = 10
BINARY_LENGTH = 96

# ======================================================================
# 所有算法的“注册中心” (保持不变)
# ======================================================================
ALGORITHM_BASE_CONFIG = {
    "HPVT (Ours)": {
        "class": HPVT_AggregatedAlgo, 
        "config": {"aloha_threshold": 128, "binary_length": 96}
    },
    "CPT": {"class": CPT_Algo, "config": {}},
    "IIP": {"class": IIP_Algo, "config": {}},
    "CR_MTI": {"class": CR_MTI_Algo, "config": {"lambda_opt": 15.0, "w": 34}},
    "CTMTI": {"class": CTMTIAlgo, "config": {"alpha": 0.54, "B": 2}}, 
    "CUMI": {"class": CUMIAlgo, "config": {'x': 3}},      
    "ECUMI": {"class": ECUMIAlgo, "config": {'x_prime': 10}},
}

# ======================================================================
# 选择本次实验要运行的算法
# ======================================================================
ALGORITHMS_TO_TEST = [
    "HPVT (Ours)", "CPT", "IIP", "CR_MTI", "CTMTI", "CUMI", "ECUMI"
]


# ==============================================================================
# 2. 【核心修改】特定场景生成函数
# ==============================================================================
def generate_robustness_scenario(total_tags, binary_length, distribution='uniform'):
    """
    根据指定的分布类型生成一组唯一的标签ID。
    - uniform: 完全随机的ID。
    - prefix-clustered: 前48位相同，后48位随机。
    - suffix-clustered: 前48位随机，后48位相同。
    """
    id_set = set()
    if binary_length != 96:
        raise ValueError("此场景生成器当前仅支持96位ID长度。")
    
    half_len = binary_length // 2

    if distribution.lower() == 'prefix-clustered':
        # 前缀聚集: 前48位相同
        common_prefix = ''.join(random.choice('01') for _ in range(half_len))
        while len(id_set) < total_tags:
            random_suffix = ''.join(random.choice('01') for _ in range(half_len))
            id_set.add(common_prefix + random_suffix)
            
    elif distribution.lower() == 'suffix-clustered':
        # 后缀聚集: 后48位相同
        common_suffix = ''.join(random.choice('01') for _ in range(half_len))
        while len(id_set) < total_tags:
            random_prefix = ''.join(random.choice('01') for _ in range(half_len))
            id_set.add(random_prefix + common_suffix)
            
    else:  # 默认为 Uniform
        while len(id_set) < total_tags:
            id_set.add(''.join(random.choice('01') for _ in range(binary_length)))
            
    return id_set


# ==============================================================================
# 3. 并行任务函数 (Worker Function for Parallel Execution) - 保持不变
# ==============================================================================
def run_single_task(task_params: tuple):
    """ 执行单次仿真任务并返回所有四个核心指标 """
    algo_info, dist_type, run_id = task_params
    
    all_ids = list(generate_robustness_scenario(N_FIXED, BINARY_LENGTH, dist_type))
    random.shuffle(all_ids)
    num_missing = int(N_FIXED * PM_FIXED)
    scenario_tags = [Tag(all_ids[i], is_present=(i < N_FIXED - num_missing)) for i in range(N_FIXED)]
    
    current_scenario_config = {
        'TOTAL_TAGS': N_FIXED,
        'BINARY_LENGTH': BINARY_LENGTH,
        'MISSING_RATE': PM_FIXED,
        'PREGENERATED_TAGS': scenario_tags 
    }
    
    result = run_missing_tag_simulation(
        scenario_config=current_scenario_config,
        algorithm_class=algo_info['class'],
        algorithm_specific_config=algo_info['config']
    )

    timeline = result.get('discovery_timeline_us', [])
    time_to_90_percent = timeline[9] if len(timeline) > 9 else -1.0

    record = {
        "algorithm": algo_info['name'],
        "distribution": dist_type,
        "run_id": run_id,
        "total_time_us": result.get('total_protocol_time_us', 0),
        "total_bits": result.get('total_reader_bits', 0) + result.get('total_tag_bits', 0),
        "time_to_first_us": result.get('time_to_first_missing_us', -1.0),
        "time_to_90p_us": time_to_90_percent,
    }
    return record


# ==============================================================================
# 4. 主执行逻辑 (Main Execution Logic)
# ==============================================================================
def main():
    """ 主函数，负责执行整个实验 """
    print(f"正在准备并行仿真任务 ({EXPERIMENT_NAME})...")
    
    tasks_to_run_params = list(product(ALGORITHMS_TO_TEST, ID_DISTRIBUTIONS, range(NUM_RUNS)))
    
    tasks_with_config = []
    for algo_name, dist_type, run_id in tasks_to_run_params:
        algo_info = ALGORITHM_BASE_CONFIG[algo_name].copy()
        algo_info['name'] = algo_name
        tasks_with_config.append((algo_info, dist_type, run_id))

    total_simulations = len(tasks_with_config)
    print(f"任务总数: {total_simulations}")

    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"将使用 {num_processes} 个CPU核心并行执行...")

    results_list = []; start_time = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(run_single_task, tasks_with_config)
        for result in tqdm(results_iterator, total=total_simulations, desc="执行鲁棒性仿真"):
            results_list.append(result)
    end_time = time.time()
    print(f"\n并行实验执行完毕。总耗时: {end_time - start_time:.2f} 秒")

    print("正在处理和分析数据...")
    df_raw = pd.DataFrame(results_list)
    
    # --- 核心修改: 数据聚合时不再计算标准差 ---
    df_summary = df_raw.groupby(['algorithm', 'distribution']).agg(
        total_time_us=('total_time_us', 'mean'),
        total_bits=('total_bits', 'mean'),
        time_to_first_us=('time_to_first_us', 'mean'),
        time_to_90p_us=('time_to_90p_us', 'mean'),
    ).reset_index()
    
    # 清理算法名称
    df_summary['algorithm'] = df_summary['algorithm'].str.replace(' (Ours)', '', regex=False)
    cleaned_algo_list = [name.replace(' (Ours)', '') for name in ALGORITHMS_TO_TEST]

    print("\n--- 实验结果摘要 (鲁棒性测试) ---")
    print(df_summary)
    df_summary.to_csv(OVERALL_SUMMARY_CSV_FILE, index=False)
    print(f"聚合数据总览已保存到: {OVERALL_SUMMARY_CSV_FILE}")
    
    # =====================================================================
    # 5. 【核心修改】将各指标数据分别保存到独立的、便于绘图的CSV文件中
    # =====================================================================
    print("\n正在导出各指标的详细对比数据到独立的CSV文件...")

    metrics_to_export = {
        "total_time": "total_time_us",
        "total_bits": "total_bits",
        "time_to_first_missing": "time_to_first_us",
        "time_to_90_completion": "time_to_90p_us",
    }

    for metric_name, value_col in metrics_to_export.items():
        # 使用 pivot 将数据重塑为“宽格式”
        df_pivot = df_summary.pivot(
            index='algorithm',
            columns='distribution',
            values=value_col
        )
        
        # 重新排序列和行，以确保与绘图顺序一致
        df_pivot = df_pivot.reindex(columns=ID_DISTRIBUTIONS)
        df_pivot = df_pivot.reindex(cleaned_algo_list)
        
        # 保存到CSV
        output_filename = os.path.join(RESULTS_DIR, f"exp2.3_data_{metric_name}.csv")
        df_pivot.to_csv(output_filename)
        print(f" -> 已成功保存 {metric_name} 数据到: {output_filename}")


    # =====================================================================
    # 6. 【可视化重构】生成2x2的分组柱状图 (Grouped Bar Chart)
    # =====================================================================
    print("\n正在生成鲁棒性对比分组柱状图...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    fig.suptitle(f"{EXPERIMENT_NAME}\n(N = {N_FIXED}, pm = {PM_FIXED})", fontsize=22, weight='bold')

    # --- 定义样式 ---
    style_config = {}
    sota_algorithms = [name for name in cleaned_algo_list if "HPVT" not in name]
    sota_colors = plt.get_cmap('viridis_r')(np.linspace(0.1, 0.8, len(sota_algorithms)))
    for i, algo_name in enumerate(sota_algorithms):
        style_config[algo_name] = sota_colors[i]
    style_config["HPVT"] = 'red'

    # --- 分组柱状图绘制函数 ---
    def plot_grouped_bar(ax, y_col, title):
        ax.set_title(title, fontsize=16)
        
        # 从刚刚生成的独立CSV文件中读取数据，确保数据源一致
        metric_name_map = {
            "total_time_us": "total_time",
            "total_bits": "total_bits",
            "time_to_first_us": "time_to_first_missing",
            "time_to_90p_us": "time_to_90_completion",
        }
        csv_path = os.path.join(RESULTS_DIR, f"exp2.3_data_{metric_name_map[y_col]}.csv")
        df_plot = pd.read_csv(csv_path, index_col='algorithm')

        algorithms = df_plot.index.tolist()
        distributions = df_plot.columns.tolist()
        n_algos = len(algorithms)
        
        bar_width = 0.25 # 减小柱子宽度以容纳三组
        index = np.arange(n_algos)
        hatches = ['', '///', 'xxx'] # 为三种分布定义不同的填充样式

        for i, dist in enumerate(distributions):
            means = df_plot[dist]
            
            # 重新计算位置以使三组柱子居中
            pos = index - bar_width + i * bar_width
            colors = [style_config.get(algo, 'gray') for algo in algorithms]
            
            ax.bar(pos, means, bar_width, label=dist, capsize=4,
                   color=colors, hatch=hatches[i], alpha=0.8, edgecolor='black')

        ax.set_xticks(index)
        ax.set_xticklabels(algorithms, rotation=20, ha="right", fontsize=12)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
        ax.legend(title='ID Distribution', fontsize=12)

    # --- 依次绘制四个子图 ---
    axes[0, 0].set_ylabel('Time (µs)', fontsize=14)
    plot_grouped_bar(axes[0, 0], 'total_time_us', 'a) Total Execution Time')
    
    axes[0, 1].set_ylabel('Total Bits', fontsize=14)
    plot_grouped_bar(axes[0, 1], 'total_bits', 'b) Total Transmitted Bits')

    axes[1, 0].set_ylabel('Time (µs)', fontsize=14)
    plot_grouped_bar(axes[1, 0], 'time_to_first_us', 'c) Time to First Missing')

    axes[1, 1].set_ylabel('Time (µs)', fontsize=14)
    plot_grouped_bar(axes[1, 1], 'time_to_90p_us', 'd) Time to 90% Completion')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(OUTPUT_PLOT_FILE, dpi=300, bbox_inches='tight')
    print(f"鲁棒性对比图表已保存到: {OUTPUT_PLOT_FILE}")
    plt.show()

if __name__ == '__main__':
    main()
