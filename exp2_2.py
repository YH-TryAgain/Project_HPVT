# -*- coding: utf-8 -*-
"""
【最终版】实验 2.2: 可扩展性测试 — 性能与标签总数 (N) 的关系

================================================================================
本实验的核心目标 (Objective):
1. 使用四个核心通用指标，全面评估 HPVT 和 SOTA 协议的可扩展性。
2. 对比的指标包括 (时间单位统一为 µs)：
    - a) 总执行时间 (时间效率)
    - b) 总能耗 (以总通信比特数为代理)
    - c) 首次发现所需时间 (响应敏捷性)
    - d) 完成90%识别所需时间 (过程质量与收尾效率)
3. 采用更模块化的配置方式，并自动配置绘图样式以突出核心算法。

本脚本采用多进程并行计算 (multiprocessing) 来加速仿真过程。
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
print("正在加载实验 2.2 (可视化优化版) 的配置...")

EXPERIMENT_NAME = "Experiment 2.2: Comprehensive Scalability Analysis vs. Tag Population (N)"
OUTPUT_DIR, RESULTS_DIR = "plots", "results"
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_PLOT_FILE = os.path.join(OUTPUT_DIR, "fig2.2_multimetric_scalability_analysis.png")
OUTPUT_CSV_FILE = os.path.join(RESULTS_DIR, "exp2.2_multimetric_scalability_analysis_summary.csv")

# --- 仿真固定参数 ---
PM_LEVELS = [0.8]  # 低缺失率和高缺失率两种场景

# --- 仿真可变参数 ---
N_RANGE = np.linspace(300, 3000, 10, dtype=int)
NUM_RUNS = 10 

# ======================================================================
# 所有算法的“注册中心”
# ======================================================================
ALGORITHM_BASE_CONFIG = {
    "HPVT (Ours)": {
        "class": HPVT_AggregatedAlgo, 
        "config": {"aloha_threshold": 64, "binary_length": 96}
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
# 选择本次实验要运行的算法，只需填入名字
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
    """ 执行单次仿真任务并返回所有需要的指标 """
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

    timeline = result.get('discovery_timeline_us', [])
    time_to_90_percent = timeline[9] if len(timeline) > 9 else -1.0

    record = {
        "algorithm": algo_info['name'],
        "total_tags": n_tags,
        "missing_rate": pm,
        "run_id": run_id,
        "total_time_us": result.get('total_protocol_time_us', 0),
        "total_bits": result.get('total_reader_bits', 0) + result.get('total_tag_bits', 0),
        "time_to_first_us": result.get('time_to_first_missing_us', -1.0),
        "time_to_90p_us": time_to_90_percent,
    }
    return record

# ==============================================================================
# 3. 主执行逻辑 (Main Execution Logic)
# ==============================================================================
def main():
    """ 主函数，负责执行整个实验 """
    print("正在准备并行仿真任务 (可扩展性测试)...")
    
    tasks_to_run_params = list(product(ALGORITHMS_TO_TEST, N_RANGE, PM_LEVELS, range(NUM_RUNS)))
    
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
        for result in tqdm(results_iterator, total=total_simulations, desc="执行可扩展性仿真"):
            results_list.append(result)
    end_time = time.time()
    print(f"\n并行实验执行完毕。总耗时: {end_time - start_time:.2f} 秒")

    print("正在处理和分析数据...")
    df_raw = pd.DataFrame(results_list)
    df_summary = df_raw.groupby(['algorithm', 'missing_rate', 'total_tags']).agg(
        mean_time=('total_time_us', 'mean'), std_time=('total_time_us', 'std'),
        mean_bits=('total_bits', 'mean'), std_bits=('total_bits', 'std'),
        mean_first_time=('time_to_first_us', 'mean'), std_first_time=('time_to_first_us', 'std'),
        mean_90p_time=('time_to_90p_us', 'mean'), std_90p_time=('time_to_90p_us', 'std'),
    ).reset_index()

    # 清理算法名称，移除 '(Ours)'
    df_summary['algorithm'] = df_summary['algorithm'].str.replace(' (Ours)', '', regex=False)
    # 更新 ALGORITHMS_TO_TEST 列表以匹配新的名称，用于绘图
    cleaned_algo_list = [name.replace(' (Ours)', '') for name in ALGORITHMS_TO_TEST]


    print("\n--- 实验结果摘要 ---")
    print(df_summary)

    # ==============================================================================
    # 4. 【格式优化】将各指标数据分别保存到更直观的CSV文件中
    # ==============================================================================
    print("\n正在导出各指标的详细对比数据到CSV文件 (已优化格式)...")
    
    # 定义要保存的指标和对应的列名
    metrics_to_save = {
        "total_time": ("mean_time", "std_time"),
        "energy_consumption": ("mean_bits", "std_bits"),
        "time_to_first_missing": ("mean_first_time", "std_first_time"),
        "time_to_90_percent": ("mean_90p_time", "std_90p_time"),
    }

    for metric_name, (mean_col, std_col) in metrics_to_save.items():
        # 创建一个以 total_tags 为索引的空 DataFrame，用于存放最终结果
        df_metric_export = pd.DataFrame(index=N_RANGE)
        df_metric_export.index.name = 'total_tags'

        # 遍历每个算法和每个缺失率，构建新的列
        for algo_name in cleaned_algo_list:
            for pm in PM_LEVELS:
                # 筛选出当前算法和缺失率对应的数据
                df_scenario = df_summary[(df_summary['algorithm'] == algo_name) & (df_summary['missing_rate'] == pm)]
                
                # 定义新的、更具描述性的列名
                # mean_col_name = f"{algo_name}(pm={pm})" 
                # std_col_name = f"{algo_name}_std(pm={pm})"
                # 定义新的、更具描述性的列名
                mean_col_name = f"{algo_name}" 
                std_col_name = f"{algo_name}_std"
                
                # 准备要合并的数据，只保留均值和标准差，并以 total_tags 为索引
                data_to_merge = df_scenario[['total_tags', mean_col, std_col]].set_index('total_tags')
                data_to_merge = data_to_merge.rename(columns={mean_col: mean_col_name, std_col: std_col_name})
                
                # 将处理好的数据合并到最终的输出 DataFrame 中
                df_metric_export = df_metric_export.join(data_to_merge)
        
        # 按列名排序，使得同一个算法的不同场景数据排在一起
        df_metric_export = df_metric_export.sort_index(axis=1)

        # 将结果保存到CSV文件
        output_filename = os.path.join(RESULTS_DIR, f"exp2.2_data_{metric_name}.csv")
        df_metric_export.to_csv(output_filename)
        print(f" -> 已成功保存 {metric_name} 数据到: {output_filename}")


    # =====================================================================
    # 5. 【可视化重构】多维度、多场景、自动样式配置的可视化
    # =====================================================================
    print("\n正在生成 2x2 可扩展性对比图表...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.suptitle(EXPERIMENT_NAME, fontsize=22, weight='bold')

    # --- 【核心修改】自动生成并分配固定的绘图样式 ---
    style_config = {}
    sota_algorithms = [name for name in cleaned_algo_list if "HPVT" not in name]
    sota_markers = ['^', 's', 'D', 'v', 'p', '*']
    # 使用 'tab10' colormap 获得一组清晰的颜色
    sota_colors = plt.get_cmap('tab10').colors
    
    for i, algo_name in enumerate(sota_algorithms):
        style_config[algo_name] = {
            'color': sota_colors[i % len(sota_colors)],
            'marker': sota_markers[i % len(sota_markers)],
            'zorder': 5
        }
    
    # 强制指定 HPVT 的样式以突出显示
    style_config["HPVT"] = {
        'color': 'red',
        'marker': 'o',
        'zorder': 10  # 确保 HPVT 的线在最上层
    }

    # --- 绘图函数 ---
    def plot_metric(ax, y_col, y_std, title, ylabel):
        ax.set_title(title, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xlabel('Total Number of Tags (N)', fontsize=14)
        
        # 按算法名称循环，确保每个算法有固定的颜色和标记
        for algo_name in cleaned_algo_list:
            df_algo = df_summary[df_summary['algorithm'] == algo_name]
            if df_algo.empty:
                continue

            # 对于每个算法，绘制不同 pm 场景下的线
            for pm, group in df_algo.groupby('missing_rate'):
                style = style_config[algo_name]
                # 用线型区分 pm，实线代表低缺失率，虚线代表高缺失率
                linestyle = '-' if pm == 0.2 else '--'
                label = f'{algo_name} (pm={pm})'
                
                ax.errorbar(group['total_tags'], group[y_col], yerr=group[y_std],
                            label=label, 
                            marker=style['marker'], 
                            color=style['color'], 
                            linestyle=linestyle,
                            capsize=4, 
                            zorder=style['zorder'])
                            
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- 依次绘制四个子图 ---
    plot_metric(axes[0, 0], 'mean_time', 'std_time', 'a) Total Execution Time vs. N', 'Time (µs)')
    plot_metric(axes[0, 1], 'mean_bits', 'std_bits', 'b) Total Energy Consumption vs. N', 'Total Bits')
    plot_metric(axes[1, 0], 'mean_first_time', 'std_first_time', 'c) Time to First Missing vs. N', 'Time (µs)')
    plot_metric(axes[1, 1], 'mean_90p_time', 'std_90p_time', 'd) Time to 90% Completion vs. N', 'Time (µs)')
    
    for row in axes:
        for ax in row:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # 调整布局以防止标题和坐标轴标签重叠
    plt.savefig(OUTPUT_PLOT_FILE, dpi=300, bbox_inches='tight')
    print(f"多维度可扩展性图表已保存到: {OUTPUT_PLOT_FILE}")
    df_summary.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"聚合数据已保存到: {OUTPUT_CSV_FILE}")
    plt.show()

if __name__ == '__main__':
    main()
