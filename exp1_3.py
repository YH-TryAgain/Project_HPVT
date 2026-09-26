# -*- coding: utf-8 -*-
"""
实验 1.3: 查询聚合 (Query Aggregation) 优化效果量化

================================================================================
本实验的核心目标 (Objective):
1. 精确地量化 “查询聚合” 优化机制对 HPVT 协议性能的实际贡献。
2. 通过创建一个不包含此优化的 “基础版” HPVT 作为对比基准，
   在时间和能耗两个维度上进行性能对比。
3. 验证该优化主要是在降低总通信比特数 (特别是读写器能耗) 方面做出贡献，
   同时可能伴随有一定的时间性能提升。
4. 【新功能】将四个核心指标的结果分别导出为独立的、适合制表的CSV文件。

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

# 导入您的仿真框架和HPVT算法
# 请确保这些模块在您的项目中路径正确
from framework import run_missing_tag_simulation
from hpvt_algo import HPVT_AggregatedAlgo, _HPVT_Task, _AlohaContext
from framework import CONSTANTS, MissingTagAlgorithmInterface, AlgorithmStepResult

# ==============================================================================
# 1. 定义一个不包含查询聚合优化的“基础版”HPVT
# ==============================================================================
class HPVT_BaseAlgo(HPVT_AggregatedAlgo):
    """
    一个基础版本的HPVT，它继承了所有逻辑，但重写了组探测方法，
    使其不执行任何查询聚合优化。
    """
    def _perform_group_probe(self) -> AlgorithmStepResult:
        """
        【已重写】这是基础版的组探测，每次都支付全部的指令开销。
        """
        if not self.task_queue:
            return AlgorithmStepResult(operation_type='idle')

        # 直接弹出任务，没有任何聚合检查
        task = self.task_queue.popleft()
        
        # 每次探测都计算完整的读写器比特数
        reader_bits = CONSTANTS.READER_CMD_BASE_BITS + len(task.prefix)
        
        # 后续的逻辑与聚合版完全相同
        responding_tags = [t for t in task.tags_in_scope if t.is_present]
        tag_bits = CONSTANTS.TAG_SHORT_RESP_BITS if responding_tags else 0
        
        if not responding_tags:
            self.metrics["pruning_events"] += 1
            self.found_missing_ids.update({t.id for t in task.tags_in_scope})
            op_desc = f"Base-Probe '{task.prefix}': Idle, Pruned"
        else:
            if len(task.tags_in_scope) <= self.aloha_threshold:
                self.metrics["aloha_frames_initiated"] += 1
                self.current_state = self.STATE_VERIFYING
                self.aloha_context = _AlohaContext(task.tags_in_scope, len(responding_tags))
                op_desc = f"Base-Probe '{task.prefix}': Active -> Start ALOHA"
            else:
                if len(task.prefix) < self.binary_length:
                    p0, p1 = task.prefix + '0', task.prefix + '1'
                    tags0 = [t for t in task.tags_in_scope if t.id.startswith(p0)]
                    tags1 = [t for t in task.tags_in_scope if t.id.startswith(p1)]
                    if tags1: self.task_queue.appendleft(_HPVT_Task(p1, tags1))
                    if tags0: self.task_queue.appendleft(_HPVT_Task(p0, tags0))
                op_desc = f"Base-Probe '{task.prefix}': Active -> Splitting"
                
        return AlgorithmStepResult('PROBE', reader_bits, tag_bits, op_desc)

# ==============================================================================
# 2. 实验配置区 (Configuration Area)
# ==============================================================================
print("正在加载实验 1.3 的配置...")
EXPERIMENT_NAME = "Experiment 1.3: Effect of Query Aggregation Optimization"
OUTPUT_DIR, RESULTS_DIR = "plots", "results"
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_PLOT_FILE = os.path.join(OUTPUT_DIR, "fig1.3_aggregation_effect.png")

SCENARIO_CONFIG = {'TOTAL_TAGS': 2000, 'BINARY_LENGTH': 96}
PM_RANGE = np.linspace(0.1, 0.9, 9)
NUM_RUNS = 20

ALGORITHMS_TO_TEST = [
    {
        "class": HPVT_BaseAlgo,
        "name": "HPVT (Base)(128)",
        "config": {"aloha_threshold": 128, "binary_length": 96}
    },
    {
        "class": HPVT_AggregatedAlgo,
        "name": "HPVT (Aggregated)(128)",
        "config": {"aloha_threshold": 128, "binary_length": 96}
    },
        {
        "class": HPVT_BaseAlgo,
        "name": "HPVT (Base)(64)",
        "config": {"aloha_threshold": 64, "binary_length": 96}
    },
    {
        "class": HPVT_AggregatedAlgo,
        "name": "HPVT (Aggregated)(64)",
        "config": {"aloha_threshold": 64, "binary_length": 96}
    },
]

# ==============================================================================
# 3. 并行任务函数与主执行逻辑
# ==============================================================================
def run_single_task(task_params: tuple):
    """ 
    【已修改】执行单次仿真任务的包装函数，现在会返回所有四个核心指标。
    """
    algo_info, pm, run_id, scenario_config_base = task_params
    current_scenario_config = scenario_config_base.copy()
    current_scenario_config['MISSING_RATE'] = pm
    result = run_missing_tag_simulation(
        scenario_config=current_scenario_config,
        algorithm_class=algo_info['class'],
        algorithm_specific_config=algo_info['config']
    )
    
    timeline = result.get('discovery_timeline_us', [])
    time_to_90_percent = timeline[9] if len(timeline) > 9 else -1.0

    # 单位统一转换为毫秒(ms)
    record = {
        "algorithm": algo_info['name'],
        "missing_rate": pm,
        "run_id": run_id,
        "total_time_ms": result.get('total_protocol_time_us', 0) / 1000.0,
        "total_bits": result.get('total_reader_bits', 0) + result.get('total_tag_bits', 0),
        "time_to_first_ms": result.get('time_to_first_missing_us', -1.0) / 1000.0,
        "time_to_90p_ms": time_to_90_percent / 1000.0,
    }
    return record

def main():
    """ 主函数，负责执行整个实验 """
    print("正在准备并行仿真任务...")
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
    df_summary = df_raw.groupby(['algorithm', 'missing_rate']).agg(
        mean_time=('total_time_ms', 'mean'), std_time=('total_time_ms', 'std'),
        mean_bits=('total_bits', 'mean'), std_bits=('total_bits', 'std'),
        mean_first_time=('time_to_first_ms', 'mean'), std_first_time=('time_to_first_ms', 'std'),
        mean_90p_time=('time_to_90p_ms', 'mean'), std_90p_time=('time_to_90p_ms', 'std'),
    ).reset_index()

    print("\n--- 实验结果摘要 ---")
    print(df_summary)

    # ==========================================================
    # 4. 【新增】将结果分别保存为适合制表的CSV文件
    # ==========================================================
    print("\n正在导出为适合制表的CSV文件...")
    
    def reshape_and_save(df, metric_prefix, output_filename):
        """ 一个辅助函数，用于重塑数据并保存为CSV """
        mean_col = f'mean_{metric_prefix}'
        std_col = f'std_{metric_prefix}'
        
        # 筛选出需要的列
        df_metric = df[['algorithm', 'missing_rate', mean_col, std_col]]
        
        # 使用pivot_table进行数据重塑
        df_pivot = df_metric.pivot_table(
            index='missing_rate',
            columns='algorithm',
            values=[mean_col, std_col]
        )
        
        # 调整列的顺序，使mean和std配对出现
        df_pivot = df_pivot.swaplevel(0, 1, axis=1).sort_index(axis=1)
        
        # 保存到CSV
        filepath = os.path.join(RESULTS_DIR, output_filename)
        df_pivot.to_csv(filepath)
        print(f" -> 已保存: {filepath}")

    # 为四个核心指标分别调用该函数
    reshape_and_save(df_summary, 'time', 'exp1.3_total_time.csv')
    reshape_and_save(df_summary, 'bits', 'exp1.3_total_bits.csv')
    reshape_and_save(df_summary, 'first_time', 'exp1.3_time_to_first.csv')
    reshape_and_save(df_summary, '90p_time', 'exp1.3_time_to_90p.csv')
    
    # ==========================================================
    # 5. 可视化部分 (保持不变，但修正以反映新数据)
    # ==========================================================
    print("\n正在生成对比图表...")
    plt.style.use('seaborn-v0_8-whitegrid')
    # 只绘制最重要的两个指标：总时间和总能耗
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), sharex=True)
    fig.suptitle(EXPERIMENT_NAME, fontsize=20, weight='bold')

    markers = {'HPVT (Base)': 's', 'HPVT (Aggregated)': 'o'}
    colors = {'HPVT (Base)': 'tab:orange', 'HPVT (Aggregated)': 'tab:blue'}

    for name, group in df_summary.groupby('algorithm'):
        # --- 绘制子图 a) 时间效率 ---
        ax1.errorbar(
            group['missing_rate'], group['mean_time'], yerr=group['std_time'],
            label=name, fmt=f'-{markers.get(name)}', capsize=5, color=colors.get(name)
        )
        # --- 绘制子图 b) 能量消耗 ---
        ax2.errorbar(
            group['missing_rate'], group['mean_bits'], yerr=group['std_bits'],
            label=name, fmt=f'--{markers.get(name)}', capsize=5, color=colors.get(name)
        )
        
    ax1.set_title('a) Time Efficiency Comparison', fontsize=14)
    ax1.set_ylabel('Total Execution Time (ms)', fontsize=12)
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax2.set_title('b) Total Energy Consumption Comparison', fontsize=14)
    ax2.set_xlabel(r'Missing Rate ($p_m$)', fontsize=12)
    ax2.set_ylabel('Total Bits', fontsize=12)
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    env_params_text = f"Environment: N = {SCENARIO_CONFIG['TOTAL_TAGS']}, τ_A = {ALGORITHMS_TO_TEST[0]['config']['aloha_threshold']}"
    fig.text(0.95, 0.97, env_params_text, ha='right', va='top', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_PLOT_FILE, dpi=300)
    print(f"对比图表已保存到: {OUTPUT_PLOT_FILE}")
    plt.show()

if __name__ == '__main__':
    main()
