# main.py
# -*- coding: utf-8 -*-
"""
一个可扩展、支持多进程的高性能RFID算法对比测试框架。
该框架采用基于比特率的动态时间计算模型，以确保对不同类型算法
（如树分裂、ALOHA等）进行公平的性能评估。
"""
import multiprocessing
import time
from tqdm import tqdm
import sys
import os
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import numpy as np
import csv
from typing import Dict, List, Any
from cpt_algo import CPTAlgo
from Framework import *
from hpvt_algo import HPVTAlgo
from iip_algo import IIPAlgo
from ctmti_algo import CTMTIAlgo
from crmti_algo import CR_MTI_Algo
def run_single_simulation_task(task_config: Dict) -> Dict:
    """
    一个独立的函数，用于在子进程中执行单次或多次仿真任务并返回平均结果。
    """
    alg_class = task_config['alg_class']
    global_cfg = task_config['global_config']
    alg_specific_cfg = task_config['alg_specific_config']
    num_runs = task_config['num_runs']
    suppress_output = task_config.get('suppress_output', True)

    accumulated_results: Dict[str, float] = {}
    accumulated_timeline: List[float] = []
    valid_runs = 0

    # 重定向标准输出以保持主进程终端的整洁
    stdout_target = open(os.devnull, 'w') if suppress_output else sys.stdout
    with redirect_stdout(stdout_target):
        for _ in range(num_runs):
            # 假设 run_missing_tag_simulation 是从Framework.py导入的
            result = run_missing_tag_simulation(global_cfg, alg_class, alg_specific_cfg)
            if result:
                valid_runs += 1
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        accumulated_results[key] = accumulated_results.get(key, 0.0) + value
                
                if 'discovery_timeline' in result and isinstance(result['discovery_timeline'], list):
                    if not accumulated_timeline:
                        accumulated_timeline = [0.0] * len(result['discovery_timeline'])
                    if len(accumulated_timeline) == len(result['discovery_timeline']):
                        for j, val in enumerate(result['discovery_timeline']):
                            accumulated_timeline[j] += val

    if suppress_output and stdout_target is not sys.stdout:
        stdout_target.close()

    if valid_runs > 0:
        averaged_result = {key: val / valid_runs for key, val in accumulated_results.items()}
        if accumulated_timeline:
            averaged_result['discovery_timeline'] = [t / valid_runs for t in accumulated_timeline]
        averaged_result['algorithm_name'] = alg_class.__name__
        return averaged_result
    return {}


def create_and_save_results(all_results: Dict, varying_param_name: str, base_config: Dict):
    """
    一个集成的函数，负责数据转换、生成表格、保存CSV和绘制图表。
    """
    if not all_results:
        print("没有可处理的仿真结果。")
        return

    # 定义要展示和保存的指标及其格式
    metrics_config = {
        'total_protocol_time_us': {'name': '总执行时间 (秒)', 'format': '.4f'},
        'total_bits': {'name': '总通信开销 (Kbits)', 'format': '.2f'},
    }
    
    # 统一进行数据转换和单位修约
    for param_key, alg_results in all_results.items():
        for alg_name, res_dict in alg_results.items():
            if 'total_reader_bits' in res_dict and 'total_tag_bits' in res_dict:
                res_dict['total_bits'] = (res_dict.get('total_reader_bits', 0) + res_dict.get('total_tag_bits', 0)) / 1024.0

    # 生成和打印对比表格
    for metric_key, config in metrics_config.items():
        create_comparative_table(all_results, varying_param_name, metric_key, config)

    # 保存每个指标到独立的CSV文件
    save_each_metric_to_csv(all_results, varying_param_name, metrics_config)
    
    # 绘制发现过程曲线图
    plot_discovery_curves(all_results, varying_param_name, base_config)

def create_comparative_table(results_by_param, varying_param_name, metric_key, metric_config):
    """为单个指标生成并打印一个对比表格。"""
    metric_name, metric_format = metric_config['name'], metric_config['format']
    print("\n" + "*"*80 + f"\n* {'性能指标对比: ' + metric_name:<76} *\n" + "*"*80 + "\n")
    
    first_result_set = next(iter(results_by_param.values()))
    all_alg_names = sorted(list(first_result_set.keys()))
    
    header = [f"{varying_param_name:<15}"] + [f"{name:<15}" for name in all_alg_names]
    print(" | ".join(header))
    print("-" * len(" | ".join(header)))

    sorted_param_values = sorted(results_by_param.keys(), key=float)
    for param_value in sorted_param_values:
        row = [f"{str(param_value):<15}"]
        for alg_name in all_alg_names:
            res = results_by_param.get(param_value, {}).get(alg_name, {})
            value = res.get(metric_key)
            if value is not None:
                # 针对不同单位进行显示转换
                display_value = value / 1e6 if 'time_us' in metric_key else value
                row.append(f"{display_value:{metric_format}}".ljust(15))
            else:
                row.append(f"{'N/A':<15}")
        print(" | ".join(row))

def save_each_metric_to_csv(results_by_param, varying_param_name, metrics_config, folder="results"):
    """为每个指标创建一个独立的CSV文件。"""
    print(f"\n正在保存各指标结果到 '{folder}' 文件夹...")
    os.makedirs(folder, exist_ok=True)
    first_result_set = next(iter(results_by_param.values()))
    all_alg_names = sorted(list(first_result_set.keys()))
    sorted_param_values = sorted(results_by_param.keys(), key=float)

    for metric_key, config in metrics_config.items():
        filename = os.path.join(folder, f"results_{metric_key}.csv")
        try:
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([varying_param_name] + all_alg_names)
                for param_value in sorted_param_values:
                    row = [param_value]
                    for alg_name in all_alg_names:
                        res = results_by_param.get(param_value, {}).get(alg_name, {})
                        value = res.get(metric_key, 'N/A')
                        if 'time_us' in metric_key and isinstance(value, (int, float)):
                            value /= 1e6
                        row.append(value)
                    writer.writerow(row)
            print(f"✅ 指标 '{config['name']}' 已保存到: {filename}")
        except IOError as e:
            print(f"❌ 无法写入文件 {filename}。错误: {e}")

def plot_discovery_curves(results_by_param, varying_param_name, base_config):
    """绘制“发现过程累积时间”曲线图。"""
    print(f"\n正在生成发现过程曲线图...")
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"警告: 设置中文字体失败: {e}")

    for param_value, results in results_by_param.items():
        plt.figure(figsize=(12, 7))
        has_data = False
        for alg_name, res_data in results.items():
            if res_data and 'discovery_timeline' in res_data and res_data['discovery_timeline']:
                has_data = True
                timeline_sec = [t / 1e6 for t in res_data['discovery_timeline']]
                percentages = np.linspace(0, 100, len(timeline_sec))
                plt.plot(percentages, timeline_sec, marker='o', linestyle='-', markersize=4, label=alg_name)
        
        if has_data:
            plt.xlabel("已发现的缺失标签百分比 (%)")
            plt.ylabel("累积所用时间 (秒)")
            plt.title(f"缺失标签发现过程对比\n({varying_param_name}: {param_value}, N={base_config['TOTAL_TAGS']})")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.xticks(np.arange(0, 101, 10))
            plt.xlim(0, 100)
            plt.ylim(bottom=0)
            
            folder = "results"
            os.makedirs(folder, exist_ok=True)
            filename = os.path.join(folder, f"discovery_curve_{varying_param_name.replace(' ', '_')}_{param_value}.png")
            plt.savefig(filename)
            print(f"✅ 已生成图表: {filename}")
            plt.close()

# ==============================================================================
#  主程序入口
# ==============================================================================
def main():
    """主函数，负责配置和驱动整个仿真流程。"""
    multiprocessing.freeze_support()
    print("启动高性能RFID算法对比测试框架...")
    
    # 1. 定义要测试的算法
    algorithms_to_test = {
        'Baseline': {'class': BaselinePollingAlgo, 'config': {}},
        'IIP': {'class': IIPAlgo, 'config': {}},
        'HPVT_Base': {'class': HPVTAlgo, 'config': {}},
        'CTMTI':{'class':CTMTIAlgo,'config':{}},
        'CRMTI':{'class':CR_MTI_Algo,'config':{}},
        'CPT':{'class':CPTAlgo,'config':{}},


        # 'HPVT_SOTA': {'class': HPVTAlgo_SOTA_Inspired, 'config': {}},
    }
    
    # 2. 定义基础配置 (采用公平的动态时间模型)
    base_global_config = {
        'TOTAL_TAGS': 10000,
        'BINARY_LENGTH': 96,
        'DATA_RATE_BPS': 160000.0,
        'BITS_PER_MICROSECOND': 160000.0 / 1.0e6,
        'T1_RTcal': 25.0,
        'T2_TRcal': 25.0,
        'MINIMAL_GUARD_TIME_US': 50.0, # 统一的最小保护间隔
        'READER_CMD_BASE_BITS': 37,
        'TAG_SHORT_RESP_BITS': 1,
    }
    
    # 3. 配置仿真任务
    # 通过改变 MISSING_RATE 来进行对比
    varying_param_name = "丢失率"
    param_values = [round(rate, 2) for rate in np.arange(0.1, 1.0, 0.2)] # e.g., 0.1, 0.3, ..., 0.9
    num_runs_per_config = 5  # 每次配置运行5次取平均值，结果更稳定
    
    tasks = []
    for param_value in param_values:
        current_global_config = base_global_config.copy()
        current_global_config['MISSING_RATE'] = param_value
        param_key_str = f"{param_value:.2f}"
        for alg_id, alg_info in algorithms_to_test.items():
            tasks.append({
                'param_key': param_key_str, 'alg_id': alg_id, 'alg_class': alg_info['class'],
                'global_config': current_global_config, 'alg_specific_config': alg_info['config'],
                'num_runs': num_runs_per_config
            })

    # 4. 使用多进程执行所有任务
    print(f"\n准备执行 {len(tasks)} 个仿真任务 (每个任务运行 {num_runs_per_config} 次)...")
    num_processes = min(multiprocessing.cpu_count() - 1, 8) if multiprocessing.cpu_count() > 1 else 1
    all_results: Dict[str, Dict[str, Any]] = {}
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        with tqdm(total=len(tasks), desc="执行仿真任务", unit="task") as pbar:
            async_results = {i: pool.apply_async(run_single_simulation_task, (task,)) for i, task in enumerate(tasks)}
            for i, async_res in async_results.items():
                try:
                    task_result = async_res.get(timeout=300) # 5分钟超时
                    if task_result:
                        param_key = tasks[i]['param_key']
                        alg_id = tasks[i]['alg_id']
                        if param_key not in all_results:
                            all_results[param_key] = {}
                        all_results[param_key][alg_id] = task_result
                except Exception as e:
                    task_info = tasks[i]
                    print(f"\n错误: 任务执行失败！算法 '{task_info['alg_id']}' 在配置 {task_info['param_key']} 下发生异常: {e}")
                pbar.update(1)

    # 5. 处理并展示结果
    if all_results:
        create_and_save_results(all_results, varying_param_name, base_global_config)
    else:
        print("\n所有仿真任务均未返回有效结果。")

    print("\n框架执行完毕。")


if __name__ == '__main__':
    # 确保此脚本作为主程序运行时，能够正确启动多进程
    main()
