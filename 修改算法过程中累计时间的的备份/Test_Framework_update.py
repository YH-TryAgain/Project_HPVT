# main.py
import multiprocessing
import time
from tqdm import tqdm
import sys
import os
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import numpy as np

# 导入仿真框架的核心函数和基线算法
# 假设框架代码保存在 Framework_Update.py
from Framework_Update import *

# 导入您自己的算法实现 (如果需要测试，请取消注释)
from iip_algo import IIPAlgo
from cpt_algo import CPTAlgo
from hpvt_algo import HPVTAlgo


def run_single_simulation_task(task_config):
    """
    一个独立的函数，用于在子进程中执行单次或多次仿真任务并返回平均结果。
    这是被多进程池调用的工作单元。
    """
    alg_class = task_config['alg_class']
    global_cfg = task_config['global_config']
    alg_specific_cfg = task_config['alg_specific_config']
    num_runs = task_config['num_runs']
    suppress_output = task_config.get('suppress_output', True)

    accumulated_results = {}
    # 为新的时间线指标初始化累加器
    accumulated_timeline = []
    valid_runs = 0

    # 抑制子进程中的标准输出，保持主进程的整洁
    stdout_target = open(os.devnull, 'w') if suppress_output else sys.stdout
    with redirect_stdout(stdout_target):
        for i in range(num_runs):
            # 假设 run_missing_tag_simulation 现在返回包含 discovery_timeline 的结果
            result = run_missing_tag_simulation(global_cfg, alg_class, alg_specific_cfg)
            if result:
                valid_runs += 1
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        accumulated_results[key] = accumulated_results.get(key, 0.0) + value
                
                # 累加 discovery_timeline
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
        
        # 计算平均的 discovery_timeline
        if accumulated_timeline:
            averaged_result['discovery_timeline'] = [t / valid_runs for t in accumulated_timeline]

        # 将一些非数值或标识性的字段加回去
        averaged_result['algorithm_name'] = alg_class.__name__
        averaged_result['config_summary'] = {'global': global_cfg, 'specific': alg_specific_cfg}
        return averaged_result
    return None


def create_comparative_table(results_by_param: dict, varying_param_name: str, metrics_config: dict):
    """
    根据不同参数下的仿真结果，生成一个格式化的对比表格。
    """
    if not results_by_param:
        print("没有可供对比的结果。")
        return

    first_result_set = next(iter(results_by_param.values()))
    if not first_result_set:
        print("结果集为空，无法生成表格。")
        return
    alg_names = list(first_result_set.keys())
    
    for metric_key, config in metrics_config.items():
        metric_name = config['name']
        metric_format = config['format']
        
        print(f"\n--- 性能指标对比: {metric_name} ---\n")
        
        header = [f"{varying_param_name:<18}"] + [f"{name:<18}" for name in alg_names]
        table = [" | ".join(header)]
        table.append("-" * len(table[0]))

        # 按参数值进行数字排序，而不是字符串排序
        sorted_param_values = sorted(results_by_param.keys(), key=float)

        for param_value in sorted_param_values:
            results = results_by_param[param_value]
            row = [f"{str(param_value):<18}"]
            for alg_name in alg_names:
                res = results.get(alg_name)
                if res and metric_key in res:
                    value = res[metric_key]
                    row.append(f"{value:{metric_format}}".ljust(18))
                else:
                    row.append(f"{'N/A':<18}")
            table.append(" | ".join(row))
        
        for line in table:
            print(line)

def print_timeline_details(results_by_param: dict):
    """
    格式化打印每个算法在不同阶段的累积时间。
    """
    print("\n" + "="*70)
    print(" " * 20 + "各算法累积时间详细数据")
    print("="*70)

    for param_value, results in results_by_param.items():
        print(f"\n--- 场景: 丢失率 = {param_value} ---")
        for alg_id, res_data in results.items():
            if 'discovery_timeline' in res_data and res_data['discovery_timeline']:
                print(f"\n算法: {res_data['algorithm_name']}")
                timeline_in_seconds = [t / 1e6 for t in res_data['discovery_timeline']]
                
                if len(timeline_in_seconds) > 0:
                    percentages = np.linspace(0, 100, len(timeline_in_seconds))
                    header = " | ".join([f"{p:5.0f}%" for p in percentages])
                    values = " | ".join([f"{t:6.3f}s" for t in timeline_in_seconds])
                    
                    print(header)
                    print("-" * len(header))
                    print(values)
            else:
                print(f"\n算法: {alg_id} - 未找到时间线数据。")

def plot_discovery_curves(results_by_param: dict, varying_param_name: str, base_config: dict):
    """
    绘制“累积时间”曲线图。
    """
    print("\n" + "="*70)
    print(" " * 20 + "生成缺失标签发现过程曲线 (累积时间)")
    print("="*70)

    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"警告: 设置中文字体失败，图表中的中文可能无法正常显示。错误: {e}")

    sorted_param_values = sorted(results_by_param.keys(), key=float)

    for param_value in sorted_param_values:
        results = results_by_param[param_value]
        plt.figure(figsize=(10, 6))
        
        has_data = False
        for alg_id, res_data in results.items():
            if 'discovery_timeline' in res_data and res_data['discovery_timeline']:
                has_data = True
                cumulative_timeline = res_data['discovery_timeline']
                cumulative_timeline_in_seconds = [t / 1e6 for t in cumulative_timeline]
                
                percentages = np.linspace(0, 100, len(cumulative_timeline))
                plt.plot(percentages, cumulative_timeline_in_seconds, marker='o', linestyle='-', label=res_data['algorithm_name'])
        
        if has_data:
            plt.xlabel("已发现的缺失标签百分比 (%)")
            plt.ylabel("累积所用时间 (秒)")
            plt.title(f"缺失标签发现过程对比 (累积时间)\n({varying_param_name}: {param_value}, N={base_config['TOTAL_TAGS']})")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.xticks(np.arange(0, 101, 10))
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            filename = f"discovery_curve_cumulative_{varying_param_name.replace(' ', '_')}_{param_value}.png"
            plt.savefig(filename)
            print(f"已生成图表: {filename}")
            plt.close()
        else:
            print(f"在 {varying_param_name} = {param_value} 时, 未找到 'discovery_timeline' 数据，跳过绘图。")


if __name__ == '__main__':
    multiprocessing.freeze_support()

    print("开始缺失标签识别算法的性能对比测试框架...")

    algorithms_to_test = {
        'BaselinePolling': {'class': BaselinePollingAlgo, 'config': {}},
        'IIPAlgo': {'class': IIPAlgo, 'config': {}},
        'CPTAlgo': {'class': CPTAlgo, 'config': {}},
        'HPVT': {'class': HPVTAlgo, 'config': {}}
    }
    
    base_global_config = {
        'TOTAL_TAGS': 10000,
        'BINARY_LENGTH': 96,
        'DATA_RATE_BPS': 160000.0,
        'T_s_SLOT_TIME_US': 400.0,
        'T_l_SLOT_TIME_US': 800.0,
        'T_tag_SLOT_TIME_US': 2400.0,
        'T1_RTcal': 25.0,
        'T2_TRcal': 25.0,
        'READER_CMD_BASE_BITS': 37,
        'TAG_SHORT_RESP_BITS': 1,
    }
    base_global_config['BITS_PER_MICROSECOND'] = base_global_config['DATA_RATE_BPS'] / 1.0e6

    # --- Section 1: 通用性能评估 ---
    print("\n" + "#"*30 + "\n#   第1部分: 通用性能评估 (随丢失率变化)   #\n" + "#"*30)
    
    general_test_params = [{'MISSING_RATE': round(rate, 2)} for rate in np.arange(0.0, 1.01, 0.1)]
    num_runs_per_config = 3
    all_results = {}

    tasks_general = []
    for params in general_test_params:
        current_global_config = base_global_config.copy()
        current_global_config.update(params)
        param_key_str = f"{params['MISSING_RATE']:.1f}"
        for alg_id, alg_info in algorithms_to_test.items():
            tasks_general.append({
                'param_key': param_key_str,
                'alg_id': alg_id,
                'alg_class': alg_info['class'],
                'global_config': current_global_config,
                'alg_specific_config': alg_info['config'],
                'num_runs': num_runs_per_config
            })

    print(f"\n通用性能评估将执行 {len(tasks_general)} 个仿真任务...")
    num_processes = min(multiprocessing.cpu_count(), 8)
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        with tqdm(total=len(tasks_general), desc="执行通用性能评估", unit="task") as pbar:
            async_results = {i: pool.apply_async(run_single_simulation_task, (task,)) for i, task in enumerate(tasks_general)}
            for i, async_res in async_results.items():
                try:
                    task_result = async_res.get(timeout=60)
                    if task_result:
                        param_key = tasks_general[i]['param_key']
                        alg_id = tasks_general[i]['alg_id']
                        if param_key not in all_results:
                            all_results[param_key] = {}
                        all_results[param_key][alg_id] = task_result
                except Exception as e:
                    task_info = tasks_general[i]
                    print(f"\n错误: 任务执行失败！算法 '{task_info['alg_id']}' 在配置 {task_info['param_key']} 下发生异常: {e}")
                pbar.update(1)

    print("\n" + "="*70)
    print(" " * 20 + "通用性能评估结果")
    print("="*70)
    
    metrics_to_display = {
        'total_protocol_time_us': {'name': '总执行时间 (秒)', 'format': '.4f'},
        'total_bits': {'name': '总通信开销 (Kbits)', 'format': '.1f'},
        'f1_score': {'name': 'F1 分数', 'format': '.4f'}
    }
    
    for param_key, alg_results in all_results.items():
        for alg_name, res_dict in alg_results.items():
            if 'total_protocol_time_us' in res_dict:
                res_dict['total_protocol_time_us'] /= 1e6
            if 'total_reader_bits' in res_dict and 'total_tag_bits' in res_dict:
                res_dict['total_bits'] = (res_dict['total_reader_bits'] + res_dict['total_tag_bits']) / 1024.0

    create_comparative_table(all_results, "丢失率", metrics_to_display)

    # --- Section 2: 累积时间深度分析 ---
    print("\n" + "#"*30 + "\n#   第2部分: 累积时间深度分析 (固定丢失率)   #\n" + "#"*30)

    TIMELINE_ANALYSIS_RATE = 0.5
    print(f"\n将对 丢失率 = {TIMELINE_ANALYSIS_RATE} 的场景进行深度分析...")

    timeline_results = {}
    param_key_for_timeline = f"{TIMELINE_ANALYSIS_RATE:.1f}"
    
    if param_key_for_timeline in all_results:
        timeline_results[param_key_for_timeline] = all_results[param_key_for_timeline]
        
        # 打印详细的时间线数据
        print_timeline_details(timeline_results)
        
        # 绘制曲线图
        plot_discovery_curves(timeline_results, "丢失率", base_global_config)
    else:
        print(f"\n未能在通用评估结果中找到丢失率 = {TIMELINE_ANALYSIS_RATE} 的数据，无法进行深度分析。")
        print("请确保 'general_test_params' 包含了这个值。")

    print("\n" + "="*70)
    print("仿真框架执行完毕。")
