# main.py
import multiprocessing
import time
from tqdm import tqdm
import sys
import os
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import numpy as np
import csv

# 导入仿真框架的核心函数和基线算法
# 假设框架代码保存在 Framework.py
from Framework import *

# 导入您自己的算法实现
from HPVT_Adaptive import  HPVTAlgo_SOTA_Inspired
from iip_algo import IIPAlgo
from cpt_algo import CPTAlgo
from hpvt_algo import HPVTAlgo
from crmti_algo import CR_MTI_Algo
from ctmti_algo import CTMTIAlgo
from ecumi_algo import CUMIAlgo
from ecumi_algo import ECUMIAlgo

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
    accumulated_timeline = []
    valid_runs = 0

    stdout_target = open(os.devnull, 'w') if suppress_output else sys.stdout
    with redirect_stdout(stdout_target):
        for i in range(num_runs):
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
        averaged_result['config_summary'] = {'global': global_cfg, 'specific': alg_specific_cfg}
        return averaged_result
    return None

def create_comparative_table(results_by_param: dict, varying_param_name: str, metrics_config: dict):
    """
    根据不同参数下的仿真结果，为每个指标生成一个独立的、格式化的对比表格。
    """
    if not results_by_param:
        print("没有可供对比的结果。")
        return

    first_result_set = next(iter(results_by_param.values()))
    if not first_result_set:
        print("结果集为空，无法生成表格。")
        return
    
    all_alg_names = sorted(list(first_result_set.keys()))
    
    for metric_key, config in metrics_config.items():
        metric_name = config['name']
        metric_format = config['format']
        
        print("\n" + "*"*80)
        print(f"* {'性能指标对比: ' + metric_name:<76} *")
        print("*"*80 + "\n")
        
        header = [f"{varying_param_name:<18}"] + [f"{name:<18}" for name in all_alg_names]
        table = [" | ".join(header)]
        table.append("-" * len(table[0]))

        sorted_param_values = sorted(results_by_param.keys(), key=float)

        for param_value in sorted_param_values:
            results = results_by_param.get(param_value, {})
            row = [f"{str(param_value):<18}"]
            for alg_name in all_alg_names:
                res = results.get(alg_name)
                if res and metric_key in res:
                    value = res[metric_key]
                    if metric_key == 'time_to_first_missing' and isinstance(value, (int, float)) and value < 0:
                        row.append(f"{'N/A':<18}")
                    else:
                        # 使用更新后的格式化字符串
                        row.append(f"{value:{metric_format}}".ljust(18))
                else:
                    row.append(f"{'N/A':<18}")
            table.append(" | ".join(row))
        
        for line in table:
            print(line)
    print("\n" + "*"*80)

def save_each_metric_to_csv(results_by_param: dict, varying_param_name: str, metrics_config: dict, folder_name: str = "simulation_results"):
    """
    为每个性能指标创建一个独立的CSV文件，并将结果存入其中。
    """
    print(f"\n正在尝试将各指标结果分别保存到 '{folder_name}' 文件夹中...")
    if not results_by_param:
        print("没有结果可供保存。")
        return

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"已创建结果文件夹: {folder_name}")

    first_result_set = next(iter(results_by_param.values()))
    all_alg_names = sorted(list(first_result_set.keys()))
    sorted_param_values = sorted(results_by_param.keys(), key=float)

    for metric_key, config in metrics_config.items():
        filename = os.path.join(folder_name, f"results_{metric_key}.csv")
        try:
            with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                headers = [varying_param_name] + all_alg_names
                writer.writerow(headers)

                for param_value in sorted_param_values:
                    row = [param_value]
                    alg_results_for_param = results_by_param.get(param_value, {})
                    for alg_name in all_alg_names:
                        res_dict = alg_results_for_param.get(alg_name, {})
                        value = res_dict.get(metric_key, 'N/A')
                        
                        if metric_key == 'time_to_first_missing' and isinstance(value, (int, float)) and value < 0:
                            row.append('N/A')
                        else:
                            row.append(value)
                    writer.writerow(row)
            
            print(f"✅ 指标 '{config['name']}' 的结果已成功保存到: {filename}")

        except IOError as e:
            print(f"❌ 错误: 无法写入CSV文件 {filename}。原因: {e}")
        except Exception as e:
            print(f"❌ 发生未知错误在保存 {filename} 时: {e}")


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
                # !!优化!!: 在这里直接对timeline数据进行修约
                timeline_in_seconds = [round(t / 1e6, 3) for t in res_data['discovery_timeline']]
                
                if len(timeline_in_seconds) > 0:
                    percentages = np.linspace(0, 100, len(timeline_in_seconds))
                    header = " | ".join([f"{p:5.0f}%" for p in percentages])
                    # 调整格式以适应三位小数
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
                # !!优化!!: 绘图时也使用修约后的数据
                cumulative_timeline_in_seconds = [round(t / 1e6, 3) for t in res_data['discovery_timeline']]
                
                percentages = np.linspace(0, 100, len(cumulative_timeline_in_seconds))
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
            
            folder_name = "simulation_results"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            filename = os.path.join(folder_name, f"discovery_curve_cumulative_{varying_param_name.replace(' ', '_')}_{param_value}.png")
            
            plt.savefig(filename)
            print(f"已生成图表: {filename}")
            plt.close()
        else:
            print(f"在 {varying_param_name} = {param_value} 时, 未找到 'discovery_timeline' 数据，跳过绘图。")


if __name__ == '__main__':
    multiprocessing.freeze_support()

    print("开始缺失标签识别算法的性能对比测试框架...")
    
    # 1. 统一仿真参数配置
    # 减少测试点以加快调试
    ALL_TEST_RATES = [round(rate, 2) for rate in np.arange(0.0, 1.01, 0.1)]
    TIMELINE_ANALYSIS_RATE = 0.1
    if TIMELINE_ANALYSIS_RATE not in ALL_TEST_RATES:
         ALL_TEST_RATES.append(TIMELINE_ANALYSIS_RATE)
         ALL_TEST_RATES.sort()
    
    # 2. 定义基础配置和算法
    algorithms_to_test = {
        'BaselinePolling': {'class': BaselinePollingAlgo, 'config': {}},
        'IIPAlgo': {'class': IIPAlgo, 'config': {}},
        'CPTAlgo': {'class': CPTAlgo, 'config': {}},
        'HPVT': {'class': HPVTAlgo, 'config': {}},
    #    'CR_MTI': {'class': CR_MTI_Algo, 'config': {}},
    #    'CTMTI': {'class': CTMTIAlgo, 'config': {}},
    #    'CUMIA': {'class': CUMIAlgo, 'config': {}},
    #    'ECUMIA': {'class': ECUMIAlgo, 'config': {}},
        'HPVTAlgo_A':{'class':HPVTAlgo_SOTA_Inspired,'config':{}},
    #    'TDSP': {'class': TSDPAlgo, 'config': {}},
    #TODO 优化失败    'HPVT_Adaptive': {'class': HPVT_AdaptiveAlgo, 'config': {}}
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
    num_runs_per_config = 3 # 增加运行次数以获得更平滑的平均值
    
    # 3. 执行通用性能评估
    print("\n" + "#"*30 + "\n#  第1部分: 通用性能评估 (随丢失率变化)  #\n" + "#"*30)
    
    tasks_general = []
    for rate in ALL_TEST_RATES:
        current_global_config = base_global_config.copy()
        current_global_config['MISSING_RATE'] = rate
        param_key_str = f"{rate:.2f}"
        for alg_id, alg_info in algorithms_to_test.items():
            tasks_general.append({
                'param_key': param_key_str, 'alg_id': alg_id, 'alg_class': alg_info['class'],
                'global_config': current_global_config, 'alg_specific_config': alg_info['config'],
                'num_runs': num_runs_per_config
            })

    print(f"\n通用性能评估将执行 {len(tasks_general)} 个仿真任务...")
    num_processes = min(multiprocessing.cpu_count(), 8)
    all_results = {}
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        with tqdm(total=len(tasks_general), desc="执行通用性能评估", unit="task") as pbar:
            async_results = {i: pool.apply_async(run_single_simulation_task, (task,)) for i, task in enumerate(tasks_general)}
            for i, async_res in async_results.items():
                try:
                    task_result = async_res.get(timeout=180) # 增加超时
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
    
    # !!优化!!: 调整显示格式为3位小数
    metrics_to_display = {
        'total_protocol_time_us': {'name': '总执行时间 (秒)', 'format': '.3f'},
        'time_to_first_missing': {'name': '首次发现时间 (秒)', 'format': '.3f'},
        'lssr': {'name': '后期迟滞比 (LSSR)', 'format': '.3f'},
        'total_bits': {'name': '总通信开销 (Kbits)', 'format': '.3f'},
        'f1_score': {'name': 'F1 分数', 'format': '.3f'}
    }
    
    # !!优化!!: 集中进行数据转换和修约
    for param_key, alg_results in all_results.items():
        for alg_name, res_dict in alg_results.items():
            # 转换单位
            if 'total_protocol_time_us' in res_dict:
                res_dict['total_protocol_time_us'] /= 1e6
            if 'time_to_first_missing' in res_dict and res_dict.get('time_to_first_missing', -1) >= 0:
                res_dict['time_to_first_missing'] /= 1e6
            if 'total_reader_bits' in res_dict and 'total_tag_bits' in res_dict:
                res_dict['total_bits'] = (res_dict['total_reader_bits'] + res_dict['total_tag_bits']) / 1024.0
            
            # 对所有待显示指标进行修约
            for key in metrics_to_display.keys():
                if key in res_dict and isinstance(res_dict[key], float):
                    res_dict[key] = round(res_dict[key], 3)

    # 为每个指标生成独立的表格
    create_comparative_table(all_results, "丢失率", metrics_to_display)

    # 为每个指标保存独立的CSV文件
    save_each_metric_to_csv(all_results, "丢失率", metrics_to_display, folder_name="simulation_results")

    # 4. 执行累积时间深度分析
    print("\n" + "#"*30 + "\n#  第2部分: 累积时间深度分析 (固定丢失率)  #\n" + "#"*30)
    print(f"\n将对 丢失率 = {TIMELINE_ANALYSIS_RATE} 的场景进行深度分析...")

    timeline_results = {}
    param_key_for_timeline = f"{TIMELINE_ANALYSIS_RATE:.2f}"
    
    if param_key_for_timeline in all_results:
        timeline_results[param_key_for_timeline] = all_results[param_key_for_timeline]
        print_timeline_details(timeline_results)
        plot_discovery_curves(timeline_results, "丢失率", base_global_config)
    else:
        print(f"\n未能在通用评估结果中找到丢失率 = {TIMELINE_ANALYSIS_RATE} 的数据，无法进行深度分析。")

    print("\n" + "="*70)
    print("仿真框架执行完毕。")