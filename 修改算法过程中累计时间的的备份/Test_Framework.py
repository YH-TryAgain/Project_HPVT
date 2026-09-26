# main.py
import multiprocessing
import time
from tqdm import tqdm
import sys
import os
from contextlib import redirect_stdout

# 导入仿真框架的核心函数和基线算法
# 假设框架代码保存在 missing_tag_simulation_framework.py
from Framework import *

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
    valid_runs = 0

    # 抑制子进程中的标准输出，保持主进程的整洁
    stdout_target = open(os.devnull, 'w') if suppress_output else sys.stdout
    with redirect_stdout(stdout_target):
        for _ in range(num_runs):
            result = run_missing_tag_simulation(global_cfg, alg_class, alg_specific_cfg)
            if result:
                valid_runs += 1
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        accumulated_results[key] = accumulated_results.get(key, 0.0) + value
    
    if suppress_output and stdout_target is not sys.stdout:
        stdout_target.close()

    if valid_runs > 0:
        averaged_result = {key: val / valid_runs for key, val in accumulated_results.items()}
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

    # 获取所有参与对比的算法名称，并保持顺序
    first_result_set = next(iter(results_by_param.values()))
    if not first_result_set:
        print("结果集为空，无法生成表格。")
        return
    alg_names = list(first_result_set.keys())
    
    # 打印每个指标的对比表格
    for metric_key, config in metrics_config.items():
        metric_name = config['name']
        metric_format = config['format']
        
        print(f"\n--- 性能指标对比: {metric_name} ---\n")
        
        # 定义表头
        header = [f"{varying_param_name:<18}"] + [f"{name:<18}" for name in alg_names]
        table = [" | ".join(header)]
        table.append("-" * len(table[0]))

        # 填充表格数据
        for param_value, results in sorted(results_by_param.items()):
            row = [f"{str(param_value):<18}"]
            for alg_name in alg_names:
                res = results.get(alg_name)
                if res and metric_key in res:
                    value = res[metric_key]
                    row.append(f"{value:{metric_format}}".ljust(18))
                else:
                    row.append(f"{'N/A':<18}")
            table.append(" | ".join(row))
        
        # 打印完整的表格
        for line in table:
            print(line)


if __name__ == '__main__':
    # 在 Windows 上使用 multiprocessing 时，建议添加此行
    multiprocessing.freeze_support()

    print("开始缺失标签识别算法的性能对比测试框架...")

    # --- 1. 定义要测试的算法 ---
    algorithms_to_test = {
        'BaselinePolling': {'class': BaselinePollingAlgo, 'config': {}},
        'IIPAlgo': {'class': IIPAlgo, 'config': {}},
        'CPTAlgo': {'class': CPTAlgo, 'config': {}},
        'HPVT':{'class': HPVTAlgo, 'config':{}}
    }
    
    # --- 2. 定义要变化的参数 ---
    # 您可以在这里定义多组实验，例如改变 TOTAL_TAGS 或 MISSING_RATE
    # 示例：我们固定 TOTAL_TAGS，改变 MISSING_RATE
    
    base_global_config = {
        'TOTAL_TAGS': 50000,
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

    varying_params_list = [
        {'MISSING_RATE': 0.1}, 
        {'MISSING_RATE': 0.2},  
        {'MISSING_RATE': 0.3}, 
        {'MISSING_RATE': 0.4}, 
        {'MISSING_RATE': 0.5}, 
        {'MISSING_RATE': 0.6}, 
        {'MISSING_RATE': 0.7}, 
    ]
    
    num_runs_per_config = 3  # 为消除随机性，每个配置运行3次取平均

    # --- 3. 构建所有仿真任务 ---
    tasks = []
    for params in varying_params_list:
        current_global_config = base_global_config.copy()
        current_global_config.update(params)
        
        # 将参数值作为任务的标识符
        param_key = f"MissingRate={params['MISSING_RATE']:.3f}"
        
        for alg_id, alg_info in algorithms_to_test.items():
            task = {
                'param_key': param_key,
                'alg_id': alg_id,
                'alg_class': alg_info['class'],
                'global_config': current_global_config,
                'alg_specific_config': alg_info['config'],
                'num_runs': num_runs_per_config
            }
            tasks.append(task)
            
    # --- 4. 使用多进程并行执行所有任务 ---
    print(f"总共将执行 {len(tasks)} 个仿真任务...")
    # 限制最大进程数，避免资源耗尽
    num_processes = min(multiprocessing.cpu_count(), 8) 
    
    results = {}
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用tqdm显示进度条
        pbar = tqdm(total=len(tasks), desc="执行仿真任务", unit="task")
        
        # 异步提交所有任务
        async_results = [pool.apply_async(run_single_simulation_task, (task,)) for task in tasks]
        
        # 收集结果
        for i, res in enumerate(async_results):
            task_result = res.get() # 等待任务完成并获取结果
            if task_result:
                param_key = tasks[i]['param_key']
                alg_id = tasks[i]['alg_id']
                if param_key not in results:
                    results[param_key] = {}
                results[param_key][alg_id] = task_result
            pbar.update(1)
        pbar.close()

    # --- 5. 格式化并打印最终的对比结果 ---
    print("\n" + "="*70)
    print(" " * 20 + "缺失标签识别算法性能对比结果")
    print("="*70)
    print(f"\n固定参数: 总标签数 N = {base_global_config['TOTAL_TAGS']}")

    metrics_to_display = {
        'total_protocol_time_us': {'name': '总执行时间 (秒)', 'format': '.4f'},
        'total_bits': {'name': '总通信开销 (Kbits)', 'format': '.1f'},
        'f1_score': {'name': 'F1 分数', 'format': '.4f'}
    }
    
    # 对结果进行预处理，方便表格显示
    for param_key, alg_results in results.items():
        for alg_name, res_dict in alg_results.items():
            # 将时间从微秒转换为秒
            if 'total_protocol_time_us' in res_dict:
                 res_dict['total_protocol_time_us'] /= 1e6
            # 将比特转换为千比特
            if 'total_reader_bits' in res_dict and 'total_tag_bits' in res_dict:
                res_dict['total_bits'] = (res_dict['total_reader_bits'] + res_dict['total_tag_bits']) / 1024.0

    create_comparative_table(results, "丢失率", metrics_to_display)

    print("\n" + "="*70)
    print("仿真框架执行完毕。")

