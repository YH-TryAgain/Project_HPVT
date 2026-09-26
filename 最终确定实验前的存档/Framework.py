# -*- coding: utf-8 -*-
"""
一个公平、可扩展的RFID缺失标签识别仿真框架 (V3)。
此版本在 run_missing_tag_simulation 中集成了对
TFM (首次发现时间), Discovery Timeline, 和 LSSR (后期迟滞比) 的计算。
"""

import random
import time
import numpy as np
from typing import Dict, List, Set, Tuple

# ==============================================================================
# 0. 仿真配置和常量
# ==============================================================================

# --- 场景配置 ---
DEFAULT_TOTAL_TAGS = 5000
DEFAULT_MISSING_RATE = 0.01
DEFAULT_BINARY_LENGTH = 96

# --- 统一物理层和链路层参数 ---
DEFAULT_DATA_RATE_BPS = 160000.0
DEFAULT_BITS_PER_MICROSECOND = DEFAULT_DATA_RATE_BPS / 1.0e6
DEFAULT_T1_RTcal = 25.0
DEFAULT_T2_TRcal = 25.0
DEFAULT_MINIMAL_GUARD_TIME_US = 50.0
DEFAULT_READER_CMD_BASE_BITS = 37
DEFAULT_TAG_SHORT_RESP_BITS = 1

# ==============================================================================
# 1. 通用类定义
# ==============================================================================

class Tag:
    """表示一个RFID标签及其在盘点场景中的状态。"""
    def __init__(self, identityCode: str, is_present: bool = True):
        self.id: str = identityCode
        self.is_present: bool = is_present

class AlgorithmStepResult:
    """封装单次算法步骤的结果，描述其通信开销。"""
    def __init__(self,
                 operation_type: str = 'idle',
                 reader_bits: float = 0.0,
                 tag_bits: float = 0.0,
                 operation_description: str = 'idle'):
        self.operation_type = operation_type
        self.reader_bits = reader_bits
        self.tag_bits = tag_bits
        self.operation_description = operation_description

# ==============================================================================
# 2. 算法接口定义
# ==============================================================================

class MissingTagAlgorithmInterface:
    """定义所有缺失标签识别算法必须实现的接口。"""
    def __init__(self, config: Dict):
        self.config: Dict = config
        self.expected_tags_db: List[Tag] = []

    def initialize(self, expected_tags: List[Tag]):
        raise NotImplementedError
    def perform_step(self) -> AlgorithmStepResult:
        raise NotImplementedError
    def is_finished(self) -> bool:
        raise NotImplementedError
    def get_results(self) -> Tuple[Set[str], Set[str]]:
        raise NotImplementedError

# ==============================================================================
# 3. 核心工具函数
# ==============================================================================

def generate_scenario(config: Dict) -> List[Tag]:
    """根据配置生成仿真场景。"""
    total_tags = config.get('TOTAL_TAGS', DEFAULT_TOTAL_TAGS)
    missing_rate = config.get('MISSING_RATE', DEFAULT_MISSING_RATE)
    binary_length = config.get('BINARY_LENGTH', DEFAULT_BINARY_LENGTH)
    num_missing = int(total_tags * missing_rate)
    id_set = set()
    while len(id_set) < total_tags:
        id_set.add(''.join(random.choice('01') for _ in range(binary_length)))
    all_ids = list(id_set)
    random.shuffle(all_ids)
    scenario_tags = [Tag(all_ids[i], is_present=(i < total_tags - num_missing)) for i in range(total_tags)]
    # print(f"场景生成完毕: 总标签数={total_tags}, 其中在场={total_tags - num_missing}, 丢失={num_missing}")
    return scenario_tags

def calculate_time_delta(step_result: AlgorithmStepResult, config: Dict) -> float:
    """
    !!已校正!!: 统一的时间计算器，能够正确处理ALOHA时隙的耗时。
    """
    if step_result.operation_type == 'idle':
        return 0.0

    bits_per_us = config['BITS_PER_MICROSECOND']
    t1 = config['T1_RTcal']
    t2 = config['T2_TRcal']
    guard_time = config['MINIMAL_GUARD_TIME_US']
    
    # --- 核心修正 ---
    # 为 ALOHA 时隙定义一个独立且正确的计算逻辑。
    # 一个ALOHA时隙的时间，是读写器等待一个潜在短响应所需的时间。
    if step_result.operation_type == 'ALOHA_SLOT':
        short_resp_bits = config.get('TAG_SHORT_RESP_BITS', 1)
        time_tag_response = short_resp_bits / bits_per_us
        # 即使时隙为空，也必须消耗掉这段“等待”时间。
        return time_tag_response + t2 + guard_time
    
    # --- 其他操作的时间计算逻辑保持不变 ---
    if step_result.reader_bits == 0:
        return 0.0

    time_reader_tx = step_result.reader_bits / bits_per_us
    
    if step_result.tag_bits > 0:
        time_tag_tx = step_result.tag_bits / bits_per_us
        return time_reader_tx + t1 + time_tag_tx + t2 + guard_time
    else:
        # 读写器发送后，至少等待T1时间来确认无响应
        return time_reader_tx + t1 + guard_time

# ==============================================================================
# 4. 仿真核心逻辑 (已修改)
# ==============================================================================

def run_missing_tag_simulation(
    global_config: Dict,
    algorithm_class,
    algorithm_specific_config: Dict
) -> Dict:
    """
    缺失标签识别仿真框架核心函数 (V3 - 集成高级指标)。
    """
    # 1. 初始化结果字典，包含新指标的占位符
    result_dict = {
        'algorithm_name': algorithm_class.__name__,
        'total_protocol_time_us': 0.0,
        'total_reader_bits': 0.0,
        'total_tag_bits': 0.0,
        'true_positives': 0, 'false_positives': 0, 'false_negatives': 0, 'true_negatives': 0,
        'time_to_first_missing': -1.0,  # -1 表示尚未发现
        'discovery_timeline': [],       # 存储发现过程的时间点
        'lssr': 0.0,                    # 后期迟滞比
    }
    
    # 2. 生成场景并初始化算法
    scenario_tags = generate_scenario(global_config)
    ground_truth_present_ids = {t.id for t in scenario_tags if t.is_present}
    ground_truth_missing_ids = {t.id for t in scenario_tags if not t.is_present}
    num_total_missing = len(ground_truth_missing_ids)

    effective_algo_config = {**global_config, **algorithm_specific_config}
    algo_instance = algorithm_class(effective_algo_config)
    algo_instance.initialize(scenario_tags)

    # print(f"开始仿真 [{algo_instance.__class__.__name__}]...")

    # 3. 初始化新指标的追踪器
    time_to_first_missing = -1.0
    discovery_timeline = []
    
    # 创建发现过程的里程碑（0%, 10%, ..., 100%）
    milestone_percentages = np.linspace(0.0, 1.0, 11)
    if num_total_missing > 0:
        discovery_milestones = [int(p * num_total_missing) for p in milestone_percentages]
        discovery_milestones[-1] = num_total_missing # 确保最后一个里程碑是总数
    else:
        discovery_milestones = [0] * 11 # 如果没有丢失标签，所有里程碑都是0
    current_milestone_index = 0

    # 4. 主仿真循环
    current_protocol_time_us = 0.0
    step_count = 0
    max_steps = global_config.get('TOTAL_TAGS', 1) * 200

    while not algo_instance.is_finished():
        step_result = algo_instance.perform_step()
        time_delta = calculate_time_delta(step_result, effective_algo_config)
        
        current_protocol_time_us += time_delta
        result_dict['total_reader_bits'] += step_result.reader_bits
        result_dict['total_tag_bits'] += step_result.tag_bits
        step_count += 1
        
        # --- !!新逻辑!!: 在每步之后检查和更新指标数据 ---
        
        # 仅当有丢失标签时，才进行进度追踪
        if num_total_missing > 0:
            # 获取算法当前的识别结果
            _, predicted_missing = algo_instance.get_results()
            correctly_identified_missing = predicted_missing.intersection(ground_truth_missing_ids)
            identified_count = len(correctly_identified_missing)
            
            # 4.1 记录首次发现时间 (只记录一次)
            if time_to_first_missing < 0 and identified_count > 0:
                time_to_first_missing = current_protocol_time_us

            # 4.2 记录发现过程时间线
            while (current_milestone_index < len(discovery_milestones) and
                   identified_count >= discovery_milestones[current_milestone_index]):
                discovery_timeline.append(current_protocol_time_us)
                current_milestone_index += 1
        
        if max_steps > 0 and step_count > max_steps:
            print(f"错误: 仿真步骤过多 ({step_count})，可能存在无限循环。强制终止。")
            break

    # 5. 仿真结束后，进行最终的数据整理和计算
    result_dict['total_protocol_time_us'] = current_protocol_time_us
    result_dict['time_to_first_missing'] = time_to_first_missing
    
    # 补全时间线，确保其长度为11
    while len(discovery_timeline) < len(milestone_percentages):
        discovery_timeline.append(current_protocol_time_us)
    result_dict['discovery_timeline'] = discovery_timeline

    # 计算 LSSR
    if len(discovery_timeline) >= 11:
        # 时间从 0% -> 10%
        time_first_10_percent = discovery_timeline[1] - discovery_timeline[0]
        # 时间从 90% -> 100%
        time_last_10_percent = discovery_timeline[10] - discovery_timeline[9]
        
        if time_first_10_percent > 1e-9: # 避免除以零
            result_dict['lssr'] = time_last_10_percent / time_first_10_percent

    # print(f"仿真结束 [{algo_instance.__class__.__name__}]. 协议总耗时: {current_protocol_time_us / 1e6:.4f} s")

    # 6. 统计准确率等最终指标
    predicted_present_ids, predicted_missing_ids = algo_instance.get_results()
    tp = len(predicted_missing_ids.intersection(ground_truth_missing_ids))
    fp = len(predicted_missing_ids.intersection(ground_truth_present_ids))
    fn = len(ground_truth_missing_ids.difference(predicted_missing_ids))
    tn = len(ground_truth_present_ids.difference(predicted_present_ids)) # 新增TN计算
    result_dict.update({'true_positives': tp, 'false_positives': fp, 'false_negatives': fn, 'true_negatives': tn})
    
    return result_dict

# ==============================================================================
# 5. BaselinePollingAlgo 算法实现
# ==============================================================================
class BaselinePollingAlgo(MissingTagAlgorithmInterface):
    """一个简单的基线算法，已适配新的动态时间计算框架。"""
    def initialize(self, expected_tags: List[Tag]):
        self.tags_to_check = list(expected_tags)
        self.found_present_ids = set()
        self.found_missing_ids = set()

    def is_finished(self) -> bool:
        return not self.tags_to_check

    def get_results(self) -> Tuple[Set[str], Set[str]]:
        return self.found_present_ids, self.found_missing_ids

    def perform_step(self) -> AlgorithmStepResult:
        if self.is_finished():
            return AlgorithmStepResult(operation_type='finished')

        tag_to_poll = self.tags_to_check.pop(0)

        reader_bits = self.config['BINARY_LENGTH'] + self.config.get('READER_CMD_BASE_BITS', DEFAULT_READER_CMD_BASE_BITS)
        tag_bits = 0
        
        if tag_to_poll.is_present:
            tag_bits = self.config['BINARY_LENGTH']
            self.found_present_ids.add(tag_to_poll.id)
            op_desc = f"轮询 {tag_to_poll.id[-6:]}... 成功响应"
        else:
            self.found_missing_ids.add(tag_to_poll.id)
            op_desc = f"轮询 {tag_to_poll.id[-6:]}... 无响应 (丢失)"

        return AlgorithmStepResult(
            operation_type='POLL',
            reader_bits=reader_bits,
            tag_bits=tag_bits,
            operation_description=op_desc
        )

# ==============================================================================
# 6. 辅助函数和主程序入口
# ==============================================================================
def print_results(results: dict):
    """一个辅助函数，用于格式化并打印单次仿真的结果。"""
    print(f"\n--- 结果汇总: {results.get('algorithm_name', 'N/A')} ---")
    print(f"协议总耗时: {results.get('total_protocol_time_us', 0) / 1e6:.4f} 秒")
    print(f"通信总开销 (bits): 读写器={results.get('total_reader_bits', 0):.0f}, 标签={results.get('total_tag_bits', 0):.0f}")
    
    # !!已修改!!: 增加更详细的准确性评估输出
    print("-" * 25)
    print("准确性评估 (针对'丢失'标签):")
    print(f"  - TP (正确找到丢失): {results.get('true_positives', 0)}")
    print(f"  - FP (将在场误报为丢失): {results.get('false_positives', 0)}")
    print(f"  - FN (漏报的丢失标签): {results.get('false_negatives', 0)}")
    print("-" * 25)

    if results.get('time_to_first_missing', -1) >= 0:
        print(f"首次发现时间: {results['time_to_first_missing'] / 1e6:.4f} 秒")
    if results.get('lssr', 0) > 0:
        print(f"后期迟滞比 (LSSR): {results['lssr']:.2f}")

def main():
    """主函数，用于演示如何使用新的仿真框架。"""
    
    global_config = {
        'TOTAL_TAGS': 2000,
        'MISSING_RATE': 0.1,
        'BINARY_LENGTH': 96,
        'DATA_RATE_BPS': 160000.0,
        'BITS_PER_MICROSECOND': 160000.0 / 1.0e6,
        'T1_RTcal': 25.0,
        'T2_TRcal': 25.0,
        'MINIMAL_GUARD_TIME_US': 50.0,
        'READER_CMD_BASE_BITS': 37,
        'TAG_SHORT_RESP_BITS': 1,
    }

    baseline_results = run_missing_tag_simulation(
        global_config=global_config,
        algorithm_class=BaselinePollingAlgo,
        algorithm_specific_config={}
    )
    print(f"基础环境参数: 总标签数:{global_config['TOTAL_TAGS']},  丢失率:{global_config['MISSING_RATE']}")
    print_results(baseline_results)

if __name__ == '__main__':
    main()
