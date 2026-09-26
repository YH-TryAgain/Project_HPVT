# -*- coding: utf-8 -*-
"""
一个公平、可扩展的RFID缺失标签识别仿真框架。

此版本对 run_missing_tag_simulation 函数进行了重大升级，使其能够
在仿真循环的每一步中追踪识别进度，从而计算出如“首次发现时间”
和“发现过程时间线”等高级过程质量指标。
"""
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

# ... 顶部的 dataclass, CONSTANTS, Tag, AlgorithmStepResult, MissingTagAlgorithmInterface 等保持不变 ...
@dataclass(frozen=True)
class SimulationConstants:
    DATA_RATE_BPS: float = 256000.0
    T1_RTcal: float = 25.0
    T2_TRcal: float = 25.0
    MINIMAL_GUARD_TIME_US: float = 50.0
    READER_CMD_BASE_BITS: int = 37
    TAG_SHORT_RESP_BITS: int = 1
    @property
    def BITS_PER_MICROSECOND(self) -> float:
        return self.DATA_RATE_BPS / 1.0e6
CONSTANTS = SimulationConstants()

class Tag:
    def __init__(self, identityCode: str, is_present: bool = True):
        self.id: str = identityCode
        self.is_present: bool = is_present

class AlgorithmStepResult:
    def __init__(self, operation_type: str = 'idle', reader_bits: float = 0.0, tag_bits: float = 0.0, operation_description: str = 'idle'):
        self.operation_type, self.reader_bits, self.tag_bits, self.operation_description = operation_type, reader_bits, tag_bits, operation_description

class MissingTagAlgorithmInterface:
    def __init__(self, **kwargs): pass
    def initialize(self, expected_tags: List[Tag]): self.expected_tags_db = list(expected_tags)
    def perform_step(self) -> AlgorithmStepResult: raise NotImplementedError
    def is_finished(self) -> bool: raise NotImplementedError
    def get_results(self) -> Tuple[Set[str], Set[str]]: raise NotImplementedError

def generate_scenario(scenario_config: Dict) -> List[Tag]:
    total_tags, missing_rate, binary_length = scenario_config.get('TOTAL_TAGS', 1000), scenario_config.get('MISSING_RATE', 0.1), scenario_config.get('BINARY_LENGTH', 96)
    num_missing = int(total_tags * missing_rate)
    id_set = set()
    while len(id_set) < total_tags: id_set.add(''.join(random.choice('01') for _ in range(binary_length)))
    all_ids = list(id_set)
    random.shuffle(all_ids)
    return [Tag(all_ids[i], is_present=(i < total_tags - num_missing)) for i in range(total_tags)]

def calculate_time_delta(step_result: AlgorithmStepResult) -> float:
    if step_result.operation_type == 'idle': return 0.0
    bits_per_us, t1, t2, guard_time = CONSTANTS.BITS_PER_MICROSECOND, CONSTANTS.T1_RTcal, CONSTANTS.T2_TRcal, CONSTANTS.MINIMAL_GUARD_TIME_US
    if step_result.operation_type == 'ALOHA_SLOT': return CONSTANTS.TAG_SHORT_RESP_BITS / bits_per_us + t2 + guard_time
    if step_result.reader_bits == 0: return 0.0
    time_reader_tx = step_result.reader_bits / bits_per_us
    if step_result.tag_bits > 0: return time_reader_tx + t1 + (step_result.tag_bits / bits_per_us) + t2 + guard_time
    else: return time_reader_tx + t1 + guard_time

# ==============================================================================
# 4. 【核心升级】仿真核心逻辑
# ==============================================================================
def run_missing_tag_simulation(
    scenario_config: Dict,
    algorithm_class,
    algorithm_specific_config: Dict
) -> Dict:
    """
    【已升级】仿真核心函数，现在会追踪并返回更丰富的过程质量指标。
    """
    # 1. 初始化结果字典，包含过程质量指标的占位符
    result_dict = {
        'total_protocol_time_us': 0.0,
        'total_reader_bits': 0.0,
        'total_tag_bits': 0.0,
        'time_to_first_missing_us': -1.0,  # -1 表示尚未发现
        'discovery_timeline_us': [],       # 存储发现过程的时间点
    }

    # 2. 场景生成与算法初始化
    scenario_tags = generate_scenario(scenario_config)
    ground_truth_missing_ids = {t.id for t in scenario_tags if not t.is_present}
    num_total_missing = len(ground_truth_missing_ids)

    algo_instance = algorithm_class(**algorithm_specific_config)
    algo_instance.initialize(scenario_tags)

    # 3. 初始化过程质量指标的追踪器
    # 创建发现过程的里程碑（0%, 10%, ..., 100%）
    milestone_percentages = np.linspace(0, 1, 11)
    if num_total_missing > 0:
        discovery_milestones_counts = [int(p * num_total_missing) for p in milestone_percentages]
        discovery_milestones_counts[-1] = num_total_missing # 确保最后一个里程碑是总数
    else:
        # 如果没有丢失标签，所有里程碑都是0
        discovery_milestones_counts = [0] * 11
    current_milestone_index = 0
    discovery_timeline_us = []

    # 4. 主仿真循环
    current_protocol_time_us = 0.0
    max_steps = scenario_config.get('TOTAL_TAGS', 1) * 200
    step_count = 0
    
    while not algo_instance.is_finished():
        step_result = algo_instance.perform_step()
        time_delta = calculate_time_delta(step_result)
        current_protocol_time_us += time_delta
        result_dict['total_reader_bits'] += step_result.reader_bits
        result_dict['total_tag_bits'] += step_result.tag_bits
        step_count += 1
        
        # --- !!新逻辑!!: 在每步之后检查和更新过程指标数据 ---
        if num_total_missing > 0:
            # 获取算法当前的识别结果 (这里的调用必须高效)
            _, predicted_missing = algo_instance.get_results()
            correctly_identified_missing = predicted_missing.intersection(ground_truth_missing_ids)
            identified_count = len(correctly_identified_missing)
            
            # 4.1 记录首次发现时间 (只记录一次)
            if result_dict['time_to_first_missing_us'] < 0 and identified_count > 0:
                result_dict['time_to_first_missing_us'] = current_protocol_time_us

            # 4.2 记录发现过程时间线
            while (current_milestone_index < len(discovery_milestones_counts) and
                   identified_count >= discovery_milestones_counts[current_milestone_index]):
                discovery_timeline_us.append(current_protocol_time_us)
                current_milestone_index += 1
        
        if max_steps > 0 and step_count > max_steps:
            print(f"错误: 仿真步骤过多 ({step_count})，强制终止。")
            break

    # 5. 仿真结束后，进行最终的数据整理和计算
    result_dict['total_protocol_time_us'] = current_protocol_time_us
    
    # 补全时间线，确保其长度为11。对于提前完成的情况，后续里程碑时间等于总时间。
    while len(discovery_timeline_us) < len(milestone_percentages):
        discovery_timeline_us.append(current_protocol_time_us)
    result_dict['discovery_timeline_us'] = discovery_timeline_us
    
    # 验证最终准确性
    _, predicted_missing_ids = algo_instance.get_results()
    tp = len(predicted_missing_ids.intersection(ground_truth_missing_ids))
    result_dict.update({'true_positives': tp, 'false_positives': len(predicted_missing_ids) - tp, 'false_negatives': len(ground_truth_missing_ids) - tp})

    # 从算法实例中提取自定义的机制性指标
    if hasattr(algo_instance, 'metrics'):
        result_dict.update(algo_instance.metrics)
        
    return result_dict
