# -*- coding: utf-8 -*-

import random
import time
import numpy as np
from typing import Dict, List, Set, Tuple

# ==============================================================================
# 0. 仿真配置和常量 (已重构)
# ==============================================================================

# --- 场景配置 ---
DEFAULT_TOTAL_TAGS = 5000
DEFAULT_MISSING_RATE = 0.01
DEFAULT_BINARY_LENGTH = 96

# --- 统一物理层和链路层参数 ---
DEFAULT_DATA_RATE_BPS = 160000.0
DEFAULT_BITS_PER_MICROSECOND = DEFAULT_DATA_RATE_BPS / 1.0e6
DEFAULT_T1_RTcal = 25.0          # 读写器 -> 标签 周转时间
DEFAULT_T2_TRcal = 25.0          # 标签 -> 读写器 周转时间
DEFAULT_MINIMAL_GUARD_TIME_US = 50.0 # !!新!!: 所有操作共享的最小物理保护间隔
DEFAULT_READER_CMD_BASE_BITS = 37  # 读写器命令的基础比特数
DEFAULT_TAG_SHORT_RESP_BITS = 1    # 标签短响应的比特数

# ==============================================================================
# 1. 通用类定义 (AlgorithmStepResult 已修改)
# ==============================================================================

class Tag:
    """表示一个RFID标签及其在盘点场景中的状态。"""
    def __init__(self, identityCode: str, is_present: bool = True):
        self.id: str = identityCode
        self.is_present: bool = is_present
        self.status_by_algo: str = 'UNKNOWN'

    def __repr__(self):
        id_short = self.id[-6:] if len(self.id) > 6 else self.id
        ground_truth = "在场" if self.is_present else "丢失"
        return f"Tag(...{id_short}, 真实状态={ground_truth}, 算法判断={self.status_by_algo})"

class AlgorithmStepResult:
    """
    封装单次算法步骤的结果。
    !!已修改!!: 不再包含time_delta_us，而是返回操作的原始信息。
    """
    def __init__(self,
                 operation_type: str = 'idle', # e.g., 'PROBE', 'POLL', 'ALOHA_SLOT'
                 reader_bits: float = 0.0,
                 tag_bits: float = 0.0,
                 operation_description: str = 'idle'):
        self.operation_type: str = operation_type
        self.reader_bits: float = reader_bits
        self.tag_bits: float = tag_bits
        self.operation_description: str = operation_description

# ==============================================================================
# 2. 算法接口定义
# ==============================================================================

class MissingTagAlgorithmInterface:
    """定义所有缺失标签识别算法必须实现的接口。"""
    def __init__(self, config: Dict):
        self.config: Dict = config
        self.expected_tags_db: List[Tag] = []

    def initialize(self, expected_tags: List[Tag]):
        raise NotImplementedError("算法必须实现 initialize 方法。")

    def perform_step(self) -> AlgorithmStepResult:
        raise NotImplementedError("算法必须实现 perform_step 方法。")

    def is_finished(self) -> bool:
        raise NotImplementedError("算法必须实现 is_finished 方法。")

    def get_results(self) -> Tuple[Set[str], Set[str]]:
        raise NotImplementedError("算法必须实现 get_results 方法。")

# ==============================================================================
# 3. 核心工具函数 (新增时间计算器)
# ==============================================================================

def random_id(length: int) -> str:
    """生成指定长度的随机二进制字符串ID。"""
    return ''.join(random.choice('01') for _ in range(length))

def generate_scenario(config: Dict) -> List[Tag]:
    """根据配置生成仿真场景。"""
    # ... 此函数逻辑无变化 ...
    total_tags = config.get('TOTAL_TAGS', DEFAULT_TOTAL_TAGS)
    missing_rate = config.get('MISSING_RATE', DEFAULT_MISSING_RATE)
    binary_length = config.get('BINARY_LENGTH', DEFAULT_BINARY_LENGTH)
    num_missing = int(total_tags * missing_rate)
    id_set = set()
    while len(id_set) < total_tags:
        id_set.add(random_id(binary_length))
    all_ids = list(id_set)
    random.shuffle(all_ids)
    scenario_tags = [Tag(all_ids[i], is_present=(i < total_tags - num_missing)) for i in range(total_tags)]
    print(f"场景生成完毕: 总标签数={total_tags}, 其中在场={total_tags - num_missing}, 丢失={num_missing}")
    return scenario_tags

def calculate_time_delta(step_result: AlgorithmStepResult, config: Dict) -> float:
    """
    !!新!!: 统一的时间计算器。
    根据原子操作的比特信息和统一的物理参数计算时间开销。
    """
    if step_result.operation_type == 'idle' or step_result.reader_bits == 0:
        return 0.0

    bits_per_us = config.get('BITS_PER_MICROSECOND', DEFAULT_BITS_PER_MICROSECOND)
    t1 = config.get('T1_RTcal', DEFAULT_T1_RTcal)
    t2 = config.get('T2_TRcal', DEFAULT_T2_TRcal)
    guard_time = config.get('MINIMAL_GUARD_TIME_US', DEFAULT_MINIMAL_GUARD_TIME_US)

    # 1. 读写器发送时间
    time_reader_tx = step_result.reader_bits / bits_per_us
    
    # 2. 标签响应时间（只有当有响应时才计算）
    time_tag_tx = 0
    if step_result.tag_bits > 0:
        time_tag_tx = step_result.tag_bits / bits_per_us
        # 完整的读写周期才需要 T1 和 T2
        total_time = time_reader_tx + t1 + time_tag_tx + t2 + guard_time
    else:
        # 如果没有标签响应（如空闲时隙），我们认为读写器至少会等待一个 T1 的时间来确认无响应
        total_time = time_reader_tx + t1 + guard_time

    return total_time

# ==============================================================================
# 4. 仿真核心逻辑 (已重构)
# ==============================================================================

def run_missing_tag_simulation(
    global_config: Dict,
    algorithm_class,
    algorithm_specific_config: Dict
) -> Dict:
    """
    缺失标签识别仿真框架核心函数 (V2 - 采用动态时间计算)。
    """
    # ... 结果字典和场景生成部分无重大变化 ...
    result_dict = { 'algorithm_name': algorithm_class.__name__, 'total_protocol_time_us': 0.0, 'total_reader_bits': 0.0, 'total_tag_bits': 0.0 }
    
    scenario_tags = generate_scenario(global_config)
    ground_truth_present_ids = {t.id for t in scenario_tags if t.is_present}
    ground_truth_missing_ids = {t.id for t in scenario_tags if not t.is_present}

    effective_algo_config = {**global_config, **algorithm_specific_config}
    algo_instance = algorithm_class(effective_algo_config)
    algo_instance.initialize(scenario_tags)

    print(f"开始仿真 [{algo_instance.__class__.__name__}] (采用动态时间模型)...")

    # 主仿真循环 (已重构)
    current_protocol_time_us = 0.0
    step_count = 0
    max_steps = global_config.get('TOTAL_TAGS', DEFAULT_TOTAL_TAGS) * 200 # 安全阈值

    while not algo_instance.is_finished():
        step_result = algo_instance.perform_step()

        # !!核心修改!!: 调用统一的时间计算器
        time_delta = calculate_time_delta(step_result, effective_algo_config)
        
        current_protocol_time_us += time_delta
        result_dict['total_reader_bits'] += step_result.reader_bits
        result_dict['total_tag_bits'] += step_result.tag_bits
        step_count += 1
        
        if step_count > max_steps and max_steps > 0:
            print(f"错误: 仿真步骤过多 ({step_count})，可能存在无限循环。强制终止。")
            break

    result_dict['total_protocol_time_us'] = current_protocol_time_us
    print(f"仿真结束 [{algo_instance.__class__.__name__}]. 协议总耗时: {current_protocol_time_us / 1e6:.4f} s")

    # ... 结果统计部分无重大变化 ...
    predicted_present_ids, predicted_missing_ids = algo_instance.get_results()
    tp = len(predicted_missing_ids.intersection(ground_truth_missing_ids))
    fp = len(predicted_missing_ids.intersection(ground_truth_present_ids))
    fn = len(ground_truth_missing_ids.difference(predicted_missing_ids))
    # ... 其他统计指标计算
    result_dict.update({'true_positives': tp, 'false_positives': fp, 'false_negatives': fn})
    return result_dict

# ==============================================================================
# 5. 示例算法实现 (已适配新框架)
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

        # 核心修改：只计算并返回比特数，不再计算时间
        reader_bits = self.config['BINARY_LENGTH'] + self.config.get('READER_CMD_BASE_BITS', DEFAULT_READER_CMD_BASE_BITS)
        tag_bits = 0
        op_desc = ""

        if tag_to_poll.is_present:
            # 轮询一个存在的标签，期望收到完整的ID作为响应
            tag_bits = self.config['BINARY_LENGTH']
            self.found_present_ids.add(tag_to_poll.id)
            op_desc = f"轮询 {tag_to_poll.id[-6:]}... 成功响应"
        else:
            # 轮询一个丢失的标签，无响应
            tag_bits = 0
            self.found_missing_ids.add(tag_to_poll.id)
            op_desc = f"轮询 {tag_to_poll.id[-6:]}... 无响应 (丢失)"

        return AlgorithmStepResult(
            operation_type='POLL',
            reader_bits=reader_bits,
            tag_bits=tag_bits,
            operation_description=op_desc
        )

# ==============================================================================
# 6. 主程序入口
# ==============================================================================
def main():
    """主函数，用于演示如何使用新的仿真框架。"""
    
    # 1. 定义全局配置 (已移除 T_s, T_l, T_tag)
    global_config = {
        'TOTAL_TAGS': 2000, # 减少数量以便快速演示
        'MISSING_RATE': 0.1,
        'BINARY_LENGTH': 96,
        'DATA_RATE_BPS': 160000.0,
        'BITS_PER_MICROSECOND': 160000.0 / 1.0e6,
        'T1_RTcal': 25.0,
        'T2_TRcal': 25.0,
        'MINIMAL_GUARD_TIME_US': 50.0, # 统一的最小保护间隔
        'READER_CMD_BASE_BITS': 37,
        'TAG_SHORT_RESP_BITS': 1,
    }

    # 2. 运行仿真
    baseline_results = run_missing_tag_simulation(
        global_config=global_config,
        algorithm_class=BaselinePollingAlgo,
        algorithm_specific_config={}
    )
    
    # 在这里，您可以运行您自己适配后的HPVT算法
    # hpvt_results = run_missing_tag_simulation(...)

    # 3. 打印结果
    print_results(baseline_results)


def print_results(results: dict):
    """一个辅助函数，用于格式化并打印单次仿真的结果。"""
    # ... 此函数逻辑无变化 ...
    print(f"\n--- 结果汇总: {results.get('algorithm_name', 'N/A')} ---")
    print(f"协议总耗时: {results.get('total_protocol_time_us', 0) / 1e6:.4f} 秒")
    print(f"通信总开销 (bits): 读写器={results.get('total_reader_bits', 0):.0f}, 标签={results.get('total_tag_bits', 0):.0f}")
    print(f"TP (正确找到丢失): {results.get('true_positives', 0)}")
    print(f"FP (将在场误报为丢失): {results.get('false_positives', 0)}")


if __name__ == '__main__':
    main()

