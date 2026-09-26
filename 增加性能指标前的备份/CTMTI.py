# -*- coding: utf-8 -*-
"""
此文件包含了适配了新的动态时间计算框架的 CTMTI 算法，
并提供了一个独立的主程序入口来运行和测试其性能。
"""
import math
import random
import numpy as np
from collections import deque
from typing import List, Set, Tuple, Dict

# ==============================================================================
# 0. 从框架导入 (或在此处定义以确保独立运行)
# ==============================================================================
# from Framework import MissingTagAlgorithmInterface, AlgorithmStepResult, Tag
# from Framework import run_missing_tag_simulation, print_results, BaselinePollingAlgo
# from Framework import calculate_time_delta, generate_scenario

from Framework import *

# ==============================================================================
# 1. CTMTI 算法实现 (已适配新框架)
# ==============================================================================
class CTMTIAlgo(MissingTagAlgorithmInterface):
    """
    复现论文 "CTMTI" 的算法，已适配动态时间计算框架。
    """
    def initialize(self, expected_tags: List[Tag]):
        """初始化算法状态。"""
        self.expected_tags_db = expected_tags
        self.unidentified_tags = list(expected_tags)
        self.predicted_present_ids = set()
        self.predicted_missing_ids = set()
        self.is_first_round = True
        self.irresolvable_groups = []

    def is_finished(self) -> bool:
        """当不再是第一轮且没有未解决的碰撞组时，算法结束。"""
        return not self.is_first_round and not self.irresolvable_groups

    def get_results(self) -> Tuple[Set[str], Set[str]]:
        """返回当前已识别的集合。最终的清理在仿真器端完成。"""
        # 将最后所有未识别的标签都归为丢失
        all_ids_in_db = {t.id for t in self.expected_tags_db}
        identified_ids = self.predicted_present_ids.union(self.predicted_missing_ids)
        unconfirmed_ids = all_ids_in_db - identified_ids
        if unconfirmed_ids:
            self.predicted_missing_ids.update(unconfirmed_ids)
        return self.predicted_present_ids, self.predicted_missing_ids

    def perform_step(self) -> AlgorithmStepResult:
        """执行一个完整的 CTMTI 轮次，并返回其总通信开销。"""
        if self.is_finished():
            return AlgorithmStepResult(operation_type="finished")

        # --- 1. 设置本轮待处理的标签和哈希空间 ---
        R1 = random.random()
        groups: Dict[int, List[Tag]] = {}
        num_elements = 0

        if self.is_first_round:
            # 第一轮：处理所有未识别标签
            tags_to_process = self.unidentified_tags
            # 根据论文，alpha是估计的在场率，但此处我们用总数乘以一个系数
            num_elements = int(self.config['TOTAL_TAGS'] * self.config.get('alpha', 0.54))
            if num_elements == 0: num_elements = 1
            groups = {i: [] for i in range(num_elements)}
            for tag in tags_to_process:
                h = hash((tag.id, R1, "Vp_hash")) % num_elements
                groups[h].append(tag)
        else:
            # 后续轮次：对上一轮未解决的碰撞组进行B-ary树分裂
            B = self.config.get('B', 2)
            num_elements = len(self.irresolvable_groups) * B
            if num_elements == 0:
                return AlgorithmStepResult(operation_type="finished")
            groups = {i: [] for i in range(num_elements)}
            group_idx_offset = 0
            for tag_group in self.irresolvable_groups:
                for tag in tag_group:
                    h = hash((tag.id, R1, "B_ary_split")) % B
                    groups[group_idx_offset + h].append(tag)
                group_idx_offset += B

        # --- 2. 碰撞调节 (Va) 和预期响应计算 ---
        R2 = random.random()
        new_irresolvable_groups = []
        expected_responses: Dict[int, Tag] = {}
        X_01_total = 0
        X_10_count = 0
        
        for i in range(num_elements):
            tags_in_elem = groups.get(i, [])
            if len(tags_in_elem) == 1:
                # 单标签元素 (01)
                tag = tags_in_elem[0]
                response_pos = X_01_total
                expected_responses[response_pos] = tag
                X_01_total += 1
            elif len(tags_in_elem) > 1:
                # 多标签元素，尝试进行3-ary子分裂
                sub_groups = {0: [], 1: [], 2: []}
                for tag in tags_in_elem:
                    h_sub = hash((tag.id, R2, "Va_hash")) % 3
                    sub_groups[h_sub].append(tag)
                
                if all(len(v) <= 1 for v in sub_groups.values()):
                    # 可解碰撞 (10)
                    for h_sub, sub_group_tags in sub_groups.items():
                        if sub_group_tags:
                            tag = sub_group_tags[0]
                            response_pos = X_01_total + 3 * X_10_count + h_sub
                            expected_responses[response_pos] = tag
                    X_10_count += 1
                else:
                    # 不可解碰撞 (11)
                    new_irresolvable_groups.append(tags_in_elem)

        # --- 3. 仿真标签响应并识别 ---
        response_len = X_01_total + 3 * X_10_count
        
        # 统计在场标签的响应
        actual_responses_count = 0
        for tag in expected_responses.values():
            if tag.is_present:
                actual_responses_count += 1
        
        # 根据预期和实际响应进行判断
        for pos, tag in expected_responses.items():
            # 此处简化：我们直接知道哪些标签在场，所以能直接判断
            if tag.is_present:
                self.predicted_present_ids.add(tag.id)
            else:
                self.predicted_missing_ids.add(tag.id)
        
        # --- 4. 核心修改：计算本轮的通信比特开销 ---
        # 论文中定义的读写器发送比特：控制参数 + Ve向量(2*m bits)
        # Ve 向量由 (00, 01, 10, 11) 四元组构成，每个元素2比特
        reader_bits = self.config.get('READER_CMD_BASE_BITS', 37) + (2.0 * num_elements)
        # 标签响应比特：一个长度为 response_len 的比特串
        tag_bits = float(response_len)

        # --- 5. 更新状态进入下一轮 ---
        self.is_first_round = False
        self.irresolvable_groups = new_irresolvable_groups
        self.unidentified_tags = [tag for group in self.irresolvable_groups for tag in group]

        return AlgorithmStepResult(
            operation_type='CTMTI_ROUND',
            reader_bits=reader_bits,
            tag_bits=tag_bits,
            operation_description=f"CTMTI Round with {num_elements} elements, {len(self.unidentified_tags)} tags remain"
        )

# ==============================================================================
# 5. 主程序入口
# ==============================================================================

if __name__ == '__main__':
    # --- 1. 定义全局仿真配置 ---
    global_config = {
        'TOTAL_TAGS': 20000,
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

    # --- 2. 定义 CTMTI 算法的特定参数 ---
    ctmti_specific_config = {
        'alpha': 0.54,  # 根据论文，用于第一轮哈希空间大小的估计系数
        'B': 2,         # B-ary 树的分裂因子
    }

    # --- 3. 运行 CTMTI 仿真 ---
    print("\n--- 运行 CTMTI 算法 (动态时间模型) ---")
    ctmti_results = run_missing_tag_simulation(
        global_config=global_config,
        algorithm_class=CTMTIAlgo,
        algorithm_specific_config=ctmti_specific_config
    )
    print_results(ctmti_results)

    # --- 4. (可选) 运行基线算法进行对比 ---
    print("\n\n--- 运行基线轮询算法作为对比 ---")
    baseline_results = run_missing_tag_simulation(
        global_config=global_config,
        algorithm_class=BaselinePollingAlgo,
        algorithm_specific_config={}
    )
    print_results(baseline_results)
