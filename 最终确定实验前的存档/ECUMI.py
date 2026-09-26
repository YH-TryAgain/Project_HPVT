# -*- coding: utf-8 -*-
"""
此文件包含了适配了新的动态时间计算框架的 CUMI 和 ECUMI 算法，
并提供了一个独立的主程序入口来运行和测试它们的性能。
"""
import math
import random
from typing import List, Set, Tuple, Dict

# 基础框架的引用
# 假设您的项目中存在一个 Framework.py 文件
from Framework import *


# ==============================================================================
# 1. CUMI 算法实现 (已适配新框架)
# ==============================================================================
class CUMIAlgo(MissingTagAlgorithmInterface):
    """
    实现了论文中的 CUMI 协议，已适配动态时间计算框架。
    """
    def initialize(self, expected_tags: List[Tag]):
        self.active_tags = list(expected_tags)
        self.identified_present_ids = set()
        self.identified_missing_ids = set()
        
        # 算法状态变量
        self._frame_in_progress = False
        self._current_slot = 0
        self._frame_size = 0
        self._slots_in_frame: List[List[Tag]] = []
        self._tags_identified_in_frame: Set[Tag] = set()
        self._seed = 0

    def is_finished(self) -> bool:
        return not self.active_tags and not self._frame_in_progress

    def get_results(self) -> Tuple[Set[str], Set[str]]:
        # 最后的清理：将所有未确认的标签都视为丢失
        if self.is_finished():
            all_ids = {t.id for t in self.active_tags}
            self.identified_missing_ids.update(all_ids)
            self.active_tags.clear()
        return self.identified_present_ids, self.identified_missing_ids

    def _start_new_frame(self) -> AlgorithmStepResult:
        """准备新帧，并返回广播指示器向量的通信开销。"""
        if not self.active_tags:
            return AlgorithmStepResult(operation_type="finished")

        self._frame_in_progress = True
        self._current_slot = 0
        self._tags_identified_in_frame.clear()
        self._seed = random.random()
        
        num_active_tags = len(self.active_tags)
        x = self.config.get('cumi_x', 3)
        self._frame_size = max(1, int(0.4263 * num_active_tags))
        
        self._slots_in_frame = [[] for _ in range(self._frame_size)]
        for tag in self.active_tags:
            slot_index = hash(tag.id + str(self._seed)) % self._frame_size
            self._slots_in_frame[slot_index].append(tag)
        
        # 核心修改：计算指示器向量的比特数并返回
        indicator_bits = self._frame_size * math.ceil(math.log2(x + 1))
        return AlgorithmStepResult(
            operation_type='CUMI_SETUP',
            reader_bits=indicator_bits,
            tag_bits=0,
            operation_description=f"CUMI Frame Start: N={num_active_tags}"
        )

    def perform_step(self) -> AlgorithmStepResult:
        """执行一个时隙的交互，并返回其通信开销。"""
        if not self._frame_in_progress:
            return self._start_new_frame()

        if self._current_slot >= self._frame_size:
            self.active_tags = [t for t in self.active_tags if t not in self._tags_identified_in_frame]
            self._frame_in_progress = False
            # 返回一个idle步骤，促使主循环检查is_finished或开始新帧
            return AlgorithmStepResult(operation_type="idle", operation_description="CUMI Frame End")

        tags_in_slot = self._slots_in_frame[self._current_slot]
        self._current_slot += 1
        j = len(tags_in_slot)
        x = self.config.get('cumi_x', 3)

        # 核心修改：所有时隙都有一个基础查询命令的开销
        reader_bits = self.config.get('READER_CMD_BASE_BITS', 37)
        tag_bits = 0

        # 判断时隙是否可用
        if not (1 <= j <= x):
            return AlgorithmStepResult('CUMI_SKIPPED_SLOT', reader_bits, 0, f"Slot {self._current_slot-1}: Skipped (j={j})")
        
        d_values = {hash(t.id + str(self._seed)) % j for t in tags_in_slot}
        if len(d_values) != j:
            return AlgorithmStepResult('CUMI_COLLISION_SLOT', reader_bits, 0, f"Slot {self._current_slot-1}: Collision (j={j})")
        
        # 可用时隙的标签响应比特数
        tag_bits = j
        
        # 识别逻辑
        actual_signal = 0
        for tag in tags_in_slot:
            if tag.is_present:
                d = hash(tag.id + str(self._seed)) % j
                actual_signal |= (1 << d)

        for tag in tags_in_slot:
            d = hash(tag.id + str(self._seed)) % j
            if (actual_signal >> d) & 1:
                self.identified_present_ids.add(tag.id)
            else:
                self.identified_missing_ids.add(tag.id)
            self._tags_identified_in_frame.add(tag)

        return AlgorithmStepResult('CUMI_USABLE_SLOT', reader_bits, tag_bits, f"Slot {self._current_slot-1}: Usable (j={j})")


# ==============================================================================
# 2. ECUMI 算法实现 (已适配新框架)
# ==============================================================================
class ECUMIAlgo(CUMIAlgo):
    """
    实现了论文中的 ECUMI 协议，继承自CUMI并修改了关键参数和逻辑。
    """
    def _start_new_frame(self) -> AlgorithmStepResult:
        if not self.active_tags:
            return AlgorithmStepResult(operation_type="finished")

        self._frame_in_progress = True
        self._current_slot = 0
        self._tags_identified_in_frame.clear()
        self._seed = random.random()

        num_active_tags = len(self.active_tags)
        self._frame_size = max(1, int(0.2341 * num_active_tags))
        
        self._slots_in_frame = [[] for _ in range(self._frame_size)]
        for tag in self.active_tags:
            slot_index = hash(tag.id + str(self._seed)) % self._frame_size
            self._slots_in_frame[slot_index].append(tag)
        
        # ECUMI的指示器向量是f-bit
        indicator_bits = float(self._frame_size)
        return AlgorithmStepResult(
            operation_type='ECUMI_SETUP',
            reader_bits=indicator_bits,
            tag_bits=0,
            operation_description=f"ECUMI Frame Start: N={num_active_tags}"
        )

    def perform_step(self) -> AlgorithmStepResult:
        if not self._frame_in_progress:
            return self._start_new_frame()

        if self._current_slot >= self._frame_size:
            self.active_tags = [t for t in self.active_tags if t not in self._tags_identified_in_frame]
            self._frame_in_progress = False
            return AlgorithmStepResult(operation_type="idle", operation_description="ECUMI Frame End")

        tags_in_slot = self._slots_in_frame[self._current_slot]
        self._current_slot += 1
        j = len(tags_in_slot)
        x_prime = self.config.get('ecumi_x_prime', 10)

        reader_bits = self.config.get('READER_CMD_BASE_BITS', 37)
        tag_bits = 0
        
        # 判断时隙是否可用
        if not (1 <= j <= x_prime):
            return AlgorithmStepResult('ECUMI_SKIPPED_SLOT', reader_bits, 0, f"Slot {self._current_slot-1}: Skipped (j={j})")

        num_present_tags = sum(1 for t in tags_in_slot if t.is_present)
        # 标签响应比特数
        tag_bits = num_present_tags * x_prime
        
        d_values = {hash(t.id + str(self._seed)) % x_prime for t in tags_in_slot}
        if len(d_values) != j:
            # 内部碰撞，标签依然响应了，但无法识别
            return AlgorithmStepResult('ECUMI_COLLISION_SLOT', reader_bits, tag_bits, f"Slot {self._current_slot-1}: Collision (j={j})")
        
        # 可用时隙的识别逻辑
        actual_signal = 0
        for tag in tags_in_slot:
            if tag.is_present:
                d = hash(tag.id + str(self._seed)) % x_prime
                actual_signal |= (1 << d)

        for tag in tags_in_slot:
            d = hash(tag.id + str(self._seed)) % x_prime
            if (actual_signal >> d) & 1:
                self.identified_present_ids.add(tag.id)
            else:
                self.identified_missing_ids.add(tag.id)
            self._tags_identified_in_frame.add(tag)
        
        return AlgorithmStepResult('ECUMI_USABLE_SLOT', reader_bits, tag_bits, f"Slot {self._current_slot-1}: Usable (j={j})")


# ==============================================================================
# 3. 主程序入口
# ==============================================================================
if __name__ == '__main__':
    # 假设您的项目中存在一个 Framework.py 文件
    from Framework import (
        run_missing_tag_simulation, 
        print_results, 
        BaselinePollingAlgo
    )

    # 1. 定义全局仿真配置
    global_config = {
        'TOTAL_TAGS': 10000,
        'MISSING_RATE': 0.1 ,
        'BINARY_LENGTH': 96,
        'DATA_RATE_BPS': 160000.0,
        'BITS_PER_MICROSECOND': 160000.0 / 1.0e6,
        'T1_RTcal': 25.0,
        'T2_TRcal': 25.0,
        'MINIMAL_GUARD_TIME_US': 50.0,
        'READER_CMD_BASE_BITS': 37,
        'TAG_SHORT_RESP_BITS': 1,
    }

    # 2. 运行 CUMI 仿真
    print("\n--- 运行 CUMI 算法 (动态时间模型) ---")
    cumi_specific_config = {'cumi_x': 3} # 论文推荐参数
    cumi_results = run_missing_tag_simulation(
        global_config=global_config,
        algorithm_class=CUMIAlgo,
        algorithm_specific_config=cumi_specific_config
    )
    print_results(cumi_results)

    # 3. 运行 ECUMI 仿真
    print("\n--- 运行 ECUMI 算法 (动态时间模型) ---")
    ecumi_specific_config = {'ecumi_x_prime': 10} # 论文推荐参数
    ecumi_results = run_missing_tag_simulation(
        global_config=global_config,
        algorithm_class=ECUMIAlgo,
        algorithm_specific_config=ecumi_specific_config
    )
    print_results(ecumi_results)

    # 4. (可选) 运行基线算法进行对比
    print("\n\n--- 运行基线轮询算法作为对比 ---")
    baseline_results = run_missing_tag_simulation(
        global_config=global_config,
        algorithm_class=BaselinePollingAlgo,
        algorithm_specific_config={}
    )
    print_results(baseline_results)

