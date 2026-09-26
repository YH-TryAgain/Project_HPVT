# -*- coding: utf-8 -*-

import random
import time
import numpy as np
import math

from Framework import *


# ==============================================================================
# CUMI 算法实现 (!! 已重构为按时隙推进 !!)
# ==============================================================================
class CUMIAlgo(MissingTagAlgorithmInterface):
    """
    实现了论文中的 CUMI 协议，已重构为按时隙(slot-by-slot)推进。
    """
    def initialize(self, expected_tags: list[Tag]):
        self.active_tags = list(expected_tags)
        self.identified_present_ids = set()
        self.identified_missing_ids = set()
        
        # 论文中的时间常量 (µs)
        self.tau_p = self.config.get('tau_p', 37.7)
        self.tau_d = self.config.get('tau_d', 18.8)
        self.tau_w = self.config.get('tau_w', 302.0)
        
        # 算法状态变量
        self._frame_in_progress = False
        self._current_slot = 0
        self._frame_size = 0
        self._slots_in_frame = []
        self._tags_identified_in_frame = set()
        self._seed = 0

    def is_finished(self) -> bool:
        return not self.active_tags and not self._frame_in_progress

    def get_results(self) -> tuple[set[str], set[str]]:
        return self.identified_present_ids, self.identified_missing_ids

    def _start_new_frame(self):
        """准备并开始一个新帧"""
        if not self.active_tags:
            return None # 没有活动标签，无法开始新帧

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
        
        # 返回帧开始时的固定开销（指示器向量）
        indicator_bits = self._frame_size * math.ceil(math.log2(x + 1))
        time_delta = indicator_bits * self.tau_p
        return AlgorithmStepResult(time_delta, indicator_bits, 0, f"CUMI Frame Start: N={num_active_tags}")

    def perform_step(self) -> AlgorithmStepResult:
        if not self._frame_in_progress:
            # 尝试开始一个新帧
            result = self._start_new_frame()
            return result if result is not None else AlgorithmStepResult(0, 0, 0, "finished")

        # 如果帧已结束，则进行清理并准备下一帧
        if self._current_slot >= self._frame_size:
            self.active_tags = [t for t in self.active_tags if t not in self._tags_identified_in_frame]
            self._frame_in_progress = False
            return AlgorithmStepResult(0, 0, 0, "CUMI Frame End")

        # --- 处理当前时隙 ---
        tags_in_slot = self._slots_in_frame[self._current_slot]
        self._current_slot += 1
        j = len(tags_in_slot)
        x = self.config.get('cumi_x', 3)

        if not (1 <= j <= x):
            # 不可用时隙，但仍有查询开销
            return AlgorithmStepResult(self.tau_w, 0, 0, f"Slot {self._current_slot-1}: Skipped (j={j})")
        
        d_values = {hash(t.id + str(self._seed)) % j for t in tags_in_slot}
        is_usable = (len(d_values) == j)

        if not is_usable:
            return AlgorithmStepResult(self.tau_w, 0, 0, f"Slot {self._current_slot-1}: Collision (j={j})")
        
        # 可用时隙的时间和比特计算
        time_delta = self.tau_w
        tag_bits = j
        num_present_tags = sum(1 for t in tags_in_slot if t.is_present)
        time_delta += num_present_tags * j * self.tau_d # 多个标签响应的时间

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

        return AlgorithmStepResult(time_delta, 0, tag_bits, f"Slot {self._current_slot-1}: Usable (j={j})")


# ==============================================================================
# ECUMI 算法实现 (!! 已重构为按时隙推进 !!)
# ==============================================================================
class ECUMIAlgo(CUMIAlgo):
    """
    实现了论文中的 ECUMI 协议，已重构为按时隙(slot-by-slot)推进。
    """
    def _start_new_frame(self):
        if not self.active_tags:
            return None

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
        
        indicator_bits = self._frame_size
        time_delta = indicator_bits * self.tau_p
        return AlgorithmStepResult(time_delta, indicator_bits, 0, f"ECUMI Frame Start: N={num_active_tags}")

    def perform_step(self) -> AlgorithmStepResult:
        if not self._frame_in_progress:
            result = self._start_new_frame()
            return result if result is not None else AlgorithmStepResult(0, 0, 0, "finished")

        if self._current_slot >= self._frame_size:
            self.active_tags = [t for t in self.active_tags if t not in self._tags_identified_in_frame]
            self._frame_in_progress = False
            return AlgorithmStepResult(0, 0, 0, "ECUMI Frame End")

        tags_in_slot = self._slots_in_frame[self._current_slot]
        self._current_slot += 1
        j = len(tags_in_slot)
        x_prime = self.config.get('ecumi_x_prime', 10)

        # ECUMI中，所有时隙都有基础查询开销
        time_delta = self.tau_w
        tag_bits = 0
        
        is_ecumi_usable = (1 <= j <= x_prime)
        if not is_ecumi_usable:
            return AlgorithmStepResult(time_delta, 0, 0, f"Slot {self._current_slot-1}: Skipped (j={j})")

        # 可用时隙的额外开销
        num_present_tags = sum(1 for t in tags_in_slot if t.is_present)
        time_delta += num_present_tags * x_prime * self.tau_d
        tag_bits = num_present_tags * x_prime
        
        d_values = {hash(t.id + str(self._seed)) % x_prime for t in tags_in_slot}
        is_fully_unfoldable = (len(d_values) == j)

        if not is_fully_unfoldable:
            # 时间已消耗，但无法识别
            return AlgorithmStepResult(time_delta, 0, tag_bits, f"Slot {self._current_slot-1}: Collision (j={j})")
        
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
        
        return AlgorithmStepResult(time_delta, 0, tag_bits, f"Slot {self._current_slot-1}: Usable (j={j})")

# ==============================================================================
# 主程序调用入口 (无修改)
# ==============================================================================
if __name__ == '__main__':
    global_simulation_config = {
        'TOTAL_TAGS': 10000,
        'MISSING_RATE': 0.01,
        'BINARY_LENGTH': 96,
    }

    cumi_specific_config = {'cumi_x': 3}
    cumi_results = run_missing_tag_simulation(global_simulation_config, CUMIAlgo, cumi_specific_config)
    print_results(cumi_results)

    ecumi_specific_config = {'ecumi_x_prime': 10}
    ecumi_results = run_missing_tag_simulation(global_simulation_config, ECUMIAlgo, ecumi_specific_config)
    print_results(ecumi_results)
