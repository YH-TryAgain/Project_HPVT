# -*- coding: utf-8 -*-
"""
此文件包含了适配了新的动态时间计算框架的 HPVT 算法。
其中 get_results 和 _finalize_aloha_frame 方法已被修复，以确保“首次发现时间”指标的精确计算。
"""
import random
from collections import deque
from typing import List, Set, Tuple, Dict

# 基础框架的引用
# 假设您的项目中存在一个 Framework.py 文件
from Framework import *

# ==============================================================================
# HPVT 算法实现 (逻辑已修复)
# ==============================================================================

class _HPVT_Task:
    """HPVT算法内部使用的任务对象。"""
    def __init__(self, prefix: str, tags_in_scope: List['Tag']):
        self.prefix = prefix
        self.tags_in_scope = tags_in_scope

class _AlohaFrameContext:
    """用于存储当前正在执行的ALOHA帧的上下文信息。"""
    def __init__(self, task: _HPVT_Task, frame_size: int, random_seed: int):
        self.original_task = task
        self.frame_size = frame_size
        self.random_seed = random_seed
        self.current_slot = 0
        self.expected_slots = [[] for _ in range(frame_size)]
        self.actual_responses = [[] for _ in range(frame_size)]
        
        for tag in task.tags_in_scope:
            slot_index = hash(tag.id + str(random_seed)) % frame_size
            self.expected_slots[slot_index].append(tag)
            if tag.is_present:
                self.actual_responses[slot_index].append(tag)


class HPVTAlgo(MissingTagAlgorithmInterface):
    """
    分层并行验证树 (HPVT) 算法，已适配新框架并修复了get_results和内部逻辑。
    """
    
    def initialize(self, expected_tags: List[Tag]):
        self.expected_tags_db = expected_tags
        self.found_present_ids = set()
        self.found_missing_ids = set()
        self.task_queue = deque()
        
        self.STATE_IDLE, self.STATE_AWAITING_VERIFICATION, self.STATE_IN_ALOHA_FRAME = 0, 1, 2
        self.current_state = self.STATE_IDLE
        
        self.pending_task = None
        self.active_aloha_frame = None

        if not self.expected_tags_db: return

        tags_0 = [t for t in self.expected_tags_db if t.id.startswith('0')]
        tags_1 = [t for t in self.expected_tags_db if t.id.startswith('1')]
        
        if tags_0: self.task_queue.append(_HPVT_Task(prefix='0', tags_in_scope=tags_0))
        if tags_1: self.task_queue.append(_HPVT_Task(prefix='1', tags_in_scope=tags_1))

    def is_finished(self) -> bool:
        return not self.task_queue and self.current_state == self.STATE_IDLE

    def get_results(self) -> Tuple[Set[str], Set[str]]:
        """
        !!已修复!!: 返回一个更准确的报告以支持时间线统计。
        - 仿真中: 只返回高置信度的识别结果，以精确测量“首次发现时间”。
        - 仿真后: 执行最终清算，将所有未确认的标签归为丢失，以确保准确率统计正确。
        """
        if not self.is_finished():
            # --- 仿真中：只报告高置信度的结果 ---
            return self.found_present_ids, self.found_missing_ids
        else:
            # --- 仿真结束：执行最终清算并报告 ---
            all_ids_in_db = {t.id for t in self.expected_tags_db}
            identified_ids = self.found_present_ids.union(self.found_missing_ids)
            unconfirmed_ids = all_ids_in_db - identified_ids
            
            # 仅在最后一步更新状态
            if unconfirmed_ids:
                self.found_missing_ids.update(unconfirmed_ids)
            
            return self.found_present_ids, self.found_missing_ids

    def perform_step(self) -> 'AlgorithmStepResult':
        if self.is_finished():
            return AlgorithmStepResult(operation_type="finished")
        if self.current_state == self.STATE_IN_ALOHA_FRAME:
            return self._perform_aloha_slot()
        if self.current_state == self.STATE_AWAITING_VERIFICATION:
            return self._setup_parallel_verification()
        if self.task_queue:
            return self._perform_group_probe()
        return AlgorithmStepResult(operation_type="idle")

    def _perform_group_probe(self) -> 'AlgorithmStepResult':
        current_task = self.task_queue.popleft()
        reader_bits = self.config['READER_CMD_BASE_BITS'] + len(current_task.prefix)
        responding_tags = [t for t in current_task.tags_in_scope if t.is_present]
        tag_bits = 0
        
        if not responding_tags:
            # 高效剪枝：当一个分支无响应时，该分支下所有标签都被高置信度地确认为丢失。
            tags_in_scope_ids = {t.id for t in current_task.tags_in_scope}
            self.found_missing_ids.update(tags_in_scope_ids)
            op_desc = f"HPVT Probe on '{current_task.prefix}': Idle. Pruned."
        else:
            tag_bits = self.config.get('TAG_SHORT_RESP_BITS', 1)
            self.current_state = self.STATE_AWAITING_VERIFICATION
            self.pending_task = current_task
            op_desc = f"HPVT Probe on '{current_task.prefix}': Active."
            
        return AlgorithmStepResult('PROBE', reader_bits, tag_bits, op_desc)

    def _setup_parallel_verification(self) -> 'AlgorithmStepResult':
        current_task = self.pending_task; self.pending_task = None
        n_subset = len(current_task.tags_in_scope)
        frame_size = n_subset if n_subset > 0 else 1
        self.active_aloha_frame = _AlohaFrameContext(current_task, frame_size, random.randint(0, 10000))
        self.current_state = self.STATE_IN_ALOHA_FRAME
        reader_bits = self.config['READER_CMD_BASE_BITS'] + 32
        return AlgorithmStepResult('ALOHA_SETUP', reader_bits, 0, f"HPVT ALOHA Setup for '{current_task.prefix}': f={frame_size}.")

    def _perform_aloha_slot(self) -> 'AlgorithmStepResult':
        frame_ctx = self.active_aloha_frame; slot_idx = frame_ctx.current_slot
        expected = frame_ctx.expected_slots[slot_idx]
        responded = frame_ctx.actual_responses[slot_idx]
        
        # !!修改!!: 此处不再更新集合，将逻辑移至_finalize_aloha_frame，以避免重复判断
        # 仅计算本时隙的通信开销
        tag_bits = len(responded) * self.config.get('TAG_SHORT_RESP_BITS', 1)
        
        frame_ctx.current_slot += 1
        if frame_ctx.current_slot >= frame_ctx.frame_size:
            self._finalize_aloha_frame()
            
        return AlgorithmStepResult('ALOHA_SLOT', 0, tag_bits, f"ALOHA slot {slot_idx+1}/{frame_ctx.frame_size}")

    def _finalize_aloha_frame(self):
        """
        !!已重写!!: 采用更严谨的逻辑来处理ALOHA帧的结果。
        """
        frame_ctx = self.active_aloha_frame
        
        # 这个集合将包含所有无法在本轮解决，需要进入下一轮分裂的标签
        unresolved_tags = set()

        for i in range(frame_ctx.frame_size):
            expected = frame_ctx.expected_slots[i]
            responded = frame_ctx.actual_responses[i]

            # Case 1: 单例时隙 - 这是高置信度识别的主要来源
            if len(expected) == 1:
                tag = expected[0]
                if len(responded) == 1:
                    self.found_present_ids.add(tag.id)
                else: # len(responded) == 0
                    self.found_missing_ids.add(tag.id)
                continue

            # Case 2: 多标签预期时隙
            if len(expected) > 1:
                # Case 2a: 发生碰撞(>1响应) 或 沉默(0响应)
                # 这两种情况都无法确定单个标签的状态，所有涉及的标签都需进一步处理
                if len(responded) != 1:
                    for tag in expected:
                        unresolved_tags.add(tag)
                
                # Case 2b: 多标签预期，但只有一个响应 (这是一个高置信度事件)
                elif len(responded) == 1:
                    present_tag = responded[0]
                    self.found_present_ids.add(present_tag.id)
                    # 该时隙中，除了响应的那个，其他所有预期的标签都可以被确认为丢失
                    for tag in expected:
                        if tag.id != present_tag.id:
                            self.found_missing_ids.add(tag.id)
        
        # 为所有未解决的标签创建新的分裂任务
        if unresolved_tags:
            prefix = frame_ctx.original_task.prefix
            if len(prefix) < self.config.get('BINARY_LENGTH', 96):
                p0, p1 = prefix + '0', prefix + '1'
                tags0 = [t for t in unresolved_tags if t.id.startswith(p0)]
                tags1 = [t for t in unresolved_tags if t.id.startswith(p1)]
                if tags0: self.task_queue.append(_HPVT_Task(p0, tags0))
                if tags1: self.task_queue.append(_HPVT_Task(p1, tags1))
        
        self.active_aloha_frame = None
        self.current_state = self.STATE_IDLE

# ==============================================================================
# 3. 主程序入口
# ==============================================================================

if __name__ == '__main__':
    # 假设您的项目中存在一个 Framework.py 文件
    from Framework import (
        BaselinePollingAlgo, run_missing_tag_simulation, print_results
    )

    # 1. 定义全局仿真配置
    global_config = {
        'TOTAL_TAGS': 1000, 'MISSING_RATE': 0.1, 'BINARY_LENGTH': 96,
        'DATA_RATE_BPS': 160000.0, 'BITS_PER_MICROSECOND': 160000.0 / 1.0e6,
        'T1_RTcal': 25.0, 'T2_TRcal': 25.0, 'MINIMAL_GUARD_TIME_US': 50.0,
        'READER_CMD_BASE_BITS': 37, 'TAG_SHORT_RESP_BITS': 1,
    }

    # 2. 运行仿真
    print("--- 运行基线轮询算法 ---")
    baseline_results = run_missing_tag_simulation(
        global_config=global_config,
        algorithm_class=BaselinePollingAlgo,
        algorithm_specific_config={}
    )
    
    print("\n--- 运行HPVT算法 (已最终修复首次发现时间逻辑) ---")
    hpvt_results = run_missing_tag_simulation(
        global_config=global_config,
        algorithm_class=HPVTAlgo,
        algorithm_specific_config={}
    )

    # 3. 打印结果
    print("\n" + "="*50 + "\n                      仿 真 结 果 对 比\n" + "="*50)
    print_results(baseline_results)
    print_results(hpvt_results)
    print("="*50)
