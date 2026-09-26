# -*- coding: utf-8 -*-
"""
【多维度指标追踪版】HPVT 的查询聚合优化算法实现。

此版本在算法内部增加了多个计数器，用于追踪关键的机制性指标，
例如剪枝事件数、并行验证启动次数和碰撞时隙数等。
"""

import os
import random
from collections import deque
from typing import List, Set, Tuple

# 导入框架依赖
from Framework import (
    MissingTagAlgorithmInterface,
    Tag,
    AlgorithmStepResult,
    CONSTANTS
)

# ... 内部辅助类 _HPVT_Task, _AlohaFrameContext 保持不变 ...
class _HPVT_Task:
    def __init__(self, prefix: str, tags_in_scope: List[Tag]):
        self.prefix: str = prefix
        self.tags_in_scope: List[Tag] = tags_in_scope

class _AlohaFrameContext:
    def __init__(self, tags_to_verify: List[Tag], frame_size: int):
        self.tags: List[Tag] = tags_to_verify
        self.slots: List[List[Tag]] = [[] for _ in range(frame_size)]
        self.current_slot_index: int = 0
        self.frame_size: int = frame_size
        for tag in self.tags:
            if tag.is_present:
                if frame_size > 0:
                    slot_idx = random.randint(0, frame_size - 1)
                    self.slots[slot_idx].append(tag)


class HPVT_AggregatedAlgo(MissingTagAlgorithmInterface):
    """
    HPVT 的查询聚合优化版本，内置了多维度性能指标的追踪。
    """
    STATE_PROBING = 'PROBING'
    STATE_VERIFYING = 'VERIFYING'

    def __init__(self, aloha_threshold: int = 128, binary_length: int = 96):
        super().__init__()
        self.aloha_threshold = aloha_threshold
        self.binary_length = binary_length
        # 调用初始化方法来设置所有状态变量
        self.initialize([])

    def initialize(self, expected_tags: List[Tag]):
        """
        初始化或重置算法的所有状态变量和性能计数器。
        """
        super().initialize(expected_tags)
        self.task_queue: deque[_HPVT_Task] = deque()
        self.found_present_ids: Set[str] = set()
        self.found_missing_ids: Set[str] = set()
        self.current_state: str = self.STATE_PROBING
        self.aloha_context: _AlohaFrameContext = None
        self._shared_overhead_just_paid: bool = False

        # ==========================================================
        # 【新】初始化多维度性能指标计数器
        # ==========================================================
        self.metrics = {
            "pruning_events": 0,           # 剪枝事件总数
            "aloha_frames_initiated": 0,   # 并行验证启动次数
            "collision_slots": 0,          # 碰撞时隙总数
            "total_steps": 0               # 协议总步骤数
        }
        # ==========================================================
        
        # 创建并加入初始任务 (前缀 "0" 和 "1")
        if expected_tags:
            tags0 = [t for t in self.expected_tags_db if t.id.startswith('0')]
            tags1 = [t for t in self.expected_tags_db if t.id.startswith('1')]
            if tags1: self.task_queue.append(_HPVT_Task('1', tags1))
            if tags0: self.task_queue.append(_HPVT_Task('0', tags0))

    def perform_step(self) -> AlgorithmStepResult:
        """
        执行协议的单个逻辑步骤，并递增总步骤计数器。
        """
        # 【新】每次执行步骤，总步数+1
        self.metrics["total_steps"] += 1

        if self.current_state == self.STATE_PROBING:
            return self._perform_group_probe()
        elif self.current_state == self.STATE_VERIFYING:
            return self._perform_aloha_slot()
        return AlgorithmStepResult(operation_type='idle')

    # ... is_finished, get_results, _are_siblings 等方法保持不变 ...
    def is_finished(self) -> bool:
        return not self.task_queue and self.current_state == self.STATE_PROBING
    def get_results(self) -> Tuple[Set[str], Set[str]]:
        return self.found_present_ids, self.found_missing_ids
    def _are_siblings(self, task1, task2):
        if not task1 or not task2 or len(task1.prefix) == 0: return False
        return len(task1.prefix) == len(task2.prefix) and task1.prefix[:-1] == task2.prefix[:-1]


    def _perform_group_probe(self) -> AlgorithmStepResult:
        """
        执行组探测，并在适当的时候更新性能计数器。
        """
        # ... 聚合检查、任务弹出、比特计算等逻辑保持不变 ...
        if not self.task_queue: return AlgorithmStepResult(operation_type='idle')
        is_first_in_pair = False
        if len(self.task_queue) >= 2 and self._are_siblings(self.task_queue[0], self.task_queue[1]):
            if not self._shared_overhead_just_paid:
                is_first_in_pair = True
                self._shared_overhead_just_paid = True
        task = self.task_queue.popleft()
        if self._shared_overhead_just_paid and not is_first_in_pair:
            reader_bits = len(task.prefix)
            self._shared_overhead_just_paid = False
        else:
            reader_bits = CONSTANTS.READER_CMD_BASE_BITS + len(task.prefix)
        responding_tags = [t for t in task.tags_in_scope if t.is_present]
        tag_bits = CONSTANTS.TAG_SHORT_RESP_BITS if responding_tags else 0

        if not responding_tags:
            # 【新】记录一次剪枝事件
            self.metrics["pruning_events"] += 1
            self.found_missing_ids.update({t.id for t in task.tags_in_scope})
            op_desc = f"Agg-Probe '{task.prefix}': Idle, Pruned"
        else:
            if len(task.tags_in_scope) <= self.aloha_threshold:
                # 【新】记录一次并行验证启动
                self.metrics["aloha_frames_initiated"] += 1
                self.current_state = self.STATE_VERIFYING
                frame_size = len(responding_tags)
                self.aloha_context = _AlohaFrameContext(task.tags_in_scope, frame_size)
                op_desc = f"Agg-Probe '{task.prefix}': Active -> Start ALOHA"
            else:
                # ... 分裂逻辑保持不变 ...
                if len(task.prefix) < self.binary_length:
                    p0, p1 = task.prefix + '0', task.prefix + '1'
                    tags0 = [t for t in task.tags_in_scope if t.id.startswith(p0)]
                    tags1 = [t for t in task.tags_in_scope if t.id.startswith(p1)]
                    if tags1: self.task_queue.appendleft(_HPVT_Task(p1, tags1))
                    if tags0: self.task_queue.appendleft(_HPVT_Task(p0, tags0))
                op_desc = f"Agg-Probe '{task.prefix}': Active -> Splitting"
                
        return AlgorithmStepResult('PROBE', reader_bits, tag_bits, op_desc)

    def _perform_aloha_slot(self) -> AlgorithmStepResult:
        """
        处理ALOHA时隙，并在发生碰撞时更新计数器。
        """
        # ... 时隙处理逻辑保持不变 ...
        ctx = self.aloha_context
        if not ctx or ctx.current_slot_index >= ctx.frame_size:
            self.current_state = self.STATE_PROBING
            self.aloha_context = None
            return self.perform_step() 
        slot_content = ctx.slots[ctx.current_slot_index]
        op_desc = f"ALOHA Slot {ctx.current_slot_index + 1}/{ctx.frame_size}: "

        if len(slot_content) == 0:
            op_desc += "Idle"
        elif len(slot_content) == 1:
            tag = slot_content[0]
            self.found_present_ids.add(tag.id)
            op_desc += f"Success ({tag.id[-6:]})"
        else:
            # 【新】记录一次碰撞事件
            self.metrics["collision_slots"] += 1
            op_desc += f"Collision ({len(slot_content)} tags)"
            collided_ids = [t.id for t in slot_content]
            common_prefix = os.path.commonprefix(collided_ids)
            self.task_queue.appendleft(_HPVT_Task(common_prefix, slot_content))

        ctx.current_slot_index += 1
        return AlgorithmStepResult(operation_type='ALOHA_SLOT', operation_description=op_desc)
