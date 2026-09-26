# -*- coding: utf-8 -*-
"""
【多维度指标追踪版】HPVT 的查询聚合优化算法实现。

此版本已修复在并行验证阶段无法识别丢失标签的逻辑错误。
"""

import os
import random
from collections import deque
from typing import List, Set, Tuple

# 导入框架依赖
from framework import (
    MissingTagAlgorithmInterface,
    Tag,
    AlgorithmStepResult,
    CONSTANTS
)


class _HPVT_Task:
    """内部辅助类，用于表示一个待处理的树分支任务。"""
    def __init__(self, prefix: str, tags_in_scope: List[Tag]):
        self.prefix: str = prefix
        self.tags_in_scope: List[Tag] = tags_in_scope

class _AlohaFrameContext:
    """
    【已重构】内部辅助类，用于管理一次并行验证 (ALOHA) 的上下文。
    
    现在它会同时记录期望映射和实际响应，以支持丢失标签的识别。
    """
    def __init__(self, tags_to_verify: List[Tag], frame_size: int):
        self.tags: List[Tag] = tags_to_verify
        self.frame_size: int = frame_size
        self.current_slot_index: int = 0
        
        # expected_slots 存储此任务范围内所有标签的期望映射
        self.expected_slots: List[List[Tag]] = [[] for _ in range(frame_size)]
        # actual_slots 仅存储在场标签的映射，用于模拟真实物理响应
        self.actual_slots: List[List[Tag]] = [[] for _ in range(frame_size)]

        # 为此帧中的每个标签（无论在场与否）预先计算一个确定的时隙映射
        # 这样可以方便地在处理时隙时进行查询
        tag_to_slot_map = {}
        if frame_size > 0:
            for tag in self.tags:
                # 使用ID和帧大小进行哈希，比纯随机更具确定性，但这里为了保持原意使用随机
                slot_idx = random.randint(0, frame_size - 1)
                tag_to_slot_map[tag.id] = slot_idx
        
        # 根据预计算的映射关系，填充期望映射表和实际响应表
        for tag in self.tags:
            if tag.id in tag_to_slot_map:
                slot_idx = tag_to_slot_map[tag.id]
                self.expected_slots[slot_idx].append(tag)
                if tag.is_present:
                    self.actual_slots[slot_idx].append(tag)


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

        self.metrics = {
            "pruning_events": 0,
            "aloha_frames_initiated": 0,
            "collision_slots": 0,
            "total_steps": 0
        }
        
        # 创建并加入初始任务 (前缀 "0" 和 "1")
        if expected_tags:
            tags0 = [t for t in self.expected_tags_db if t.id.startswith('0')]
            tags1 = [t for t in self.expected_tags_db if t.id.startswith('1')]
            # 队列先进先出，先处理'0'，所以后进
            if tags1: self.task_queue.append(_HPVT_Task('1', tags1))
            if tags0: self.task_queue.append(_HPVT_Task('0', tags0))

    def perform_step(self) -> AlgorithmStepResult:
        """
        执行协议的单个逻辑步骤，并递增总步骤计数器。
        """
        self.metrics["total_steps"] += 1

        if self.current_state == self.STATE_PROBING:
            return self._perform_group_probe()
        elif self.current_state == self.STATE_VERIFYING:
            return self._perform_aloha_slot()
        return AlgorithmStepResult(operation_type='idle')

    def is_finished(self) -> bool:
        """当任务队列为空，且当前不处于并行验证状态时，算法结束。"""
        return not self.task_queue and self.current_state == self.STATE_PROBING

    def get_results(self) -> Tuple[Set[str], Set[str]]:
        """直接返回已识别的在场和丢失标签集合。"""
        return self.found_present_ids, self.found_missing_ids

    def _are_siblings(self, task1: _HPVT_Task, task2: _HPVT_Task) -> bool:
        """检查两个任务是否为兄弟节点。"""
        if not task1 or not task2 or not task1.prefix or not task2.prefix:
            return False
        return len(task1.prefix) == len(task2.prefix) and task1.prefix[:-1] == task2.prefix[:-1]

    def _perform_group_probe(self) -> AlgorithmStepResult:
        """
        执行组探测，包含查询聚合优化。
        """
        if not self.task_queue:
            return AlgorithmStepResult(operation_type='idle')

        is_first_in_pair = False
        # 检查是否可以进行查询聚合
        if len(self.task_queue) >= 2 and self._are_siblings(self.task_queue[0], self.task_queue[1]):
            if not self._shared_overhead_just_paid:
                is_first_in_pair = True
                self._shared_overhead_just_paid = True
        
        task = self.task_queue.popleft()
        
        # 根据是否为聚合查询的第二部分来计算读写器开销
        if self._shared_overhead_just_paid and not is_first_in_pair:
            reader_bits = len(task.prefix) - (len(task.prefix) - 1) # 只支付最后一位不同的比特
            self._shared_overhead_just_paid = False
        else:
            reader_bits = CONSTANTS.READER_CMD_BASE_BITS + len(task.prefix)
            
        responding_tags = [t for t in task.tags_in_scope if t.is_present]
        tag_bits = CONSTANTS.TAG_SHORT_RESP_BITS if responding_tags else 0

        if not responding_tags:
            # 记录剪枝事件，并批量识别丢失标签
            self.metrics["pruning_events"] += 1
            self.found_missing_ids.update({t.id for t in task.tags_in_scope})
            op_desc = f"Agg-Probe '{task.prefix}': Idle, Pruned"
        else:
            # 探测到活跃分支
            if len(task.tags_in_scope) <= self.aloha_threshold:
                # 启动并行验证
                self.metrics["aloha_frames_initiated"] += 1
                self.current_state = self.STATE_VERIFYING
                frame_size = len(responding_tags)
                self.aloha_context = _AlohaFrameContext(task.tags_in_scope, frame_size)
                op_desc = f"Agg-Probe '{task.prefix}': Active -> Start ALOHA"
            else:
                # 继续分裂
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
        【已修复】处理单个ALOHA时隙，能正确识别丢失和碰撞的标签。
        """
        ctx = self.aloha_context
        # 如果并行验证帧结束，则返回探测状态
        if not ctx or ctx.current_slot_index >= ctx.frame_size:
            self.current_state = self.STATE_PROBING
            self.aloha_context = None
            return self.perform_step() 
        
        current_index = ctx.current_slot_index
        # 获取当前时隙的“实际响应”和“期望映射”
        actual_slot_content = ctx.actual_slots[current_index]
        expected_slot_content = ctx.expected_slots[current_index]
        
        op_desc = f"ALOHA Slot {current_index + 1}/{ctx.frame_size}: "

        if len(actual_slot_content) == 0:  # 时隙空闲 (Idle)
            op_desc += "Idle"
            # 【关键修复】如果时隙为空闲，所有期望在此响应的标签都为丢失
            if len(expected_slot_content) > 0:
                missing_tags_in_slot = {t.id for t in expected_slot_content}
                self.found_missing_ids.update(missing_tags_in_slot)
                op_desc += f", Found {len(missing_tags_in_slot)} Missing"

        elif len(actual_slot_content) == 1:  # 时隙成功 (Success)
            tag = actual_slot_content[0]
            self.found_present_ids.add(tag.id)
            op_desc += f"Success ({tag.id[-6:]})"

        else:  # 时隙碰撞 (Collision)
            self.metrics["collision_slots"] += 1
            op_desc += f"Collision ({len(actual_slot_content)} tags)"
            # 【关键修复】将所有期望在此响应的标签（包括丢失的）重新加入任务队列
            if len(expected_slot_content) > 1:
                collided_ids = [t.id for t in expected_slot_content]
                common_prefix = os.path.commonprefix(collided_ids)
                self.task_queue.appendleft(_HPVT_Task(common_prefix, expected_slot_content))

        ctx.current_slot_index += 1
        return AlgorithmStepResult(operation_type='ALOHA_SLOT', operation_description=op_desc)
