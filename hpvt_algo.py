# -*- coding: utf-8 -*-
"""
HPVT 算法的真实模拟实现 (查询聚合优化版)。

此版本移除了所有“上帝视角”和理想化假设，旨在更真实地模拟协议在物理世界中的行为。

核心修改:
1.  **动态ALOHA帧 (Q-protocol)**: 并行验证不再使用预知在架标签数的最优帧长，
    而是通过模拟Q-protocol，根据上一轮的空闲/成功/碰撞结果动态调整帧长(Q值)，
    可能需要多轮才能完全识别一个子集。
2.  **现实的碰撞处理**: 算法不再能预知碰撞时隙中的具体标签。一个ALOHA会话
    结束后，所有成功识别的标签被确认，剩余的则被推断为丢失。这更贴近现实。
3.  **重构的状态机**: 算法状态机被重构以支持多轮ALOHA验证过程，逻辑更清晰、鲁棒。
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


class _AlohaContext:
    """
    【全新重构】内部辅助类，用于管理一次多轮并行验证 (ALOHA) 的完整会话。
    """
    def __init__(self, original_prefix: str, tags_to_verify: List[Tag], initial_q: float = 4.0):
        # --- 会话级状态 ---
        self.original_prefix: str = original_prefix
        self.tags_to_verify: List[Tag] = tags_to_verify
        # 真实在架的标签，这是本会话需要全部找出的目标
        self.present_tags_in_scope: Set[Tag] = {t for t in tags_to_verify if t.is_present}
        # 在本会话中已成功识别的标签
        self.identified_tags_in_session: Set[Tag] = set()
        self.q_value: float = initial_q

        # --- 当前帧状态 ---
        self.frame_size: int = 0
        self.slots: List[List[Tag]] = []
        self.current_slot_index: int = 0
        # 当前帧的统计数据
        self.idle_count_this_frame: int = 0
        self.success_count_this_frame: int = 0
        self.collision_count_this_frame: int = 0

    def is_session_resolved(self) -> bool:
        """检查是否本会话中的所有在架标签都已被识别。"""
        return len(self.identified_tags_in_session) == len(self.present_tags_in_scope)

    def prepare_next_frame(self):
        """
        根据当前的Q值，准备下一轮(或第一轮)ALOHA帧。
        """
        self.frame_size = round(2**self.q_value)
        self.slots = [[] for _ in range(self.frame_size)]
        self.current_slot_index = 0
        self.idle_count_this_frame = 0
        self.success_count_this_frame = 0
        self.collision_count_this_frame = 0

        # 找出本轮需要参与竞争的标签 (尚未被识别的在架标签)
        tags_to_compete = [t for t in self.present_tags_in_scope if t not in self.identified_tags_in_session]

        # 如果没有需要竞争的标签了，直接返回 (安全检查)
        if not tags_to_compete:
            return

        # 模拟标签选择时隙
        for tag in tags_to_compete:
            slot_idx = random.randint(0, self.frame_size - 1)
            self.slots[slot_idx].append(tag)


class HPVT_AggregatedAlgo(MissingTagAlgorithmInterface):
    """
    HPVT 的查询聚合优化版本，实现了更真实的动态ALOHA验证机制。
    """
    STATE_PROBING = 'PROBING'
    STATE_VERIFYING = 'VERIFYING'
    ADJUSTMENT_C = 0.3  # Q值调整步长

    def __init__(self, aloha_threshold: int = 128, binary_length: int = 96):
        super().__init__()
        self.aloha_threshold = aloha_threshold
        self.binary_length = binary_length
        self.initialize([])

    def initialize(self, expected_tags: List[Tag]):
        """初始化或重置算法的所有状态变量。"""
        super().initialize(expected_tags)
        self.task_queue: deque[_HPVT_Task] = deque()
        self.found_present_ids: Set[str] = set()
        self.found_missing_ids: Set[str] = set()
        self.current_state: str = self.STATE_PROBING
        self.aloha_context: _AlohaContext = None
        self._shared_overhead_just_paid: bool = False

        self.metrics = {
            "pruning_events": 0,
            "aloha_frames_initiated": 0,
            "collision_slots": 0,
            "total_steps": 0
        }

        if expected_tags:
            tags0 = [t for t in self.expected_tags_db if t.id.startswith('0')]
            tags1 = [t for t in self.expected_tags_db if t.id.startswith('1')]
            if tags1: self.task_queue.append(_HPVT_Task('1', tags1))
            if tags0: self.task_queue.append(_HPVT_Task('0', tags0))

    def perform_step(self) -> AlgorithmStepResult:
        """执行协议的单个逻辑步骤。"""
        self.metrics["total_steps"] += 1

        if self.current_state == self.STATE_PROBING:
            return self._perform_group_probe()
        
        elif self.current_state == self.STATE_VERIFYING:
            # 检查当前ALOHA帧是否已处理完毕
            if self.aloha_context and self.aloha_context.current_slot_index >= self.aloha_context.frame_size:
                return self._handle_end_of_aloha_frame()
            else:
                return self._perform_aloha_slot()
                
        return AlgorithmStepResult(operation_type='idle')

    def is_finished(self) -> bool:
        """当任务队列为空，且当前不处于并行验证状态时，算法结束。"""
        return not self.task_queue and self.current_state == self.STATE_PROBING

    def get_results(self) -> Tuple[Set[str], Set[str]]:
        """返回已识别的在场和丢失标签集合。"""
        return self.found_present_ids, self.found_missing_ids

    def _are_siblings(self, task1: _HPVT_Task, task2: _HPVT_Task) -> bool:
        """检查两个任务是否为兄弟节点。"""
        if not task1 or not task2 or not task1.prefix or not task2.prefix:
            return False
        return len(task1.prefix) == len(task2.prefix) and task1.prefix[:-1] == task2.prefix[:-1]

    def _perform_group_probe(self) -> AlgorithmStepResult:
        """执行组探测，包含查询聚合优化。"""
        if not self.task_queue:
            return AlgorithmStepResult(operation_type='idle')

        is_first_in_pair = False
        if len(self.task_queue) >= 2 and self._are_siblings(self.task_queue[0], self.task_queue[1]):
            if not self._shared_overhead_just_paid:
                is_first_in_pair = True
                self._shared_overhead_just_paid = True
        
        task = self.task_queue.popleft()
        
        if self._shared_overhead_just_paid and not is_first_in_pair:
            reader_bits = len(task.prefix) - (len(task.prefix) - 1)
            self._shared_overhead_just_paid = False
        else:
            reader_bits = CONSTANTS.READER_CMD_BASE_BITS + len(task.prefix)
            
        responding_tags = [t for t in task.tags_in_scope if t.is_present]
        tag_bits = CONSTANTS.TAG_SHORT_RESP_BITS if responding_tags else 0

        if not responding_tags:
            self.metrics["pruning_events"] += 1
            self.found_missing_ids.update({t.id for t in task.tags_in_scope})
            op_desc = f"Agg-Probe '{task.prefix}': Idle, Pruned"
        else:
            if len(task.tags_in_scope) <= self.aloha_threshold:
                self.metrics["aloha_frames_initiated"] += 1
                self.current_state = self.STATE_VERIFYING
                self.aloha_context = _AlohaContext(task.prefix, task.tags_in_scope)
                self.aloha_context.prepare_next_frame()
                op_desc = f"Agg-Probe '{task.prefix}': Active -> Start ALOHA (Q={self.aloha_context.q_value:.2f})"
            else:
                if len(task.prefix) < self.binary_length:
                    p0, p1 = task.prefix + '0', task.prefix + '1'
                    tags0 = [t for t in task.tags_in_scope if t.id.startswith(p0)]
                    tags1 = [t for t in task.tags_in_scope if t.id.startswith(p1)]
                    if tags1: self.task_queue.appendleft(_HPVT_Task(p1, tags1))
                    if tags0: self.task_queue.appendleft(_HPVT_Task(p0, tags0))
                op_desc = f"Agg-Probe '{task.prefix}': Active -> Splitting"
                
        return AlgorithmStepResult('PROBE', reader_bits, tag_bits, op_desc)

    def _perform_aloha_slot(self) -> AlgorithmStepResult:
        """处理当前ALOHA帧中的单个时隙。"""
        ctx = self.aloha_context
        if not ctx or ctx.current_slot_index >= ctx.frame_size:
            # 此处逻辑应由 perform_step 调度，理论上不应直接进入
            return AlgorithmStepResult(operation_type='idle', operation_description="Error: Invalid ALOHA slot state")

        current_index = ctx.current_slot_index
        slot_content = ctx.slots[current_index]
        op_desc = f"ALOHA Slot {current_index + 1}/{ctx.frame_size}: "

        if not slot_content:
            op_desc += "Idle"
            ctx.idle_count_this_frame += 1
        elif len(slot_content) == 1:
            tag = slot_content[0]
            self.found_present_ids.add(tag.id)
            ctx.identified_tags_in_session.add(tag)
            op_desc += f"Success ({tag.id[-6:]})"
            ctx.success_count_this_frame += 1
        else:
            op_desc += f"Collision ({len(slot_content)} tags)"
            ctx.collision_count_this_frame += 1
            self.metrics["collision_slots"] += 1
            
        ctx.current_slot_index += 1
        return AlgorithmStepResult(operation_type='ALOHA_SLOT', operation_description=op_desc)

    def _handle_end_of_aloha_frame(self) -> AlgorithmStepResult:
        """处理一轮ALOHA帧结束后的逻辑：评估结果并决定下一步。"""
        ctx = self.aloha_context
        op_desc = f"End of ALOHA Frame for '{ctx.original_prefix}': "

        if ctx.is_session_resolved():
            # 会话成功结束，所有在架标签已找到
            all_ids_in_scope = {t.id for t in ctx.tags_to_verify}
            identified_ids = {t.id for t in ctx.identified_tags_in_session}
            missing_ids_in_scope = all_ids_in_scope - identified_ids
            self.found_missing_ids.update(missing_ids_in_scope)
            
            op_desc += f"Resolved. Found {len(identified_ids)} present, {len(missing_ids_in_scope)} missing."
            
            # 清理上下文，返回探测状态
            self.aloha_context = None
            self.current_state = self.STATE_PROBING
            return AlgorithmStepResult('ALOHA_END', operation_description=op_desc)
        else:
            # 会话未结束，调整Q值并准备下一轮
            idle = ctx.idle_count_this_frame
            coll = ctx.collision_count_this_frame
            
            if coll > idle:
                ctx.q_value += self.ADJUSTMENT_C
            elif idle > coll:
                ctx.q_value -= self.ADJUSTMENT_C
            
            ctx.q_value = max(0.0, min(15.0, ctx.q_value)) # 限制Q值范围
            
            op_desc += f"S/C/I=({ctx.success_count_this_frame}/{coll}/{idle}). Not resolved. New Q={ctx.q_value:.2f}"
            
            # 准备下一帧
            ctx.prepare_next_frame()
            return AlgorithmStepResult('ALOHA_NEW_ROUND', operation_description=op_desc)
