# -*- coding: utf-8 -*-
"""
此文件包含了适配了新的动态时间计算框架的 IIP (Iterative ID-free Protocol) 算法，
并提供了一个独立的主程序入口来运行和测试其性能。
"""

import math
import random
import time
from collections import deque
from typing import List, Set, Tuple, Dict

# ==============================================================================
# 0. 从框架导入 (或在此处定义以确保独立运行)
# ==============================================================================
# from Framework import MissingTagAlgorithmInterface, AlgorithmStepResult, Tag
# from Framework import run_missing_tag_simulation, print_results, BaselinePollingAlgo

from Framework import *

class _IIPFrameContext:
    """用于存储当前正在执行的 IIP ALOHA 帧的上下文信息。"""
    def __init__(self, unverified_tags: List['Tag'], frame_size: int, random_seed: int, final_check_mode: bool):
        self.frame_size = frame_size
        self.random_seed = random_seed
        self.final_check_mode = final_check_mode
        self.current_slot = 0
        
        # 预计算期望时隙和 pre-frame 向量
        self.expected_slots = [[] for _ in range(frame_size)]
        for tag in unverified_tags:
            slot_index = hash(tag.id + str(random_seed)) % frame_size
            self.expected_slots[slot_index].append(tag)
        
        self.pre_frame_vector = [1 if len(tags_in_slot) > 1 else 0 for tags_in_slot in self.expected_slots]
        
        # 预计算真实响应
        self.actual_responses = [[] for _ in range(frame_size)]
        present_unverified_tags = [tag for tag in unverified_tags if tag.is_present]
        
        for tag in present_unverified_tags:
            slot_index = hash(tag.id + str(self.random_seed)) % frame_size
            should_respond = True
            
            # 模拟IIP协议的碰撞时隙二次哈希抑制机制
            if not self.final_check_mode and self.pre_frame_vector[slot_index] == 1:
                if hash(tag.id + str(self.random_seed) + "second_hash") % 2 == 0:
                    should_respond = False
            
            if should_respond:
                self.actual_responses[slot_index].append(tag)
        
        self.tags_that_responded_count = sum(len(s) for s in self.actual_responses)


class IIPAlgo(MissingTagAlgorithmInterface):
    """
    IIP (Iterative ID-free Protocol) 算法实现，已适配动态时间计算框架。
    """
    
    def initialize(self, expected_tags: List['Tag']):
        """初始化算法状态。"""
        self.expected_tags_db = expected_tags
        self.unverified_tags = list(self.expected_tags_db)
        self.found_present_ids = set()
        self.found_missing_ids = set()
        
        self.random_seed = 0
        self.optimal_rho = 1.516
        self.final_check_mode = False
        
        self.STATE_IDLE = 0
        self.STATE_IN_FRAME = 1
        self.current_state = self.STATE_IDLE
        self.active_frame_context = None

    def is_finished(self) -> bool:
        """当不再有未验证的标签且状态机空闲时，算法结束。"""
        return not self.unverified_tags and self.current_state == self.STATE_IDLE

    def get_results(self) -> Tuple[Set[str], Set[str]]:
        """返回逐步确认的集合。"""
        # 最后的清理逻辑：将仍未确认的标签都视为丢失
        all_ids_in_db = {t.id for t in self.expected_tags_db}
        unconfirmed_ids = all_ids_in_db - self.found_present_ids - self.found_missing_ids
        if unconfirmed_ids:
            self.found_missing_ids.update(unconfirmed_ids)
        return self.found_present_ids, self.found_missing_ids

    def perform_step(self) -> 'AlgorithmStepResult':
        """执行一个微步骤，要么是帧设置，要么是单时隙处理。"""
        if self.is_finished():
            return AlgorithmStepResult(operation_type="finished")
        
        if self.current_state == self.STATE_IN_FRAME:
            return self._perform_iip_slot()
        
        return self._setup_iip_frame()

    def _setup_iip_frame(self) -> 'AlgorithmStepResult':
        """执行“帧设置”微步骤：计算帧长并返回广播向量的比特开销。"""
        n_star = len(self.unverified_tags)
        if n_star == 0:
            self.current_state = self.STATE_IDLE
            return AlgorithmStepResult(operation_type="idle")

        # 使用论文中的最优负载因子计算帧长，并增加保护机制
        frame_size = int(round(n_star / self.optimal_rho)) if self.optimal_rho > 0 else n_star
        if n_star > 0:
            frame_size = max(frame_size, n_star, 1) # 确保帧长至少为1且不小于标签数

        self.active_frame_context = _IIPFrameContext(self.unverified_tags, frame_size, self.random_seed, self.final_check_mode)
        self.current_state = self.STATE_IN_FRAME
        
        # 核心修改：计算比特开销。IIP需要广播一个 pre-frame 向量。
        # 论文中还提到了一个 post-frame 向量，这里简化为只广播一个。
        reader_bits = frame_size # 广播一个长度为 f 的比特向量
        
        op_desc = f"IIP Frame Setup: N*={n_star}, f={frame_size}"
        return AlgorithmStepResult(
            operation_type='IIP_SETUP',
            reader_bits=reader_bits,
            tag_bits=0, # 标签此步不响应
            operation_description=op_desc
        )

    def _perform_iip_slot(self) -> 'AlgorithmStepResult':
        """执行“单时隙处理”微步骤，返回标签响应的比特开销。"""
        frame_ctx = self.active_frame_context
        slot_idx = frame_ctx.current_slot
        
        responded_in_slot = frame_ctx.actual_responses[slot_idx]
        expected_in_slot = frame_ctx.expected_slots[slot_idx]
        
        # 算法逻辑：根据时隙结果更新已发现/丢失集合
        if len(responded_in_slot) == 1:
            identified_tag = responded_in_slot[0]
            if identified_tag.id not in self.found_present_ids:
                self.found_present_ids.add(identified_tag.id)
        elif len(responded_in_slot) == 0 and len(expected_in_slot) == 1:
            missing_tag = expected_in_slot[0]
            if missing_tag.id not in self.found_missing_ids:
                self.found_missing_ids.add(missing_tag.id)
        
        # 核心修改：计算标签响应的比特数
        tag_bits = len(responded_in_slot) * self.config.get('TAG_SHORT_RESP_BITS', 1)

        frame_ctx.current_slot += 1
        if frame_ctx.current_slot >= frame_ctx.frame_size:
            self._finalize_iip_frame()
            
        op_desc = f"IIP Slot {slot_idx+1}/{frame_ctx.frame_size}"
        return AlgorithmStepResult(
            operation_type='ALOHA_SLOT', # 可视为一种特殊的ALOHA时隙
            reader_bits=0, # 读写器开销已在SETUP中计算
            tag_bits=tag_bits,
            operation_description=op_desc
        )

    def _finalize_iip_frame(self):
        """在一轮IIP帧结束后，更新状态。此方法不直接产生时间开销。"""
        frame_ctx = self.active_frame_context
        
        resolved_ids_this_frame = set()
        for i in range(frame_ctx.frame_size):
            if len(frame_ctx.actual_responses[i]) == 1:
                resolved_ids_this_frame.add(frame_ctx.actual_responses[i][0].id)
            elif len(frame_ctx.actual_responses[i]) == 0 and len(frame_ctx.expected_slots[i]) == 1:
                resolved_ids_this_frame.add(frame_ctx.expected_slots[i][0].id)
        
        # 根据本轮是否有标签被解决，来决定下一步策略
        if resolved_ids_this_frame:
            self.final_check_mode = False
            self.unverified_tags = [t for t in self.unverified_tags if t.id not in resolved_ids_this_frame]
        else: # 没有标签被解决，可能进入最终检查或确认全部丢失
            if frame_ctx.tags_that_responded_count == 0 and self.unverified_tags:
                if not self.final_check_mode:
                    self.final_check_mode = True
                else: # 如果最终检查模式下依然无响应，则全部标记为丢失
                    for tag in self.unverified_tags:
                        self.found_missing_ids.add(tag.id)
                    self.unverified_tags = []
        
        self.random_seed += 1
        self.current_state = self.STATE_IDLE
        self.active_frame_context = None


# ==============================================================================
# 3. 主程序入口
# ==============================================================================

if __name__ == '__main__':
    # 1. 定义全局仿真配置 (采用动态时间模型)
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

    # 2. 运行仿真
    print("--- 运行基线轮询算法 ---")
    baseline_results = run_missing_tag_simulation(
        global_config=global_config,
        algorithm_class=BaselinePollingAlgo,
        algorithm_specific_config={}
    )
    
    print("\n--- 运行 IIP 算法 (动态时间模型) ---")
    iip_results = run_missing_tag_simulation(
        global_config=global_config,
        algorithm_class=IIPAlgo,
        algorithm_specific_config={}
    )

    # 3. 打印结果
    print("\n" + "="*50)
    print("                      仿 真 结 果 对 比")
    print("="*50)
    print_results(baseline_results)
    print_results(iip_results)
    print("="*50)

    # 4. 性能对比总结
    if baseline_results and iip_results:
        baseline_time = baseline_results.get('total_protocol_time_us', float('inf'))
        iip_time = iip_results.get('total_protocol_time_us', float('inf'))

        if baseline_time > 1 and iip_time > 1:
            reduction_vs_base = (1 - iip_time / baseline_time) * 100
            print(f"\n时间效率: IIP 算法相比基线轮询，执行时间减少了 {reduction_vs_base:.2f}%")
        else:
            print("\n无法进行有意义的时间效率对比。")

