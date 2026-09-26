# -*- coding: utf-8 -*-
"""
IIP (Iterative ID-free Protocol) 算法的实现，已适配新框架。

此版本根据 Tao Li 等人的经典论文 "Identifying the Missing Tags in a Large RFID System" 
的核心思想进行实现，并完全适配重构后的仿真框架。
"""

import math
import random
from typing import List, Set, Tuple, Dict

# 导入新框架的基类、数据结构和全局常量
from framework import (
    MissingTagAlgorithmInterface,
    Tag,
    AlgorithmStepResult,
    CONSTANTS
)

# ==============================================================================
# 1. IIP 算法内部辅助类
# ==============================================================================

class _IIPFrameContext:
    """用于存储当前正在执行的 IIP ALOHA 帧的上下文信息。"""
    def __init__(self, unverified_tags: List['Tag'], frame_size: int, random_seed: int, final_check_mode: bool):
        self.frame_size = frame_size
        self.random_seed = random_seed
        self.final_check_mode = final_check_mode
        self.current_slot = 0
        
        # 预计算期望时隙和 pre-frame 向量
        self.expected_slots: List[List[Tag]] = [[] for _ in range(frame_size)]
        for tag in unverified_tags:
            # 使用一个简单的哈希函数来模拟 H(id, r)
            slot_index = hash(tag.id + str(random_seed)) % frame_size
            self.expected_slots[slot_index].append(tag)
        
        # 预计算真实响应
        self.actual_responses: List[List[Tag]] = [[] for _ in range(frame_size)]
        present_unverified_tags = [tag for tag in unverified_tags if tag.is_present]
        
        for tag in present_unverified_tags:
            slot_index = hash(tag.id + str(self.random_seed)) % frame_size
            should_respond = True
            
            # 模拟IIP协议的碰撞时隙二次哈希抑制机制
            # 如果一个时隙期望有多个标签（碰撞），则标签有50%的概率不响应
            if not self.final_check_mode and len(self.expected_slots[slot_index]) > 1:
                # 使用另一个哈希来模拟50%的概率
                if hash(tag.id + str(self.random_seed) + "second_hash") % 2 == 0:
                    should_respond = False
            
            if should_respond:
                self.actual_responses[slot_index].append(tag)

# ==============================================================================
# 2. IIP 算法实现 (适配新框架)
# ==============================================================================

class IIP_Algo(MissingTagAlgorithmInterface):
    """
    IIP (Iterative ID-free Protocol) 算法实现。
    """
    
    # 定义算法内部状态
    STATE_IDLE = "IDLE"
    STATE_IN_FRAME = "IN_FRAME"

    def __init__(self):
        """
        【已重构】构造函数非常简洁，不接收任何外部配置。
        所有参数都在算法内部定义或计算。
        """
        super().__init__()
        # IIP论文中提出的最优负载因子
        self.optimal_rho = 1.516
        # 调用初始化方法来设置所有状态变量
        self.initialize([])

    def initialize(self, expected_tags: List[Tag]):
        """初始化或重置算法的所有状态变量。"""
        super().initialize(expected_tags)
        self.unverified_tags = list(self.expected_tags_db)
        self.found_present_ids: Set[str] = set()
        self.found_missing_ids: Set[str] = set()
        
        self.random_seed = 0
        self.final_check_mode = False
        self.current_state = self.STATE_IDLE
        self.active_frame_context: _IIPFrameContext = None

    def is_finished(self) -> bool:
        """当不再有未验证的标签且状态机空闲时，算法结束。"""
        return not self.unverified_tags and self.current_state == self.STATE_IDLE

    def get_results(self) -> Tuple[Set[str], Set[str]]:
        """
        返回逐步确认的集合。在仿真结束后进行最终清算。
        """
        # 为了支持过程质量追踪，我们需要一个临时的、非破坏性的结果
        # “最佳猜测”的丢失集 = 总集 - 已确认在场的集合
        all_db_ids = {t.id for t in self.expected_tags_db}
        predicted_missing = all_db_ids - self.found_present_ids
        return self.found_present_ids, predicted_missing

    def perform_step(self) -> AlgorithmStepResult:
        """执行一个微步骤，要么是帧设置，要么是单时隙处理。"""
        if self.is_finished():
            return AlgorithmStepResult(operation_type="finished")
        
        if self.current_state == self.STATE_IN_FRAME:
            return self._perform_iip_slot()
        
        # 否则，当前状态是 IDLE，需要设置新的一帧
        return self._setup_iip_frame()

    def _setup_iip_frame(self) -> AlgorithmStepResult:
        """
        执行“帧设置”微步骤。根据论文，此步骤包含两次向量广播。
        """
        n_star = len(self.unverified_tags)
        if n_star == 0:
            self.current_state = self.STATE_IDLE
            return AlgorithmStepResult(operation_type="idle")

        # 使用论文中的最优负载因子计算帧长
        frame_size = int(round(n_star / self.optimal_rho)) if self.optimal_rho > 0 else n_star
        frame_size = max(frame_size, 1)

        self.active_frame_context = _IIPFrameContext(self.unverified_tags, frame_size, self.random_seed, self.final_check_mode)
        self.current_state = self.STATE_IN_FRAME
        
        # 【核心修改】直接从全局常量获取基础指令开销
        # 读写器开销是两次广播（pre-frame 和 post-frame vector）。
        # 论文中提到，如果向量过长，会分段在tag_slot中传输，但为简化，
        # 我们在此假设一次广播的成本是基础指令+向量长度。
        # 您的旧代码实现是 2 * (base + f)，我们沿用这个合理的简化。
        reader_bits = 2 * (CONSTANTS.READER_CMD_BASE_BITS + frame_size)
        
        op_desc = f"IIP Frame Setup: N*={n_star}, f={frame_size}"
        return AlgorithmStepResult(
            operation_type='IIP_SETUP',
            reader_bits=reader_bits,
            tag_bits=0, # 标签此步不响应
            operation_description=op_desc
        )

    def _perform_iip_slot(self) -> AlgorithmStepResult:
        """执行“单时隙处理”微步骤。"""
        frame_ctx = self.active_frame_context
        slot_idx = frame_ctx.current_slot
        
        responded_in_slot = frame_ctx.actual_responses[slot_idx]
        expected_in_slot = frame_ctx.expected_slots[slot_idx]
        
        # 算法逻辑：根据时隙结果更新已发现/丢失集合
        if len(responded_in_slot) == 1:
            # 成功时隙 -> 确认在场
            identified_tag = responded_in_slot[0]
            self.found_present_ids.add(identified_tag.id)
        elif len(responded_in_slot) == 0 and len(expected_in_slot) == 1:
            # 期望singleton但实际empty -> 确认丢失
            missing_tag = expected_in_slot[0]
            self.found_missing_ids.add(missing_tag.id)
        
        # 【核心修改】直接从全局常量获取标签短响应的比特数
        tag_bits = len(responded_in_slot) * CONSTANTS.TAG_SHORT_RESP_BITS

        frame_ctx.current_slot += 1
        # 如果一帧处理完毕，则进行帧的最终化处理
        if frame_ctx.current_slot >= frame_ctx.frame_size:
            self._finalize_iip_frame()
            
        op_desc = f"IIP Slot {slot_idx+1}/{frame_ctx.frame_size}"
        return AlgorithmStepResult(
            operation_type='ALOHA_SLOT',
            reader_bits=0, # 读写器在时隙间的开销已计入SETUP
            tag_bits=tag_bits,
            operation_description=op_desc
        )

    def _finalize_iip_frame(self):
        """在一轮IIP帧结束后，更新状态。此方法本身不产生通信开销。"""
        # 确定本轮解决了哪些标签
        resolved_ids_this_frame = self.found_present_ids.union(self.found_missing_ids)
        
        # 从待验证集合中移除已解决的标签
        initial_unverified_count = len(self.unverified_tags)
        self.unverified_tags = [t for t in self.unverified_tags if t.id not in resolved_ids_this_frame]
        
        # 检查是否需要进入或退出 final_check_mode
        if len(self.unverified_tags) == initial_unverified_count:
            # 如果一整轮没有任何新发现
            tags_responded_in_frame = sum(len(s) for s in self.active_frame_context.actual_responses)
            if tags_responded_in_frame == 0 and self.unverified_tags:
                # 且没有任何标签响应
                if not self.final_check_mode:
                    # 第一次发生，进入最终检查模式
                    self.final_check_mode = True
                else:
                    # 已经是最终检查模式，但仍然没有任何响应，
                    # 说明所有剩余的未验证标签都确定为丢失
                    for tag in self.unverified_tags:
                        self.found_missing_ids.add(tag.id)
                    self.unverified_tags = []
        else:
            # 只要有新发现，就退出最终检查模式
            self.final_check_mode = False
        
        self.random_seed += 1
        self.current_state = self.STATE_IDLE
        self.active_frame_context = None
