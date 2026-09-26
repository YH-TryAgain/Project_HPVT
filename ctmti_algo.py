# -*- coding: utf-8 -*-
"""
CTMTI (Collision Reconciling-based B-ary Tree Splitting Missing Tag Identification) 算法的实现。

此版本根据论文 "A Collision Reconciling-Based B-Ary Tree-Splitting Missing
Tag Identification Protocol" 的核心思想进行实现，并完全适配重构后的仿真框架。
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
# 1. CTMTI 算法实现 (适配新框架)
# ==============================================================================
class CTMTIAlgo(MissingTagAlgorithmInterface):
    """
    复现论文 "CTMTI" 的算法，已适配最新的仿真框架。
    """
    def __init__(self, alpha: float = 0.54, B: int = 2):
        """
        【已重构】构造函数现在只接收算法自身的超参数。
        """
        super().__init__()
        # 保存算法的特定配置
        self.alpha = alpha
        self.B = B
        # 调用初始化方法来设置所有状态变量
        self.initialize([])

    def initialize(self, expected_tags: List[Tag]):
        """初始化或重置算法的所有状态变量。"""
        super().initialize(expected_tags)
        
        # 为每个标签分配一个初始计数器Ac(T_t) = 1
        self.tag_counters = {tag.id: 1 for tag in expected_tags}
        
        self.unidentified_tags: Set[Tag] = set(self.expected_tags_db)
        self.predicted_present_ids: Set[str] = set()
        
        # CTMTI是逐轮次的，每一步执行一整轮
        self.round_count = 0

    def is_finished(self) -> bool:
        """当所有标签都被识别后，算法结束。"""
        return not self.unidentified_tags

    def get_results(self) -> Tuple[Set[str], Set[str]]:
        """
        返回已识别的在场和丢失标签ID集合。
        """
        # “最佳猜测”的丢失集 = 总集 - 已确认在场的集合
        all_db_ids = {t.id for t in self.expected_tags_db}
        predicted_missing = all_db_ids - self.predicted_present_ids
        return self.predicted_present_ids, predicted_missing

    def perform_step(self) -> AlgorithmStepResult:
        """
        执行一个完整的 CTMTI 轮次，并返回其总通信开销。
        在我们的仿真框架中，我们将一个完整的轮次视为一个“步骤”。
        """
        if self.is_finished():
            return AlgorithmStepResult(operation_type="finished")
        
        self.round_count += 1
        
        # --- 1. 设置本轮待处理的标签和哈希空间 ---
        R1 = random.random()
        num_elements = 0
        groups: Dict[int, List[Tag]] = {}

        if self.round_count == 1:
            # 第一轮：处理所有未识别标签
            tags_to_process = list(self.unidentified_tags)
            num_tags_to_process = len(tags_to_process)
            # 根据论文，f1 = alpha * K (K是待测标签总数)
            num_elements = math.ceil(self.alpha * num_tags_to_process)
            if num_elements == 0: num_elements = 1
            
            groups = {i: [] for i in range(num_elements)}
            for tag in tags_to_process:
                # E_{1,t} = H(ID_t, R1) mod f1
                h = hash((tag.id, R1, "Vp_hash")) % num_elements
                groups[h].append(tag)
        else:
            # 后续轮次：对上一轮未解决的碰撞组进行B-ary树分裂
            tags_to_process = list(self.unidentified_tags)
            num_irresolvable_groups = self._count_irresolvable_groups()
            
            num_elements = num_irresolvable_groups * self.B
            if num_elements == 0:
                self.unidentified_tags.clear()
                return AlgorithmStepResult(operation_type="finished")

            groups = {i: [] for i in range(num_elements)}
            
            # 简化版 B-ary 分裂
            # 论文中的 Ac(t) 映射更复杂，这里我们基于计数器值分组
            # 这是一个合理的简化，能体现算法核心思想
            irresolvable_tags_by_group = self._get_irresolvable_tags_by_group()
            group_idx_offset = 0
            for group_tags in irresolvable_tags_by_group.values():
                for tag in group_tags:
                    h = hash((tag.id, R1, "B_ary_split")) % self.B
                    groups[group_idx_offset + h].append(tag)
                group_idx_offset += self.B


        # --- 2. 碰撞调节 (Va) 和预期响应计算 ---
        R2 = random.random()
        unidentified_next_round = set()
        expected_responses: Dict[int, Tag] = {}
        X_01_total = 0
        X_10_count = 0
        
        num_irresolvable_elements_this_round = 0

        for i in range(num_elements):
            tags_in_elem = groups.get(i, [])
            if not tags_in_elem: continue
            
            if len(tags_in_elem) == 1:
                # 单标签元素 (01)
                tag = tags_in_elem[0]
                response_pos = X_01_total
                expected_responses[response_pos] = tag
                X_01_total += 1
                self.tag_counters[tag.id] = 0 # 标记为已解决
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
                            self.tag_counters[tag.id] = 0 # 标记为已解决
                    X_10_count += 1
                else:
                    # 不可解碰撞 (11)
                    num_irresolvable_elements_this_round += 1
                    for tag in tags_in_elem:
                        unidentified_next_round.add(tag)
                        # 更新 Ac(T_t)，使其指向下一个 B-ary 树的组
                        self.tag_counters[tag.id] = num_irresolvable_elements_this_round

        # --- 3. 仿真标签响应并识别 ---
        response_len = X_01_total + 3 * X_10_count
        
        # 统计在场标签的响应并标记
        for tag in expected_responses.values():
            if tag.is_present:
                self.predicted_present_ids.add(tag.id)
        
        # --- 4. 计算本轮的通信比特开销 ---
        # 论文中定义的读写器发送比特：控制参数 + Ve向量(2*m bits)
        reader_bits = CONSTANTS.READER_CMD_BASE_BITS + (2.0 * num_elements)
        # 标签响应比特：一个长度为 response_len 的比特串
        tag_bits = float(response_len)

        # --- 5. 更新状态进入下一轮 ---
        self.unidentified_tags = unidentified_next_round

        return AlgorithmStepResult(
            operation_type='CTMTI_ROUND',
            reader_bits=reader_bits,
            tag_bits=tag_bits,
            operation_description=f"CTMTI Round with {num_elements} elements, {len(self.unidentified_tags)} tags remain"
        )

    def _count_irresolvable_groups(self) -> int:
        """一个辅助函数，用于计算不可解组的数量。"""
        # Ac(t) > 0 的标签被认为是不可解的
        irresolvable_counters = {self.tag_counters.get(t.id) for t in self.unidentified_tags if self.tag_counters.get(t.id, 0) > 0}
        return len(irresolvable_counters)
        
    def _get_irresolvable_tags_by_group(self) -> Dict[int, List[Tag]]:
        """辅助函数，将不可解的标签按其 Ac(t) 计数器值分组。"""
        groups = {}
        for tag in self.unidentified_tags:
            ac_t = self.tag_counters.get(tag.id)
            if ac_t is not None and ac_t > 0:
                if ac_t not in groups:
                    groups[ac_t] = []
                groups[ac_t].append(tag)
        return groups
