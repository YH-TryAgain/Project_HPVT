# -*- coding: utf-8 -*-
"""
CR-MTI (Collision Resolving based Missing Tag Identification) 算法的实现，已适配新框架。

此版本根据论文 "An Efficient Missing Tag Identification Approach in RFID Collisions" 的
核心思想进行实现，并完全适配重构后的仿真框架。
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
# 1. CR-MTI 算法实现 (适配新框架)
# ==============================================================================
class CR_MTI_Algo(MissingTagAlgorithmInterface):
    """
    复现论文 "An Efficient Missing Tag Identification Approach in RFID Collisions"
    中的 CR-MTI 算法，已适配最新的仿真框架。
    """
    def __init__(self, lambda_opt: float = 15.0, w: int = 34):
        """
        【已重构】构造函数现在只接收算法自身的超参数。
        """
        super().__init__()
        # 保存算法的特定配置
        self.lambda_opt = lambda_opt
        self.w = w
        # 调用初始化方法来设置所有状态变量
        self.initialize([])

    def initialize(self, expected_tags: List[Tag]):
        """初始化或重置算法的所有状态变量。"""
        super().initialize(expected_tags)
        self.candidate_tags: Set[Tag] = set(self.expected_tags_db)
        self.identified_present_ids: Set[str] = set()
        self.identified_missing_ids: Set[str] = set()

        # round_plan 存储当前轮次需要执行的所有微操作
        self.round_plan: List[Tuple] = []

        # 如果有待测标签，则启动第一轮规划
        if self.candidate_tags:
            self._start_new_round()

    def is_finished(self) -> bool:
        """当候选标签集为空，且当前轮次的计划也执行完毕时，算法结束。"""
        return not self.candidate_tags and not self.round_plan

    def get_results(self) -> Tuple[Set[str], Set[str]]:
        """
        返回已识别的在场和丢失标签ID集合。
        为了支持过程质量追踪，这个方法需要高效且无副作用。
        """
        # “最佳猜测”的丢失集 = 总集 - 已确认在场的集合
        all_db_ids = {t.id for t in self.expected_tags_db}
        predicted_missing = all_db_ids - self.identified_present_ids
        return self.identified_present_ids, predicted_missing

    def perform_step(self) -> AlgorithmStepResult:
        """执行仿真的一步，即处理一个时隙的交互或一次广播。"""
        if not self.round_plan:
            # 如果当前轮次计划已空，但仍有候选标签，则开始新一轮
            if self.candidate_tags:
                self._start_new_round()
            
            # 如果新一轮后计划仍为空（例如剩余标签太少），则结束
            if not self.round_plan:
                # 在结束前，将所有剩余的候选标签都标记为丢失
                for tag in self.candidate_tags:
                    self.identified_missing_ids.add(tag.id)
                self.candidate_tags.clear()
                return AlgorithmStepResult(operation_type="finished")

        # 从计划中取出一个步骤并执行
        step_type, data = self.round_plan.pop(0)

        # --- 根据步骤类型，计算并返回该步骤的通信开销 ---
        reader_bits = 0
        tag_bits = 0
        op_desc = ""

        if step_type == 'BROADCAST_VI':
            # 步骤：广播Vi向量
            # data 中存储的是预计算好的Vi向量比特数
            # 论文中提到广播包含一个基础指令头
            reader_bits = CONSTANTS.READER_CMD_BASE_BITS + data
            op_desc = f"广播Vi向量 ({reader_bits} bits)"
            
        elif step_type == 'SINGLETON_SLOT':
            # 步骤：查询单例时隙
            tag: Tag = data
            # 读写器只需要发送一个短指令来轮询一个时隙
            reader_bits = CONSTANTS.READER_CMD_BASE_BITS
            if tag in self.candidate_tags:
                if tag.is_present:
                    tag_bits = CONSTANTS.TAG_SHORT_RESP_BITS
                    self.identified_present_ids.add(tag.id)
                else:
                    self.identified_missing_ids.add(tag.id)
                self.candidate_tags.remove(tag)
            op_desc = f"查询单例时隙 (Tag ...{tag.id[-6:]})"
            
        elif step_type == 'RESOLVABLE_SLOT':
            # 步骤：查询可解碰撞时隙
            tags_in_slot: List[Tag] = data
            # 同样，读写器只需要发送一个短指令
            reader_bits = CONSTANTS.READER_CMD_BASE_BITS
            present_tags_count = sum(1 for t in tags_in_slot if t.is_present)
            
            # 只有当至少有一个在场标签时，才会形成 w-bit 的响应
            if present_tags_count > 0:
                tag_bits = self.w
            
            # 无论响应如何，本时隙内所有标签的状态都会被确定
            for tag in tags_in_slot:
                if tag in self.candidate_tags:
                    if tag.is_present:
                        self.identified_present_ids.add(tag.id)
                    else:
                        self.identified_missing_ids.add(tag.id)
                    self.candidate_tags.remove(tag)
            op_desc = f"查询可解碰撞时隙 ({len(tags_in_slot)} 个标签)"

        else:
            return AlgorithmStepResult(operation_type="idle")

        return AlgorithmStepResult(
            operation_type=step_type,
            reader_bits=reader_bits,
            tag_bits=tag_bits,
            operation_description=op_desc
        )
        
    def _start_new_round(self):
        """
        规划新一轮的识别计划，此方法只生成计划，不计算时间。
        """
        num_candidates = len(self.candidate_tags)
        if num_candidates == 0:
            self.round_plan = []
            return

        # 计算主过滤器大小 f
        f = math.ceil(num_candidates / self.lambda_opt) if self.lambda_opt > 0 else num_candidates
        if f == 0:
            self.round_plan = []
            return

        # 生成本轮使用的随机种子
        r1 = str(random.random())
        r2 = str(random.random())

        # --- Phase 1: 将标签哈希到 f 个时隙中 ---
        slot_map: List[List[Tag]] = [[] for _ in range(f)]
        for tag in self.candidate_tags:
            slot_index = hash(tag.id + r1) % f
            slot_map[slot_index].append(tag)

        # --- Phase 2: 分析时隙类型并制定计划 ---
        temp_plan = []
        num_resolvable = 0
        num_irresolvable = 0
        num_singleton = 0
        num_empty = 0

        for tags_in_slot in slot_map:
            k = len(tags_in_slot)
            if k == 0:
                num_empty += 1
            elif k == 1:
                num_singleton += 1
                temp_plan.append(('SINGLETON_SLOT', tags_in_slot[0]))
            elif 1 < k <= self.w:
                # 尝试用第二个哈希函数解决碰撞
                hashed_positions = {hash(tag.id + r2) % self.w for tag in tags_in_slot}
                if len(hashed_positions) == k:
                    num_resolvable += 1
                    temp_plan.append(('RESOLVABLE_SLOT', tags_in_slot))
                else:
                    num_irresolvable += 1
            else: # k > w
                num_irresolvable += 1

        # --- Phase 3: 构建最终的执行计划 ---
        # 1. 根据论文，计算广播Vi向量的比特成本
        vi_vector_bits = (num_resolvable + num_irresolvable) * 1 + (num_singleton + num_empty) * 2
        
        # 2. 将广播操作作为本轮的第一个步骤
        self.round_plan = [('BROADCAST_VI', vi_vector_bits)]
        
        # 3. 将所有可识别的时隙查询操作加入计划
        self.round_plan.extend(temp_plan)
