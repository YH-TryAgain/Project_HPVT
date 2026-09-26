# -*- coding: utf-8 -*-
"""
此文件包含了适配了新的动态时间计算框架的 CR-MTI 算法，
并提供了一个独立的主程序入口来运行和测试其性能。
"""
import math
import random
import time
from typing import List, Set, Tuple, Dict

# 基础框架的引用
# 假设您的项目中存在一个 Framework.py 文件，其中定义了以下内容
from Framework import *

# ==============================================================================
# 1. CR-MTI 算法实现 (已适配新框架)
# ==============================================================================
class CR_MTI_Algo(MissingTagAlgorithmInterface):
    """
    复现论文 "An Efficient Missing Tag Identification Approach in RFID Collisions"
    中的 CR-MTI 算法，已适配动态时间计算框架。
    """
    def initialize(self, expected_tags: List[Tag]):
        """初始化算法状态。"""
        self.candidate_tags: Set[Tag] = set(expected_tags)
        self.identified_present_ids: Set[str] = set()
        self.identified_missing_ids: Set[str] = set()

        # round_plan 存储当前轮次需要执行的所有微操作
        self.round_plan: List[Tuple] = []

        # 启动第一轮规划
        self._start_new_round()

    def is_finished(self) -> bool:
        """当候选标签集为空，且当前轮次的计划也执行完毕时，算法结束。"""
        return not self.candidate_tags and not self.round_plan

    def get_results(self) -> Tuple[Set[str], Set[str]]:
        """返回已识别的在场和丢失标签ID集合。"""
        # 最后的清理：如果结束后仍有标签未被识别，保守地认为它们是丢失的
        if self.is_finished():
            remaining_ids = {t.id for t in self.candidate_tags}
            self.identified_missing_ids.update(remaining_ids)
            self.candidate_tags.clear()
        return self.identified_present_ids, self.identified_missing_ids

    def perform_step(self) -> AlgorithmStepResult:
        """执行仿真的一步，即处理一个时隙的交互或一次广播。"""
        if not self.round_plan:
            # 如果当前轮次计划已空，但仍有候选标签，则开始新一轮
            if self.candidate_tags:
                self._start_new_round()
            
            # 如果新一轮后计划仍为空（例如剩余标签太少），则结束
            if not self.round_plan:
                return AlgorithmStepResult(operation_type="finished")

        # 从计划中取出一个步骤并执行
        step_type, data = self.round_plan.pop(0)

        # --- 根据步骤类型，计算并返回该步骤的通信开销 ---
        reader_bits = self.config.get('READER_CMD_BASE_BITS', 37)
        tag_bits = 0

        if step_type == 'BROADCAST_VI':
            # 步骤：广播Vi向量
            reader_bits = data  # data 中存储的是预计算好的Vi向量比特数
            tag_bits = 0
            op_desc = f"广播Vi向量 ({reader_bits} bits)"
            
        elif step_type == 'SINGLETON_SLOT':
            # 步骤：查询单例时隙
            tag: Tag = data[0]
            if tag in self.candidate_tags:
                if tag.is_present:
                    tag_bits = self.config.get('TAG_SHORT_RESP_BITS', 1)
                    self.identified_present_ids.add(tag.id)
                else:
                    self.identified_missing_ids.add(tag.id)
                self.candidate_tags.remove(tag)
            op_desc = f"查询单例时隙 (Tag ...{tag.id[-6:]})"
            
        elif step_type == 'RESOLVABLE_SLOT':
            # 步骤：查询可解碰撞时隙
            tags_in_slot: List[Tag] = data
            present_tags_count = sum(1 for t in tags_in_slot if t.is_present)
            
            # 只有当至少有一个在场标签时，才会形成 w-bit 的响应
            if present_tags_count > 0:
                tag_bits = self.config.get('CR_MTI_W', 34)
            
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

        # --- 从配置中获取算法参数 ---
        lambda_opt = self.config.get('CR_MTI_LAMBDA', 15.0)
        self.w = self.config.get('CR_MTI_W', 34)
        
        # 计算主过滤器大小 f
        self.f = math.ceil(num_candidates / lambda_opt) if lambda_opt > 0 else num_candidates
        if self.f == 0:
            self.round_plan = []
            return

        # 生成本轮使用的随机种子
        r1 = str(random.random())
        r2 = str(random.random())

        # --- Phase 1: 将标签哈希到 f 个时隙中 ---
        slot_map: List[List[Tag]] = [[] for _ in range(self.f)]
        for tag in self.candidate_tags:
            slot_index = hash(tag.id + r1) % self.f
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
                temp_plan.append(('SINGLETON_SLOT', tags_in_slot))
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
        # 1. 计算广播Vi向量的比特成本
        vi_vector_bits = (num_resolvable + num_irresolvable) * 1 + (num_singleton + num_empty) * 2
        
        # 2. 将广播操作作为本轮的第一个步骤
        self.round_plan = [('BROADCAST_VI', vi_vector_bits)]
        
        # 3. 将所有可识别的时隙查询操作加入计划
        self.round_plan.extend(temp_plan)

# ==============================================================================
# 2. 主程序入口
# ==============================================================================

if __name__ == '__main__':
    # 1. 定义全局仿真配置 (采用动态时间模型)
    global_config = {
        'TOTAL_TAGS': 20000,
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

    # 2. 定义 CR-MTI 算法的特定参数
    crmti_specific_config = {
        'CR_MTI_LAMBDA': 15.0,  # 论文推荐值
        'CR_MTI_W': 34,         # 论文推荐值
    }

    # 3. 运行 CR-MTI 仿真
    print("\n--- 运行 CR-MTI 算法 (动态时间模型) ---")
    # 假设 run_missing_tag_simulation 和 print_results 已从框架导入
    crmti_results = run_missing_tag_simulation(
        global_config=global_config,
        algorithm_class=CR_MTI_Algo,
        algorithm_specific_config=crmti_specific_config
    )
    print_results(crmti_results)

    # --- (可选) 运行基线算法进行对比 ---
    print("\n--- 运行基线轮询算法作为对比 ---")
    baseline_results = run_missing_tag_simulation(
        global_config=global_config,
        algorithm_class=BaselinePollingAlgo,
        algorithm_specific_config={}
    )
    print_results(baseline_results)

