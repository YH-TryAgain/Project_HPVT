# -*- coding: utf-8 -*-

import random
import time
import math
import numpy as np  # 导入numpy用于科学计算

from Framework import *

# ==============================================================================
# 6. 新增算法实现 (CR-MTI 论文复现)
# ==============================================================================
class CR_MTI_Algo(MissingTagAlgorithmInterface):
    """
    复现论文 'An Efficient Missing Tag Identification Approach in RFID Collisions'
    中的 CR-MTI 算法。
    该算法通过协调碰撞时隙，在一个时隙内同时验证多个标签，以提高效率。
    """
    def initialize(self, expected_tags: list[Tag]):
        """
        初始化算法状态，并准备开始第一轮识别。
        """
        # CR-MTI算法的特定参数从config中读取
        super().__init__({**self.config})
        
        # 待验证的候选标签集，会随着算法进行而减少
        self.candidate_tags: set[Tag] = set(expected_tags)
        
        # 存储最终识别结果
        self.identified_present_ids: set[str] = set()
        self.identified_missing_ids: set[str] = set()

        # 用于暂存当前轮次的所有仿真步骤（每个步骤对应一个时隙的交互）
        self.round_plan: list[tuple[AlgorithmStepResult, list[Tag], str]] = []

        # 启动第一轮
        self._start_new_round()

    def is_finished(self) -> bool:
        """
        当候选标签集为空，且当前轮次的计划也执行完毕时，算法结束。
        """
        return not self.candidate_tags and not self.round_plan

    def get_results(self) -> tuple[set[str], set[str]]:
        """
        返回已识别的在场和丢失标签ID集合。
        """
        return self.identified_present_ids, self.identified_missing_ids

    def perform_step(self) -> AlgorithmStepResult:
        """
        执行仿真的一步，即处理一个时隙的交互。
        """
        if self.is_finished():
            return AlgorithmStepResult(operation_description="finished")

        # 如果当前轮次计划已空，但仍有候选标签，则开始新一轮
        if not self.round_plan and self.candidate_tags:
            self._start_new_round()
            # 如果新一轮还是没计划（例如剩余标签太少），则清空剩余标签并结束
            if not self.round_plan:
                for tag in self.candidate_tags:
                    # 无法通过CR-MTI判断的，保守地认为是丢失的
                    self.identified_missing_ids.add(tag.id)
                self.candidate_tags.clear()
                return AlgorithmStepResult(operation_description="clearing remaining tags and finishing")

        # 从计划中取出一个步骤并执行
        step_result_tuple = self.round_plan.pop(0)

        # 检查元组的长度，兼容新旧两种计划格式
        if len(step_result_tuple) == 3:
            step_result, tags_in_slot, step_type = step_result_tuple
        else: # 处理不包含标签和类型的旧格式（仅用于广播Vi）
            step_result = step_result_tuple[0]
            return step_result

        # --- 核心识别逻辑 (对应论文 4.2.3 Present Tag Verifying Phase) ---
        if step_type == 'singleton':
            tag = tags_in_slot[0]
            if tag in self.candidate_tags: # 确保标签还在候选集中
                if tag.is_present:
                    self.identified_present_ids.add(tag.id)
                else: # 标签未响应，被识别为丢失
                    self.identified_missing_ids.add(tag.id)
                self.candidate_tags.remove(tag)

        elif step_type == 'resolvable_collision':
            # 在可解碰撞时隙，一次性判断该组中所有标签的状态
            for tag in tags_in_slot:
                if tag in self.candidate_tags:
                    if tag.is_present:
                        self.identified_present_ids.add(tag.id)
                    else: # 未参与构成最终响应信号的标签，被识别为丢失
                        self.identified_missing_ids.add(tag.id)
                    self.candidate_tags.remove(tag)
        
        return step_result
        
    def _start_new_round(self):
        """
        构建新一轮的识别计划，对应论文 4.2.1 和 4.2.2。
        【已修正】增加了广播Vi向量的成本计算。
        """
        num_candidates = len(self.candidate_tags)
        if num_candidates == 0:
            return

        # --- 参数优化 (对应论文 4.3 Parameter Optimization) ---
        lambda_opt = self.config.get('CR_MTI_LAMBDA', 15.0)
        self.w = self.config.get('CR_MTI_W', 34)
        
        self.f = math.ceil(num_candidates / lambda_opt) if lambda_opt > 0 else num_candidates
        if self.f == 0: return

        r1 = str(time.time() + random.random())
        r2 = str(time.time() + random.random() + 1)

        # --- Phase 1: Main Filter Vector Construction ---
        slot_map: list[list[Tag]] = [[] for _ in range(self.f)]
        for tag in self.candidate_tags:
            slot_index = hash(tag.id + r1) % self.f
            slot_map[slot_index].append(tag)

        # --- Phase 2: Collision Slot Reconciling & Vi Vector Cost Calculation ---
        temp_plan = []
        num_resolvable = 0
        num_irresolvable = 0
        num_singleton = 0
        num_empty = 0

        for tags_in_slot in slot_map:
            k = len(tags_in_slot)
            if k == 0:
                num_empty += 1
                continue
            elif k == 1:
                num_singleton += 1
                temp_plan.append(('singleton', tags_in_slot))
            elif 1 < k <= self.w:
                hashed_positions = {hash(tag.id + r2) % self.w for tag in tags_in_slot}
                if len(hashed_positions) == k:
                    num_resolvable += 1
                    temp_plan.append(('resolvable_collision', tags_in_slot))
                else:
                    num_irresolvable += 1
            else: # k > w
                num_irresolvable += 1

        # 【新增逻辑】计算广播Vi向量的成本
        # 根据论文4.2.2节的编码规则: 碰撞(可解/不可解)为1 bit, 单例/空闲为2 bits
        vi_vector_bits = (num_resolvable + num_irresolvable) * 1 + (num_singleton + num_empty) * 2
        
        bits_per_us = self.config.get('BITS_PER_MICROSECOND')
        time_for_vi = vi_vector_bits / bits_per_us
        
        # 将广播Vi作为一个步骤加入计划的开头
        broadcast_step = AlgorithmStepResult(
            time_delta_us=time_for_vi,
            reader_bits=vi_vector_bits,
            tag_bits=0,
            operation_description=f"广播Vi向量 ({vi_vector_bits} bits)"
        )
        self.round_plan = [(broadcast_step,)] # 使用元组包装以区分

        # --- 将识别计划转换为带有时间和比特开销的仿真步骤 ---
        for step_type, tags in temp_plan:
            bits_per_us = self.config.get('BITS_PER_MICROSECOND')
            t1 = self.config.get('T1_RTcal')
            t2 = self.config.get('T2_TRcal')
            
            reader_bits = self.config.get('READER_CMD_BASE_BITS')
            time_delta = reader_bits / bits_per_us + t1
            
            present_tags_count = sum(1 for t in tags if t.is_present)
            tag_bits = 0

            if present_tags_count > 0:
                if step_type == 'singleton':
                    resp_bits = self.config.get('TAG_SHORT_RESP_BITS')
                    tag_bits = resp_bits
                    time_delta += (resp_bits / bits_per_us) + t2
                elif step_type == 'resolvable_collision':
                    resp_bits = self.w
                    tag_bits = resp_bits
                    time_delta += (resp_bits / bits_per_us) + t2
            else:
                time_delta += t1 

            op_desc = f"查询 {step_type} 时隙，预期 {len(tags)} 个标签"
            self.round_plan.append((
                AlgorithmStepResult(time_delta, reader_bits, tag_bits, op_desc),
                tags,
                step_type
            ))

# ==============================================================================
# 7. 仿真执行入口
# ==============================================================================
if __name__ == '__main__':
    # --- 严格基于框架开头的常量来创建全局配置 ---
    # 这样可以确保所有算法都在完全相同的物理条件下进行比较
    global_simulation_config = {
        'TOTAL_TAGS': DEFAULT_TOTAL_TAGS,
        'MISSING_RATE': DEFAULT_MISSING_RATE,
        'BINARY_LENGTH': DEFAULT_BINARY_LENGTH,
        'DATA_RATE_BPS': DEFAULT_DATA_RATE_BPS,
        'BITS_PER_MICROSECOND': DEFAULT_BITS_PER_MICROSECOND,
        'T1_RTcal': DEFAULT_T1_RTcal,
        'T2_TRcal': DEFAULT_T2_TRcal,
        'READER_CMD_BASE_BITS': DEFAULT_READER_CMD_BASE_BITS,
        'TAG_SHORT_RESP_BITS': DEFAULT_TAG_SHORT_RESP_BITS,
    }

    # --- 运行基线算法 ---
    print("="*50)
    print("运行基线轮询算法 (BaselinePollingAlgo)...")
    baseline_results = run_missing_tag_simulation(
        global_config=global_simulation_config,
        algorithm_class=BaselinePollingAlgo,
        algorithm_specific_config={} # 基线算法无特殊参数
    )
    print_results(baseline_results)
    
    # --- 运行 CR-MTI 算法 ---
    print("\n" + "="*50)
    print("运行论文复现算法 (CR_MTI_Algo)...")
    # CR-MTI 算法的特定超参数
    crmti_specific_config = {
        'CR_MTI_LAMBDA': 15.0,  # 论文推荐值
        'CR_MTI_W': 34,         # 论文推荐值
    }
    crmti_results = run_missing_tag_simulation(
        global_config=global_simulation_config,
        algorithm_class=CR_MTI_Algo,
        algorithm_specific_config=crmti_specific_config
    )
    print_results(crmti_results)
