# -*- coding: utf-8 -*-
"""
此文件包含了 HPVT 算法，一个经过优化的新版 HPVTAdaptiveAlgo 算法，
以及一个经过校正的、能够正确计算 ALOHA 
时隙时间的基础仿真框架。
"""
import random
from collections import deque
from typing import List, Set, Tuple, Dict

# ==============================================================================
# 1. 基础框架 - 通用类和接口定义
# ==============================================================================
from Framework import *
# ==============================================================================
# 2. 基础框架 - 核心工具函数 (calculate_time_delta 已校正)
# ==============================================================================

def calculate_time_delta(step_result: AlgorithmStepResult, config: Dict) -> float:
    """
    !!已校正!!: 统一的时间计算器，能够正确处理ALOHA时隙的耗时。
    """
    if step_result.operation_type == 'idle':
        return 0.0

    bits_per_us = config['BITS_PER_MICROSECOND']
    t1 = config['T1_RTcal']
    t2 = config['T2_TRcal']
    guard_time = config['MINIMAL_GUARD_TIME_US']
    
    # --- 核心修正 ---
    # 为 ALOHA 时隙定义一个独立且正确的计算逻辑。
    # 一个ALOHA时隙的时间，是读写器等待一个潜在短响应所需的时间。
    if step_result.operation_type == 'ALOHA_SLOT':
        short_resp_bits = config.get('TAG_SHORT_RESP_BITS', 1)
        time_tag_response = short_resp_bits / bits_per_us
        # 即使时隙为空，也必须消耗掉这段“等待”时间。
        return time_tag_response + t2 + guard_time
    
    # --- 其他操作的时间计算逻辑保持不变 ---
    if step_result.reader_bits == 0:
        return 0.0

    time_reader_tx = step_result.reader_bits / bits_per_us
    
    if step_result.tag_bits > 0:
        time_tag_tx = step_result.tag_bits / bits_per_us
        return time_reader_tx + t1 + time_tag_tx + t2 + guard_time
    else:
        # 读写器发送后，至少等待T1时间来确认无响应
        return time_reader_tx + t1 + guard_time

# （此处省略 generate_scenario 和 run_missing_tag_simulation 的定义，
#  因为它们在您的基础框架中无需修改，我们将在主程序中直接调用）

# ==============================================================================
# 3. HPVT 算法实现 (原始版本)
# ==============================================================================

class _HPVT_Task:
    def __init__(self, prefix: str, tags_in_scope: List['Tag']):
        self.prefix, self.tags_in_scope = prefix, tags_in_scope

class _AlohaFrameContext:
    def __init__(self, task: _HPVT_Task, frame_size: int, random_seed: int):
        self.original_task, self.frame_size, self.random_seed = task, frame_size, random_seed
        self.current_slot = 0
        self.expected_slots = [[] for _ in range(frame_size)]
        self.actual_responses = [[] for _ in range(frame_size)]
        for tag in task.tags_in_scope:
            idx = hash(tag.id + str(random_seed)) % frame_size
            self.expected_slots[idx].append(tag)
            if tag.is_present: self.actual_responses[idx].append(tag)

class HPVTAlgo(MissingTagAlgorithmInterface):
    """
    分层并行验证树 (HPVT) 算法 (原始版本，用于对比)。
    """
    def initialize(self, expected_tags: List[Tag]):
        self.expected_tags_db = expected_tags
        self.found_present_ids, self.found_missing_ids = set(), set()
        self.task_queue = deque()
        self.STATE_IDLE, self.STATE_AWAITING_VERIFICATION, self.STATE_IN_ALOHA_FRAME = 0, 1, 2
        self.current_state = self.STATE_IDLE
        self.pending_task, self.active_aloha_frame = None, None
        if not expected_tags: return
        tags_0 = [t for t in expected_tags if t.id.startswith('0')]
        tags_1 = [t for t in expected_tags if t.id.startswith('1')]
        if tags_0: self.task_queue.append(_HPVT_Task('0', tags_0))
        if tags_1: self.task_queue.append(_HPVT_Task('1', tags_1))

    def is_finished(self) -> bool:
        return not self.task_queue and self.current_state == self.STATE_IDLE

    def get_results(self) -> Tuple[Set[str], Set[str]]:
        if not self.is_finished():
            return self.found_present_ids, self.found_missing_ids
        else:
            all_ids = {t.id for t in self.expected_tags_db}
            identified = self.found_present_ids.union(self.found_missing_ids)
            self.found_missing_ids.update(all_ids - identified)
            return self.found_present_ids, self.found_missing_ids

    def perform_step(self) -> AlgorithmStepResult:
        if self.is_finished(): return AlgorithmStepResult(operation_type="finished")
        if self.current_state == self.STATE_IN_ALOHA_FRAME: return self._perform_aloha_slot()
        if self.current_state == self.STATE_AWAITING_VERIFICATION: return self._setup_parallel_verification()
        if self.task_queue: return self._perform_group_probe()
        return AlgorithmStepResult(operation_type="idle")

    def _perform_group_probe(self) -> AlgorithmStepResult:
        task = self.task_queue.popleft()
        reader_bits = self.config['READER_CMD_BASE_BITS'] + len(task.prefix)
        responding = [t for t in task.tags_in_scope if t.is_present]
        if not responding:
            self.found_missing_ids.update({t.id for t in task.tags_in_scope})
            return AlgorithmStepResult('PROBE', reader_bits, 0, f"Probe '{task.prefix}': Idle")
        else:
            self.current_state = self.STATE_AWAITING_VERIFICATION
            self.pending_task = task
            return AlgorithmStepResult('PROBE', reader_bits, self.config['TAG_SHORT_RESP_BITS'], f"Probe '{task.prefix}': Active")

    def _setup_parallel_verification(self) -> AlgorithmStepResult:
        task = self.pending_task; self.pending_task = None
        frame_size = len(task.tags_in_scope) or 1
        self.active_aloha_frame = _AlohaFrameContext(task, frame_size, random.randint(0, 10000))
        self.current_state = self.STATE_IN_ALOHA_FRAME
        reader_bits = self.config['READER_CMD_BASE_BITS'] + 32
        return AlgorithmStepResult('ALOHA_SETUP', reader_bits, 0, f"Setup f={frame_size}")

    def _perform_aloha_slot(self) -> AlgorithmStepResult:
        ctx = self.active_aloha_frame
        tag_bits = len(ctx.actual_responses[ctx.current_slot]) * self.config['TAG_SHORT_RESP_BITS']
        ctx.current_slot += 1
        if ctx.current_slot >= ctx.frame_size: self._finalize_aloha_frame()
        return AlgorithmStepResult('ALOHA_SLOT', 0, tag_bits, f"Slot {ctx.current_slot}/{ctx.frame_size}")

    def _finalize_aloha_frame(self):
        ctx = self.active_aloha_frame
        unresolved = set()
        for i in range(ctx.frame_size):
            expected, responded = ctx.expected_slots[i], ctx.actual_responses[i]
            if len(expected) == 1:
                if len(responded) == 1: self.found_present_ids.add(expected[0].id)
                else: self.found_missing_ids.add(expected[0].id)
            elif len(expected) > 1:
                if len(responded) == 1:
                    self.found_present_ids.add(responded[0].id)
                    for t in expected:
                        if t.id != responded[0].id: self.found_missing_ids.add(t.id)
                else:
                    unresolved.update(expected)
        if unresolved and len(ctx.original_task.prefix) < self.config['BINARY_LENGTH']:
            p = ctx.original_task.prefix; p0, p1 = p + '0', p + '1'
            tags0 = [t for t in unresolved if t.id.startswith(p0)]
            tags1 = [t for t in unresolved if t.id.startswith(p1)]
            if tags0: self.task_queue.append(_HPVT_Task(p0, tags0))
            if tags1: self.task_queue.append(_HPVT_Task(p1, tags1))
        self.active_aloha_frame = None; self.current_state = self.STATE_IDLE

# ==============================================================================
# 4. HPVTAdaptiveAlgo 算法实现 (!!新增的优化版本!!)
# ==============================================================================
class HPVTAdaptiveAlgo(HPVTAlgo):
    """
    HPVT 的自适应优化版本。
    引入阈值，仅对小规模标签集启动ALOHA并行验证，对大规模集则继续进行树分裂。
    """
    def initialize(self, expected_tags: List[Tag]):
        # 调用父类的初始化方法
        super().initialize(expected_tags)
        # 获取自适应算法的特定参数
        self.aloha_threshold = self.config.get('ALOHA_THRESHOLD', 140)

    def _perform_group_probe(self) -> AlgorithmStepResult:
        """
        !!已重写!!: 这是实现自适应策略的核心。
        """
        task = self.task_queue.popleft()
        reader_bits = self.config['READER_CMD_BASE_BITS'] + len(task.prefix)
        responding = [t for t in task.tags_in_scope if t.is_present]

        if not responding:
            # Case 1: 分支无响应 (剪枝)，这是最高效的发现丢失标签的方式
            self.found_missing_ids.update({t.id for t in task.tags_in_scope})
            return AlgorithmStepResult('PROBE', reader_bits, 0, f"Probe '{task.prefix}': Idle")
        
        # Case 2: 分支有响应
        # 检查当前分支的标签总数是否小于阈值
        if len(task.tags_in_scope) <= self.aloha_threshold:
            # 分支足够小，适合启动ALOHA并行验证
            self.current_state = self.STATE_AWAITING_VERIFICATION
            self.pending_task = task
            # 传递实际响应者数量，用于优化帧大小
            setattr(self.pending_task, 'actual_responders_count', len(responding))
            return AlgorithmStepResult('PROBE', reader_bits, self.config['TAG_SHORT_RESP_BITS'], f"Probe '{task.prefix}': Active, entering ALOHA")
        else:
            # 分支太大，不适合ALOHA，继续进行二进制分裂
            if len(task.prefix) < self.config['BINARY_LENGTH']:
                p0, p1 = task.prefix + '0', task.prefix + '1'
                tags0 = [t for t in task.tags_in_scope if t.id.startswith(p0)]
                tags1 = [t for t in task.tags_in_scope if t.id.startswith(p1)]
                # 使用 appendleft 实现深度优先搜索，尽快深入树的底层
                if tags1: self.task_queue.appendleft(_HPVT_Task(p1, tags1))
                if tags0: self.task_queue.appendleft(_HPVT_Task(p0, tags0))
            # 本次操作依然消耗了一次探测的时间，但状态机不变，继续处理队列中的新任务
            return AlgorithmStepResult('PROBE', reader_bits, self.config['TAG_SHORT_RESP_BITS'], f"Probe '{task.prefix}': Active, splitting")

    def _setup_parallel_verification(self) -> AlgorithmStepResult:
        """
        !!已优化!!: 使用探测阶段得到的实际响应数来设置帧大小。
        """
        task = self.pending_task; self.pending_task = None
        # 如果有预估的响应数则使用，否则用子集大小作为后备
        n_actual = getattr(task, 'actual_responders_count', len(task.tags_in_scope))
        frame_size = n_actual or 1 # 确保帧大小至少为1
        
        self.active_aloha_frame = _AlohaFrameContext(task, frame_size, random.randint(0, 10000))
        self.current_state = self.STATE_IN_ALOHA_FRAME
        reader_bits = self.config['READER_CMD_BASE_BITS'] + 32
        return AlgorithmStepResult('ALOHA_SETUP', reader_bits, 0, f"Setup Adaptive ALOHA f={frame_size}")

# ==============================================================================
# 5. 主程序入口
# ==============================================================================

if __name__ == '__main__':
    # 假设您的项目中存在一个 Framework.py 文件，它提供了以下函数和类：
    from Framework import (
        BaselinePollingAlgo, run_missing_tag_simulation, print_results
    )

    # 1. 定义全局仿真配置
    global_config = {
        'TOTAL_TAGS': 2000, 'MISSING_RATE': 0.1, 'BINARY_LENGTH': 96,
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
    
    print("\n--- 运行原始HPVT算法 ---")
    hpvt_original_results = run_missing_tag_simulation(
        global_config=global_config,
        algorithm_class=HPVTAlgo,
        algorithm_specific_config={}
    )

    print("\n--- 运行自适应HPVT算法 ---")
    # 为自适应算法设置特定的阈值参数
    hpvt_adaptive_config = {'ALOHA_THRESHOLD': 32} # 当分组内标签数小于等于32时，启动ALOHA
    hpvt_adaptive_results = run_missing_tag_simulation(
        global_config=global_config,
        algorithm_class=HPVTAdaptiveAlgo,
        algorithm_specific_config=hpvt_adaptive_config
    )

    # 3. 打印结果
    print("\n" + "="*50 + "\n                      仿 真 结 果 对 比\n" + "="*50)
    print_results(baseline_results)
    print_results(hpvt_original_results)
    print_results(hpvt_adaptive_results)
    print("="*50)
