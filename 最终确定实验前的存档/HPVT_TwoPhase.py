# -*- coding: utf-8 -*-
"""
此文件实现了 HPVTTwoPhaseAlgo (两阶段自适应HPVT算法)。

该算法通过一个动态的两阶段策略，旨在同时优化“首次发现时间”和“总耗时”。
"""
import random
from collections import deque
from typing import List, Set, Tuple, Dict

# 基础框架的引用
# 假设您的项目中存在一个 Framework.py 文件
from Framework import MissingTagAlgorithmInterface, AlgorithmStepResult, Tag, run_missing_tag_simulation, print_results, BaselinePollingAlgo
# 为了进行对比，我们也需要原始的HPVTAdaptiveAlgo
from HPVT_Adaptive import HPVTAdaptiveAlgo
from hpvt_algo import _HPVT_Task, _AlohaFrameContext


# ==============================================================================
# 1. 两阶段自适应 HPVT 算法实现
# ==============================================================================
class HPVTTwoPhaseAlgo(MissingTagAlgorithmInterface):
    """
    HPVT 的两阶段自适应优化版本。

    它首先采用低阈值进行“快速侦察”，在找到第一个丢失标签后，
    立即切换到最优阈值进行“高效清扫”，从而兼顾“首次发现时间”和“总耗时”。
    """
    def initialize(self, expected_tags: List[Tag]):
        """初始化算法，并设置两阶段策略所需的参数。"""
        self.expected_tags_db = expected_tags
        self.found_present_ids = set()
        self.found_missing_ids = set()
        self.task_queue = deque()

        # 状态机定义
        self.STATE_IDLE, self.STATE_AWAITING_VERIFICATION, self.STATE_IN_ALOHA_FRAME = 0, 1, 2
        self.current_state = self.STATE_IDLE
        self.pending_task = None
        self.active_aloha_frame = None
        
        # --- 两阶段策略的核心参数 ---
        # 从配置中读取侦察阶段和清扫阶段的阈值
        self.recon_threshold = self.config.get('RECON_THRESHOLD', 32)
        self.cleanup_threshold = self.config.get('CLEANUP_THRESHOLD', 128)
        self.is_recon_phase = True # 算法启动时处于侦察阶段
        
        if not self.expected_tags_db: return
        tags_0 = [t for t in self.expected_tags_db if t.id.startswith('0')]
        tags_1 = [t for t in self.expected_tags_db if t.id.startswith('1')]
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
            
    def _check_and_switch_phase(self):
        """检查是否应从侦察阶段切换到清扫阶段。"""
        # 仅当处于侦察阶段且首次发现丢失标签时，执行一次切换
        if self.is_recon_phase and self.found_missing_ids:
            self.is_recon_phase = False
            # (可选的调试信息) print(f"--- Phase Switched to Cleanup (Threshold: {self.cleanup_threshold}) ---")

    def perform_step(self) -> AlgorithmStepResult:
        if self.is_finished(): return AlgorithmStepResult(operation_type="finished")
        if self.current_state == self.STATE_IN_ALOHA_FRAME: return self._perform_aloha_slot()
        if self.current_state == self.STATE_AWAITING_VERIFICATION: return self._setup_parallel_verification()
        if self.task_queue: return self._perform_group_probe()
        return AlgorithmStepResult(operation_type="idle")

    def _perform_group_probe(self) -> AlgorithmStepResult:
        """
        已重写以实现两阶段策略：根据当前阶段选择不同的阈值。
        """
        task = self.task_queue.popleft()
        reader_bits = self.config['READER_CMD_BASE_BITS'] + len(task.prefix)
        responding = [t for t in task.tags_in_scope if t.is_present]

        if not responding:
            self.found_missing_ids.update({t.id for t in task.tags_in_scope})
            self._check_and_switch_phase() # 发现丢失标签，检查是否需要切换阶段
            return AlgorithmStepResult('PROBE', reader_bits, 0, f"Probe '{task.prefix}': Idle")
        
        # 根据当前所处阶段，动态选择使用的阈值
        current_threshold = self.recon_threshold if self.is_recon_phase else self.cleanup_threshold
        
        if len(task.tags_in_scope) <= current_threshold:
            self.current_state = self.STATE_AWAITING_VERIFICATION
            self.pending_task = task
            setattr(self.pending_task, 'actual_responders_count', len(responding))
            return AlgorithmStepResult('PROBE', reader_bits, self.config['TAG_SHORT_RESP_BITS'], f"Probe '{task.prefix}': Active, entering ALOHA")
        else:
            if len(task.prefix) < self.config['BINARY_LENGTH']:
                p0, p1 = task.prefix + '0', task.prefix + '1'
                tags0 = [t for t in task.tags_in_scope if t.id.startswith(p0)]
                tags1 = [t for t in task.tags_in_scope if t.id.startswith(p1)]
                if tags1: self.task_queue.appendleft(_HPVT_Task(p1, tags1))
                if tags0: self.task_queue.appendleft(_HPVT_Task(p0, tags0))
            return AlgorithmStepResult('PROBE', reader_bits, self.config['TAG_SHORT_RESP_BITS'], f"Probe '{task.prefix}': Active, splitting")

    def _setup_parallel_verification(self) -> AlgorithmStepResult:
        task = self.pending_task; self.pending_task = None
        n_actual = getattr(task, 'actual_responders_count', len(task.tags_in_scope))
        frame_size = n_actual or 1
        self.active_aloha_frame = _AlohaFrameContext(task, frame_size, random.randint(0, 10000))
        self.current_state = self.STATE_IN_ALOHA_FRAME
        reader_bits = self.config['READER_CMD_BASE_BITS'] + 32
        return AlgorithmStepResult('ALOHA_SETUP', reader_bits, 0, f"Setup ALOHA f={frame_size}")

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
        
        self._check_and_switch_phase() # 在一帧结束后，也检查是否需要切换阶段
        
        if unresolved and len(ctx.original_task.prefix) < self.config['BINARY_LENGTH']:
            p = ctx.original_task.prefix; p0, p1 = p + '0', p + '1'
            tags0 = [t for t in unresolved if t.id.startswith(p0)]; tags1 = [t for t in unresolved if t.id.startswith(p1)]
            if tags1: self.task_queue.appendleft(_HPVT_Task(p1, tags1))
            if tags0: self.task_queue.appendleft(_HPVT_Task(p0, tags0))
        
        self.active_aloha_frame = None; self.current_state = self.STATE_IDLE

# ==============================================================================
#  主程序入口
# ==============================================================================
if __name__ == '__main__':
    print("开始 HPVT 两阶段自适应算法的性能验证...")

    # 1. 定义固定的全局仿真配置
    base_global_config = {
        'TOTAL_TAGS': 10000,
        'MISSING_RATE': 0.5,
        'BINARY_LENGTH': 96,
        'DATA_RATE_BPS': 160000.0,
        'BITS_PER_MICROSECOND': 160000.0 / 1.0e6,
        'T1_RTcal': 25.0,
        'T2_TRcal': 25.0,
        'MINIMAL_GUARD_TIME_US': 50.0,
        'READER_CMD_BASE_BITS': 37,
        'TAG_SHORT_RESP_BITS': 1,
    }
    
    # 2. 运行单阈值自适应算法作为对比基准
    print("\n--- 运行单阈值自适应算法 (最优参数) ---")
    # 使用您从调优脚本中找到的最优阈值
    hpvt_adaptive_config = {'ALOHA_THRESHOLD': 128}
    adaptive_results = run_missing_tag_simulation(
        global_config=base_global_config,
        algorithm_class=HPVTAdaptiveAlgo,
        algorithm_specific_config=hpvt_adaptive_config
    )
    
    # 3. 运行新的两阶段自适应算法
    print("\n--- 运行两阶段自适应算法 ---")
    two_phase_config = {
        'RECON_THRESHOLD': 32,    # 侦察阶段使用低阈值，追求“敏捷性”
        'CLEANUP_THRESHOLD': 128  # 清扫阶段使用最优阈值，追求“总效率”
    }
    two_phase_results = run_missing_tag_simulation(
        global_config=base_global_config,
        algorithm_class=HPVTTwoPhaseAlgo,
        algorithm_specific_config=two_phase_config
    )

    # 4. 打印和对比结果
    print("\n" + "="*80)
    print(f"{'最终性能对比':^80}")
    print("="*80)
    print_results(adaptive_results)
    print_results(two_phase_results)
    print("="*80)

    # 提取关键指标进行直接对比
    time_adaptive = adaptive_results.get('total_protocol_time_us', 0)
    tfm_adaptive = adaptive_results.get('time_to_first_missing', 0)
    
    time_two_phase = two_phase_results.get('total_protocol_time_us', 0)
    tfm_two_phase = two_phase_results.get('time_to_first_missing', 0)

    print("\n性能提升分析:")
    if time_adaptive > 0 and time_two_phase > 0:
        time_change = ((time_two_phase - time_adaptive) / time_adaptive) * 100
        print(f"总耗时变化: {time_change:+.2f}%")

    if tfm_adaptive > 0 and tfm_two_phase > 0:
        tfm_change = ((tfm_two_phase - tfm_adaptive) / tfm_adaptive) * 100
        print(f"首次发现时间变化: {tfm_change:+.2f}%")

    print("\n两阶段自适应算法验证完毕。")
