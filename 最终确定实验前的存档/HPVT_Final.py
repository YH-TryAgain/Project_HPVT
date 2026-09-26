# -*- coding: utf-8 -*-
"""
此文件实现了 HPVT_AggregatedAlgo，这是对 HPVT_Adaptive 的进一步优化。

该算法采用“查询聚合”或“批量探测”策略，将对兄弟节点的两次独立探测
合并为一次操作，从而共享一次指令头开销，旨在显著降低总能耗。
"""
import random
from collections import deque
from typing import List, Set, Tuple, Dict

# ==============================================================================
# 1. 导入依赖
# ==============================================================================
# 假设您的项目中存在以下文件和类
from Framework import *
from HPVT_Adaptive import HPVTAdaptiveAlgo
from hpvt_algo import _HPVT_Task, _AlohaFrameContext


# ==============================================================================
# 2. 查询聚合优化算法实现 (HPVT_AggregatedAlgo)
# ==============================================================================
class HPVT_AggregatedAlgo(HPVTAdaptiveAlgo):
    """
    HPVT 的查询聚合优化版本。

    它通过将对兄弟节点的探测合并为一次操作，来摊销指令头开销，
    从而在保持时间性能的同时，显著降低读写器的总能耗。
    """
    def initialize(self, expected_tags: List[Tag]):
        # 调用父类的初始化方法，以继承所有基础设置
        super().initialize(expected_tags)
        # 新增一个状态变量，用于追踪是否刚刚支付了共享开销
        self._shared_overhead_just_paid = False

    def _are_siblings(self, task1: _HPVT_Task, task2: _HPVT_Task) -> bool:
        """检查两个任务是否是“兄弟”节点（拥有相同的父前缀）。"""
        if not task1 or not task2:
            return False
        # 如果两个前缀长度相同，且除了最后一位外都相同，则它们是兄弟
        return (len(task1.prefix) == len(task2.prefix) and
                task1.prefix[:-1] == task2.prefix[:-1])

    def _perform_group_probe(self) -> AlgorithmStepResult:
        """
        !!已重写以实现查询聚合!!
        """
        # --- 1. 机会主义的聚合检查 ---
        is_first_in_pair = False
        # 检查队列中是否有至少两个任务，并且它们是兄弟
        if len(self.task_queue) >= 2 and self._are_siblings(self.task_queue[0], self.task_queue[1]):
            # 如果是，并且我们没有刚刚支付过共享开销，那么这次探测将作为聚合对的“发起者”
            if not self._shared_overhead_just_paid:
                is_first_in_pair = True
                self._shared_overhead_just_paid = True # 标记我们即将支付共享开销

        # --- 2. 弹出并处理当前任务 ---
        task = self.task_queue.popleft()
        
        # --- 3. 核心修改：动态计算读写器能耗 ---
        reader_bits = 0
        if self._shared_overhead_just_paid and not is_first_in_pair:
            # 这是聚合对的“跟随者”，它享受了前一个兄弟节点支付的开销
            # 因此，它不需要再支付基础指令开销，只需支付自己的前缀数据即可
            reader_bits = len(task.prefix)
            self._shared_overhead_just_paid = False # 重置状态
        else:
            # 这是独立的探测，或者是聚合对的“发起者”，需要支付全部开销
            reader_bits = self.config['READER_CMD_BASE_BITS'] + len(task.prefix)
        
        # --- 4. 算法的其余逻辑保持不变 ---
        responding = [t for t in task.tags_in_scope if t.is_present]
        tag_bits = self.config.get('TAG_SHORT_RESP_BITS', 1) if responding else 0
        
        if not responding:
            self.found_missing_ids.update({t.id for t in task.tags_in_scope})
            op_desc = f"Agg-Probe '{task.prefix}': Idle"
        else:
            # 这里的逻辑与 HPVTAdaptiveAlgo 完全相同
            current_threshold = self.config.get('ALOHA_THRESHOLD', 128)
            if len(task.tags_in_scope) <= current_threshold:
                self.current_state = self.STATE_AWAITING_VERIFICATION
                self.pending_task = task
                setattr(self.pending_task, 'actual_responders_count', len(responding))
                op_desc = f"Agg-Probe '{task.prefix}': Active -> ALOHA"
            else:
                if len(task.prefix) < self.config['BINARY_LENGTH']:
                    p0, p1 = task.prefix + '0', task.prefix + '1'
                    tags0 = [t for t in task.tags_in_scope if t.id.startswith(p0)]
                    tags1 = [t for t in task.tags_in_scope if t.id.startswith(p1)]
                    # 关键：将兄弟任务连续地插入队列头部，为下一次聚合创造机会
                    if tags1: self.task_queue.appendleft(_HPVT_Task(p1, tags1))
                    if tags0: self.task_queue.appendleft(_HPVT_Task(p0, tags0))
                op_desc = f"Agg-Probe '{task.prefix}': Active -> Splitting"
                
        return AlgorithmStepResult('PROBE', reader_bits, tag_bits, op_desc)

# ==============================================================================
#  主程序入口
# ==============================================================================
if __name__ == '__main__':
    print("开始 HPVT 查询聚合优化算法的性能验证...")

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
    print("\n--- 运行 HPVT_Adaptive (单阈值自适应算法) ---")
    hpvt_adaptive_config = {'ALOHA_THRESHOLD': 128} # 使用已知的最优阈值
    adaptive_results = run_missing_tag_simulation(
        global_config=base_global_config,
        algorithm_class=HPVTAdaptiveAlgo,
        algorithm_specific_config=hpvt_adaptive_config
    )
    
    # 3. 运行新的查询聚合优化算法
    print("\n--- 运行 HPVT_Aggregated (查询聚合优化算法) ---")
    # 它也需要一个阈值，我们使用相同的最优值
    aggregated_config = {'ALOHA_THRESHOLD': 128}
    aggregated_results = run_missing_tag_simulation(
        global_config=base_global_config,
        algorithm_class=HPVT_AggregatedAlgo,
        algorithm_specific_config=aggregated_config
    )

    # 4. 打印和对比结果
    print("\n" + "="*80)
    print(f"{'最终能耗与性能对比':^80}")
    print("="*80)
    print_results(adaptive_results)
    print_results(aggregated_results)
    print("="*80)

    # 提取关键指标进行直接对比
    bits_adaptive = adaptive_results.get('total_reader_bits', 0) + adaptive_results.get('total_tag_bits', 0)
    time_adaptive = adaptive_results.get('total_protocol_time_us', 0)
    
    bits_aggregated = aggregated_results.get('total_reader_bits', 0) + aggregated_results.get('total_tag_bits', 0)
    time_aggregated = aggregated_results.get('total_protocol_time_us', 0)

    print("\n性能提升分析:")
    if bits_adaptive > 0 and bits_aggregated > 0:
        bits_change = ((bits_aggregated - bits_adaptive) / bits_adaptive) * 100
        print(f"总能耗变化: {bits_change:+.2f}%")

    if time_adaptive > 0 and time_aggregated > 0:
        time_change = ((time_aggregated - time_adaptive) / time_adaptive) * 100
        print(f"总耗时变化: {time_change:+.2f}%")

    print("\n查询聚合优化算法验证完毕。")
