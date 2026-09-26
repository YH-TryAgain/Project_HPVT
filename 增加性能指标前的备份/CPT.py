# -*- coding: utf-8 -*-
"""
此文件包含了适配了新的动态时间计算框架的 CPT (Collision-Partition Tree) 算法，
并提供了一个独立的主程序入口来运行和测试其性能。
"""
import math
import random
from collections import deque
from typing import List, Set, Tuple, Dict

# 基础框架的引用
# 假设您的项目中存在一个 Framework.py 文件，其中定义了以下内容
from Framework import *


# ==============================================================================
# 1. CPT 算法实现 (get_results 方法已修复)
# ==============================================================================

class _CPTNode:
    """CPT树的内部节点类，用于构建树形结构。"""
    def __init__(self):
        self.is_leaf = False
        self.split_bit_index = -1
        self.child_0: '_CPTNode' = None
        self.child_1: '_CPTNode' = None
        self.tags: List[Tag] = []
        self.path_from_root: List[Tuple[int, str]] = []


class CPTAlgo(MissingTagAlgorithmInterface):
    """
    论文 'Revisiting RFID Missing Tag Identification' 中提出的 CPT 算法，
    已适配动态时间计算框架，并修复了时间线报告问题。
    """
    
    def initialize(self, expected_tags: List[Tag]):
        """
        初始化算法，完成CPT算法的所有离线准备工作（无通信开销）。
        """
        self.expected_tags_db = expected_tags
        self.found_present_ids: Set[str] = set()
        self.found_missing_ids: Set[str] = set()
        
        self.pseudo_id_map: Dict[str, str] = {}
        self.pseudo_id_len = 0
        self.query_queue: deque = deque()
        
        if not self.expected_tags_db:
            return
            
        # 步骤一：生成伪ID (离线计算)
        self._generate_pseudo_ids()

        # 步骤二：构建CPT树 (离线计算)
        self.root = self._build_cpt_recursive(self.expected_tags_db, [])
        
        # 步骤三：准备查询队列 (离线计算)
        self._collect_leaf_queries(self.root)

    def is_finished(self) -> bool:
        """当查询队列为空时，代表所有叶子节点都已盘点完毕，算法结束。"""
        return not self.query_queue

    def get_results(self) -> Tuple[Set[str], Set[str]]:
        """
        !!已修改!!: 返回一个更全面的临时报告以支持时间线统计。
        
        该方法现在区分“仿真中”和“仿真后”两种情况：
        - 仿真中: 返回一个临时的“最佳猜测”丢失集，以供框架追踪进度。
        - 仿真后: 执行最终清算，返回精确的结果用于最终评估。
        """
        if not self.is_finished():
            # --- 仿真中：提供临时报告 ---
            # 这个报告不改变算法的内部状态 (self.found_missing_ids)
            all_db_ids = {t.id for t in self.expected_tags_db}
            
            # “预测丢失集” = 所有标签 - 已确认在场的标签
            # 这是我们对当前状态的最佳猜测，用于时间线统计
            predicted_missing_for_timeline = all_db_ids - self.found_present_ids
            
            return self.found_present_ids, predicted_missing_for_timeline
        else:
            # --- 仿真结束：提供最终精确结果 ---
            all_db_ids = {t.id for t in self.expected_tags_db}
            identified_ids = self.found_present_ids.union(self.found_missing_ids)
            unconfirmed_ids = all_db_ids - identified_ids
            
            # 执行最终清算，将所有未确认的标签更新到内部状态中
            self.found_missing_ids.update(unconfirmed_ids)
            
            return self.found_present_ids, self.found_missing_ids


    def perform_step(self) -> AlgorithmStepResult:
        """
        执行一个查询步骤，返回查询该叶子节点所需的通信比特开销。
        """
        if self.is_finished():
            return AlgorithmStepResult(operation_type="finished")

        leaf_node = self.query_queue.popleft()
        
        # 1. 计算读写器发送的比特数
        path = leaf_node.path_from_root
        bits_for_path_len = math.ceil(math.log2(self.pseudo_id_len)) if self.pseudo_id_len > 1 else 1
        reader_bits = len(path) * (bits_for_path_len + 1)
        
        if len(leaf_node.tags) == 2:
            tag0_pid = self.pseudo_id_map[leaf_node.tags[0].id]
            tag1_pid = self.pseudo_id_map[leaf_node.tags[1].id]
            for i in range(self.pseudo_id_len):
                if tag0_pid[i] != tag1_pid[i]:
                    reader_bits += bits_for_path_len
                    break
        
        # 2. 模拟标签响应并计算标签响应的比特数
        responding_tags = [tag for tag in leaf_node.tags if tag.is_present]
        tag_bits = len(responding_tags) * self.config.get('TAG_SHORT_RESP_BITS', 1)

        # 3. 更新内部的、高置信度的识别结果
        responding_ids = {t.id for t in responding_tags}
        self.found_present_ids.update(responding_ids)
        
        all_tags_in_leaf_ids = {t.id for t in leaf_node.tags}
        missing_in_this_leaf = all_tags_in_leaf_ids - responding_ids
        self.found_missing_ids.update(missing_in_this_leaf)
        
        op_desc = f"CPT 查询: 路径深度={len(path)}, 叶子大小={len(leaf_node.tags)}, 响应数={len(responding_tags)}"
        
        return AlgorithmStepResult(
            operation_type='CPT_QUERY',
            reader_bits=reader_bits,
            tag_bits=tag_bits,
            operation_description=op_desc
        )

    # --- CPT 算法内部辅助方法 (离线计算，无通信开销) ---
    def _generate_pseudo_ids(self):
        n = len(self.expected_tags_db)
        if n == 0: return
        self.pseudo_id_len = math.ceil(2 * math.log2(n)) if n > 1 else 1
        while True:
            self.pseudo_id_map.clear()
            pseudo_id_set = set()
            collision_detected = False
            for tag in self.expected_tags_db:
                pseudo_hash = hash(tag.id + str(random.random()))
                pseudo_id = format(pseudo_hash & ((1 << self.pseudo_id_len) - 1), f'0{self.pseudo_id_len}b')
                if pseudo_id in pseudo_id_set:
                    collision_detected = True; break
                pseudo_id_set.add(pseudo_id)
                self.pseudo_id_map[tag.id] = pseudo_id
            if not collision_detected: break
    
    def _build_cpt_recursive(self, tags: List[Tag], current_path: List) -> _CPTNode:
        node = _CPTNode()
        node.path_from_root = current_path
        if len(tags) <= 2:
            node.is_leaf = True; node.tags = tags; return node
        best_split_bit_index, min_difference = -1, float('inf')
        for bit_idx in range(self.pseudo_id_len):
            count_0 = sum(1 for tag in tags if self.pseudo_id_map[tag.id][bit_idx] == '0')
            difference = abs(count_0 - (len(tags) - count_0))
            if difference < min_difference:
                min_difference, best_split_bit_index = difference, bit_idx
        node.split_bit_index = best_split_bit_index
        tags_0 = [t for t in tags if self.pseudo_id_map[t.id][best_split_bit_index] == '0']
        tags_1 = [t for t in tags if self.pseudo_id_map[t.id][best_split_bit_index] == '1']
        node.child_0 = self._build_cpt_recursive(tags_0, current_path + [(best_split_bit_index, '0')])
        node.child_1 = self._build_cpt_recursive(tags_1, current_path + [(best_split_bit_index, '1')])
        return node
    
    def _collect_leaf_queries(self, node: _CPTNode):
        if node is None: return
        if node.is_leaf:
            if node.tags: self.query_queue.append(node)
            return
        self._collect_leaf_queries(node.child_0)
        self._collect_leaf_queries(node.child_1)


# ==============================================================================
# 2. 主程序入口
# ==============================================================================

if __name__ == '__main__':

    # 1. 定义全局仿真配置 (采用动态时间模型)
    global_config = {
        'TOTAL_TAGS': 50000,
        'MISSING_RATE': 0.01,
        'BINARY_LENGTH': 96,
        'DATA_RATE_BPS': 160000.0,
        'BITS_PER_MICROSECOND': 160000.0 / 1.0e6,
        'T1_RTcal': 25.0,
        'T2_TRcal': 25.0,
        'MINIMAL_GUARD_TIME_US': 50.0,
        'READER_CMD_BASE_BITS': 37,
        'TAG_SHORT_RESP_BITS': 1,
    }

    # 2. 运行 CPT 仿真
    print("\n--- 运行 CPT 算法 (动态时间模型, 已修复时间线) ---")
    cpt_results = run_missing_tag_simulation(
        global_config=global_config,
        algorithm_class=CPTAlgo,
        algorithm_specific_config={}
    )
    print_results(cpt_results)

    # 3. (可选) 运行基线算法进行对比
    print("\n--- 运行基线轮询算法作为对比 ---")
    baseline_results = run_missing_tag_simulation(
        global_config=global_config,
        algorithm_class=BaselinePollingAlgo,
        algorithm_specific_config={}
    )
    print_results(baseline_results)
