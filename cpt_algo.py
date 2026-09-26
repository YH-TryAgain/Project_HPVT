# -*- coding: utf-8 -*-
"""
CPT (Collision-Partition Tree) 算法的实现，已适配新框架。

此版本遵循最新的框架设计，将全局物理常量与算法逻辑分离，
并净化了算法的初始化接口。
【新增】加入了对关键结构性指标的追踪，以支持多维度性能分析。
"""
import math
import random
from collections import deque
from typing import List, Set, Tuple, Dict
import os # 导入os模块以获取进程ID
# 导入新框架的基类、数据结构和全局常量
from framework import (
    MissingTagAlgorithmInterface,
    Tag,
    AlgorithmStepResult,
    CONSTANTS
)

# ==============================================================================
# 1. CPT 算法内部辅助类
# ==============================================================================

class _CPTNode:
    """CPT树的内部节点类，用于构建树形结构。"""
    def __init__(self):
        self.is_leaf: bool = False
        self.split_bit_index: int = -1
        self.child_0: '_CPTNode' = None
        self.child_1: '_CPTNode' = None
        self.tags: List[Tag] = []
        self.path_from_root: List[Tuple[int, str]] = []

# ==============================================================================
# 2. CPT 算法实现 (带调试打印)
# ==============================================================================
class CPT_Algo(MissingTagAlgorithmInterface):
    """
    CPT 算法，已加入调试打印功能。
    """
    
    def __init__(self):
        super().__init__()
        self.expected_tags_db: List[Tag] = []
        self.found_present_ids: Set[str] = set()
        self.found_missing_ids: Set[str] = set()
        self.pseudo_id_map: Dict[str, str] = {}
        self.pseudo_id_len: int = 0
        self.query_queue: deque[_CPTNode] = deque()
        self.root: _CPTNode = None
        self.metrics: Dict[str, int] = {}

    def initialize(self, expected_tags: List[Tag]):
        super().initialize(expected_tags)
        
        # 重置所有状态和指标
        # ... (此处省略与之前版本相同的重置代码) ...
        self.metrics = {
            "total_leaves": 0, "tree_depth": 0,
            "singleton_leaves": 0, "doubleton_leaves": 0,
            "total_steps": 0
        }
        
        if not self.expected_tags_db:
            return
            
        # 步骤一：生成伪ID (离线计算)
        self._generate_pseudo_ids()

        # 步骤二：构建CPT树 (离线计算)
       # print(f"[PID: {os.getpid()}] 开始构建CPT树...")
        self.root = self._build_cpt_recursive(self.expected_tags_db, [], 1)
       # print(f"[PID: {os.getpid()}] CPT树构建完成。")
        
        # 步骤三：分析树结构
        self._analyze_tree(self.root, 1)
        self.metrics['total_steps'] = self.metrics['total_leaves']
        
        # 步骤四：准备查询队列
        self._collect_leaf_queries(self.root)

    # --- is_finished, get_results, perform_step 保持不变 ---
    def is_finished(self) -> bool: return not self.query_queue
    def get_results(self) -> Tuple[Set[str], Set[str]]:
        all_db_ids = {t.id for t in self.expected_tags_db}
        final_missing_ids = all_db_ids - self.found_present_ids
        return self.found_present_ids, final_missing_ids
    def perform_step(self) -> AlgorithmStepResult:
        if self.is_finished(): return AlgorithmStepResult(operation_type="finished")
        leaf_node = self.query_queue.popleft()
        path = leaf_node.path_from_root
        bits_for_path_len = math.ceil(math.log2(self.pseudo_id_len)) if self.pseudo_id_len > 1 else 1
        reader_bits = len(path) * (bits_for_path_len + 1)
        if len(leaf_node.tags) == 2:
            tag0_pid, tag1_pid = self.pseudo_id_map[leaf_node.tags[0].id], self.pseudo_id_map[leaf_node.tags[1].id]
            for i in range(self.pseudo_id_len):
                if tag0_pid[i] != tag1_pid[i]:
                    reader_bits += bits_for_path_len; break
        responding_tags = [tag for tag in leaf_node.tags if tag.is_present]
        tag_bits = len(responding_tags) * CONSTANTS.TAG_SHORT_RESP_BITS
        responding_ids = {t.id for t in responding_tags}
        self.found_present_ids.update(responding_ids)
        all_tags_in_leaf_ids = {t.id for t in leaf_node.tags}
        self.found_missing_ids.update(all_tags_in_leaf_ids - responding_ids)
        op_desc = f"CPT Query: path_depth={len(path)}, leaf_size={len(leaf_node.tags)}, responses={len(responding_tags)}"
        return AlgorithmStepResult(operation_type='CPT_QUERY', reader_bits=reader_bits, tag_bits=tag_bits, operation_description=op_desc)

    # --- CPT 算法内部辅助方法 ---
    def _generate_pseudo_ids(self):
        n = len(self.expected_tags_db)
        if n == 0: return
        self.pseudo_id_len = math.ceil(2 * math.log2(n)) if n > 1 else 1
        
        attempt_count = 0
        while True:
            attempt_count += 1
            # 【调试打印】
         #   print(f"[PID: {os.getpid()}] 正在生成伪ID (长度={self.pseudo_id_len}), 尝试次数: {attempt_count}...")
            
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
                
            if not collision_detected:
             #   print(f"[PID: {os.getpid()}] 伪ID生成成功！")
                break
            #else:
                # 【调试打印】
              #  print(f"[PID: {os.getpid()}] 检测到哈希碰撞，正在重试...")
    
    def _build_cpt_recursive(self, tags: List[Tag], current_path: List, depth: int) -> _CPTNode:
        # 【调试打印】每1000个节点打印一次，避免刷屏
        #if len(tags) % 1000 == 0 and len(tags) > 0:
          # print(f"[PID: {os.getpid()}] ...构建CPT节点，当前节点标签数: {len(tags)}, 深度: {depth}")
            
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
                if min_difference == 0: break # 找到了完美分割，提前退出
                
        node.split_bit_index = best_split_bit_index
        tags_0 = [t for t in tags if self.pseudo_id_map[t.id][best_split_bit_index] == '0']
        tags_1 = [t for t in tags if self.pseudo_id_map[t.id][best_split_bit_index] == '1']
        
        node.child_0 = self._build_cpt_recursive(tags_0, current_path + [(best_split_bit_index, '0')], depth + 1)
        node.child_1 = self._build_cpt_recursive(tags_1, current_path + [(best_split_bit_index, '1')], depth + 1)
        return node
    
    def _collect_leaf_queries(self, node: _CPTNode):
        if node is None: return
        if node.is_leaf:
            if node.tags: self.query_queue.append(node)
            return
        self._collect_leaf_queries(node.child_0)
        self._collect_leaf_queries(node.child_1)

    def _analyze_tree(self, node: _CPTNode, depth: int):
        if node is None: return
        if self.metrics['tree_depth'] < depth: self.metrics['tree_depth'] = depth
        if node.is_leaf:
            if node.tags:
                self.metrics["total_leaves"] += 1
                if len(node.tags) == 1: self.metrics["singleton_leaves"] += 1
                elif len(node.tags) == 2: self.metrics["doubleton_leaves"] += 1
            return
        self._analyze_tree(node.child_0, depth + 1)
        self._analyze_tree(node.child_1, depth + 1)

