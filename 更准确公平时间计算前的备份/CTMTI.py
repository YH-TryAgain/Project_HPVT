# -*- coding: utf-8 -*-

import random
import time
import numpy as np  # 导入numpy用于科学计算

# ==============================================================================
# 0. 仿真配置和常量
# ==============================================================================

# --- 场景配置 ---
DEFAULT_TOTAL_TAGS = 50000      # 数据库中的标签总数 (N)
DEFAULT_MISSING_RATE = 0.01     # 标签丢失率 (α)，例如 1%
DEFAULT_BINARY_LENGTH = 96      # 标签ID长度 (bits)

# --- 物理层和链路层时间常量 (µs) ---
DEFAULT_DATA_RATE_BPS = 160000.0
DEFAULT_BITS_PER_MICROSECOND = DEFAULT_DATA_RATE_BPS / 1.0e6
DEFAULT_T1_RTcal = 25.0         # 读写器 -> 标签 周转时间
DEFAULT_T2_TRcal = 25.0         # 标签 -> 读写器 周转时间
DEFAULT_READER_CMD_BASE_BITS = 37 # 读写器命令的基础比特数
DEFAULT_TAG_SHORT_RESP_BITS = 1   # 标签短响应的比特数 (例如，1比特表示'我在')

from Framework import *
# ==============================================================================
# 6. CTMTI 算法实现 (新增)
# ==============================================================================
class CTMTIAlgo(MissingTagAlgorithmInterface):
    """
    复现论文 "A Collision Reconciling-Based B-Ary Tree-Splitting
    Missing Tag Identification Protocol" (CTMTI) 的算法。
    """
    def initialize(self, expected_tags: list[Tag]):
        """初始化算法状态。"""
        self.expected_tags_db = expected_tags
        # 初始时，所有标签都未被识别
        self.unidentified_tags = list(expected_tags)
        self.predicted_present_ids = set()
        self.predicted_missing_ids = set()
        # is_first_slot 用于区分 S1 和 Si (i>1)
        self.is_first_slot = True
        # irresolvable_groups 存储上一轮未解决的碰撞组
        self.irresolvable_groups = []

    def is_finished(self) -> bool:
        """
        当不再是第一轮且没有未解决的碰撞组时，算法结束。
        这意味着所有标签都已被识别。
        """
        return not self.is_first_slot and not self.irresolvable_groups

    def get_results(self) -> tuple[set[str], set[str]]:
        """
        返回当前已识别的在场和丢失标签集合。
        注意：最终的统计在 run_missing_tag_simulation 中完成，
        它会将所有未被识别为在场的标签都归为丢失。
        """
        return self.predicted_present_ids, self.predicted_missing_ids

    def perform_step(self) -> AlgorithmStepResult:
        """执行一个完整的 CTMTI 时隙（slot）。"""
        if self.is_finished():
            return AlgorithmStepResult(operation_description="finished")

        # --- 1. 设置本轮待处理的标签和哈希空间 ---
        R1 = random.random()
        groups = {} # key: element_index, value: list of tags
        num_elements = 0

        if self.is_first_slot:
            # S1: 处理所有未识别标签
            tags_to_process = self.unidentified_tags
            num_elements = int(self.config['TOTAL_TAGS'] * self.config.get('alpha', 0.54))
            if num_elements == 0: num_elements = 1
            groups = {i: [] for i in range(num_elements)}
            for tag in tags_to_process:
                h = hash((tag.id, R1, "Vp_hash")) % num_elements
                groups[h].append(tag)
        else:
            # Si (i>1): B-ary 树分裂
            B = self.config.get('B', 2)
            num_elements = len(self.irresolvable_groups) * B
            if num_elements == 0:
                return AlgorithmStepResult(operation_description="finished")
            groups = {i: [] for i in range(num_elements)}
            group_idx_offset = 0
            for tag_group in self.irresolvable_groups:
                for tag in tag_group:
                    h = hash((tag.id, R1, "B_ary_split")) % B
                    groups[group_idx_offset + h].append(tag)
                group_idx_offset += B

        # --- 2. 碰撞调节 (Va) 并构建指示向量 (Ve) ---
        R2 = random.random()
        new_irresolvable_groups = []
        
        # 预先分类，以确定响应字符串的长度
        element_indices_01 = [] # 单标签
        element_indices_10 = [] # 可解碰撞
        
        for i in range(num_elements):
            tags_in_elem = groups.get(i, [])
            if len(tags_in_elem) == 1:
                element_indices_01.append(i)
            elif len(tags_in_elem) > 1:
                sub_groups = {0: [], 1: [], 2: []}
                for tag in tags_in_elem:
                    h_sub = hash((tag.id, R2, "Va_hash")) % 3
                    sub_groups[h_sub].append(tag)
                
                is_resolvable = all(len(v) <= 1 for v in sub_groups.values())
                if is_resolvable:
                    element_indices_10.append(i)
                else:
                    new_irresolvable_groups.append(tags_in_elem)

        # --- 3. 确定响应位置并构建预期响应 ---
        # {response_bit_position: Tag}
        expected_responses = {}
        
        # 处理单标签
        for i, elem_idx in enumerate(element_indices_01):
            tag = groups[elem_idx][0]
            response_pos = i
            expected_responses[response_pos] = tag
        
        X_01_total = len(element_indices_01)
        # 处理可解碰撞
        for i, elem_idx in enumerate(element_indices_10):
            tags_in_elem = groups[elem_idx]
            for tag in tags_in_elem:
                h_sub = hash((tag.id, R2, "Va_hash")) % 3
                response_pos = X_01_total + 3 * i + h_sub
                expected_responses[response_pos] = tag

        # --- 4. 仿真标签响应并识别 ---
        response_len = X_01_total + 3 * len(element_indices_10)
        received_string = [0] * response_len

        # 仿真在场标签进行响应
        for pos, tag in expected_responses.items():
            if tag.is_present:
                received_string[pos] = 1

        # 读写器根据收到的响应进行判断
        for pos, tag in expected_responses.items():
            if received_string[pos] == 1:
                self.predicted_present_ids.add(tag.id)
            else:
                self.predicted_missing_ids.add(tag.id)

        # --- 5. 计算本轮开销 (已修正) ---
        # 首先计算有效比特数
        reader_useful_bits = (2.0 * num_elements) + 4.0 + 16.0 * 4.0
        tag_useful_bits = float(response_len)

        # 核心修正：根据论文的离散时间模型计算耗时
        # 论文的时间模型 T = ceil(bits / packet_size) * time_per_packet
        # 这模拟了数据按固定大小的包进行传输的开销
        PACKET_SIZE = 96.0  # bits
        # t_id 是传输一个 packet 所需的时间 (单位: µs)
        time_per_packet_us = PACKET_SIZE / self.config['BITS_PER_MICROSECOND']

        # 计算读写器和标签传输需要的包数量 (向上取整)
        num_reader_packets = np.ceil(reader_useful_bits / PACKET_SIZE) if reader_useful_bits > 0 else 0
        num_tag_packets = np.ceil(tag_useful_bits / PACKET_SIZE) if tag_useful_bits > 0 else 0

        # 计算实际传输时间
        time_reader_tx = num_reader_packets * time_per_packet_us
        time_tag_tx = num_tag_packets * time_per_packet_us

        # 计算本轮时隙的总时间
        time_delta = time_reader_tx + self.config['T1_RTcal'] + time_tag_tx + self.config['T2_TRcal']

        # --- 6. 更新状态进入下一轮 ---
        self.is_first_slot = False
        self.irresolvable_groups = new_irresolvable_groups
        self.unidentified_tags = [tag for group in self.irresolvable_groups for tag in group]

        return AlgorithmStepResult(
            time_delta_us=time_delta,
            reader_bits=reader_useful_bits,  # 仍然返回有效比特数，以保持指标一致性
            tag_bits=tag_useful_bits,
            operation_description=f"Slot with {num_elements} elements, {len(self.unidentified_tags)} tags remain"
        )


# ==============================================================================
# 7. 主程序入口 (新增)
# ==============================================================================

if __name__ == '__main__':
    # --- 1. 定义全局仿真参数 ---
    # 您可以修改这些值来进行不同的实验
    global_simulation_config = {
        'TOTAL_TAGS': 20000,
        'MISSING_RATE': 0.1, # 10%
        'BINARY_LENGTH': 96,
        'DATA_RATE_BPS': DEFAULT_DATA_RATE_BPS,
        'BITS_PER_MICROSECOND': DEFAULT_BITS_PER_MICROSECOND,
        'T1_RTcal': DEFAULT_T1_RTcal,
        'T2_TRcal': DEFAULT_T2_TRcal,
    }

    # --- 2. 定义 CTMTI 算法的特定参数 ---
    # 根据论文，最优参数为 alpha = 0.54, B = 2
    ctmti_specific_config = {
        'alpha': 0.54,
        'B': 2,
    }

    # --- 3. 运行 CTMTI 仿真 ---
    ctmti_results = run_missing_tag_simulation(
        global_config=global_simulation_config,
        algorithm_class=CTMTIAlgo,
        algorithm_specific_config=ctmti_specific_config
    )
    print_results(ctmti_results)

    # --- 4. (可选) 运行基线算法进行对比 ---
    print("\n\n" + "="*40)
    print("正在运行基线轮询算法作为对比...")
    print("="*40)
    
    baseline_specific_config = {} # 基线算法没有特定参数
    
    baseline_results = run_missing_tag_simulation(
        global_config=global_simulation_config,
        algorithm_class=BaselinePollingAlgo,
        algorithm_specific_config=baseline_specific_config
    )
    print_results(baseline_results)
