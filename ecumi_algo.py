# -*- coding: utf-8 -*-
"""
适配仿真框架的 CUMI 和 ECUMI 算法实现。

该文件包含了 "Efficient collision-slot utilization for missing tags 
identification in RFID system" 论文中 CUMI 和 ECUMI 算法的 Python 实现。
代码已完全适配 V5 版本的仿真框架接口，可直接用于仿真和性能评估。

---
关键可配置参数:
- CUMIAlgo:
  - x (int): CUMI 协议中，一个时隙被视为“可用”的最大标签数。论文推荐值为 3。
- ECUMIAlgo:
  - x_prime (int): ECUMI 协议中，可用时隙的最大标签数，同时也作为标签响应的比特长度。论文推荐值为 10。
---
"""
import math
import random
from typing import List, Set, Tuple

from framework import *

# ==============================================================================
# 1. CUMI 算法实现 (适配新框架)
# ==============================================================================
class CUMIAlgo(MissingTagAlgorithmInterface):
    """
    实现了论文中的 CUMI 协议，已适配仿真框架。
    
    CUMI 通过一个精巧的指示器向量和内部哈希机制，将部分碰撞时隙
    转化为可识别的“伪单例时隙”，从而提高识别效率。
    """
    def __init__(self, x: int = 3):
        """
        CUMI 算法构造函数。
        
        Args:
            x (int): CUMI 协议中，一个时隙被视为“可用”的最大标签数。
                     论文通过仿真证明 x=3 是最优选择。
        """
        super().__init__()
        self.x = x
        # 状态变量的初始化将由框架调用 initialize() 方法完成
        self.active_tags: List[Tag] = []
        self.identified_present_ids: Set[str] = set()
        self.identified_missing_ids: Set[str] = set() # 新增：用于精确追踪已识别的丢失标签
        
        # 算法内部状态
        self._frame_in_progress: bool = False
        self._current_slot: int = 0
        self._frame_size: int = 0
        self._slots_in_frame: List[List[Tag]] = []
        self._tags_identified_in_frame: Set[Tag] = set()
        self._seed: float = 0.0

    def initialize(self, expected_tags: List[Tag]):
        """
        初始化或重置算法的所有状态变量。
        这个方法由仿真器在每次运行开始时调用。
        """
        super().initialize(expected_tags)
        self.active_tags = list(self.expected_tags_db)
        self.identified_present_ids.clear()
        self.identified_missing_ids.clear() # 新增：重置丢失标签集
        
        self._frame_in_progress = False
        self._current_slot = 0
        self._frame_size = 0
        self._slots_in_frame.clear()
        self._tags_identified_in_frame.clear()
        self._seed = 0.0

    def is_finished(self) -> bool:
        """
        判断算法是否完成。当所有预期待识别的标签都被处理完毕时，算法结束。
        """
        return not self.active_tags and not self._frame_in_progress

    def get_results(self) -> Tuple[Set[str], Set[str]]:
        """
        【已修正】返回当前已明确识别的在场和丢失标签集合。
        这个实现更精确，能够支持框架进行准确的过程质量追踪。
        """
        return self.identified_present_ids, self.identified_missing_ids

    def perform_step(self) -> AlgorithmStepResult:
        """
        执行算法的一个逻辑步骤。
        可能是开始一个新帧，或处理当前帧中的一个时隙。
        """
        if not self._frame_in_progress:
            return self._start_new_frame()

        if self._current_slot >= self._frame_size:
            # 当前帧的所有时隙已处理完毕，进行帧的收尾工作
            self.active_tags = [t for t in self.active_tags if t not in self._tags_identified_in_frame]
            self._frame_in_progress = False
            # 返回一个空闲步骤，让主循环有机会检查 is_finished() 或开始新帧
            return AlgorithmStepResult(operation_type="FRAME_END", operation_description="CUMI Frame End")
        
        # 处理当前帧的一个时隙
        return self._process_slot()

    def _start_new_frame(self) -> AlgorithmStepResult:
        """
        准备并开始一个新帧，计算并返回广播指示器向量的通信开销。
        """
        if not self.active_tags:
            # 如果没有活跃标签了，则算法结束
            return AlgorithmStepResult(operation_type="FINISHED")

        self._frame_in_progress = True
        self._current_slot = 0
        self._tags_identified_in_frame.clear()
        self._seed = random.random()
        
        num_active_tags = len(self.active_tags)
        
        # 【死循环修复】
        # 当活跃标签数 N_i 很小时，理论公式可能导致 f=1，如果此时 N_i > x，
        # 所有标签会被强制放入一个不可用的时隙，导致无限循环。
        f_from_formula = max(1, int(0.4263 * num_active_tags))
        if f_from_formula == 1 and num_active_tags > self.x:
            # 覆盖帧大小为更安全的值，以打破循环
            self._frame_size = num_active_tags
        else:
            self._frame_size = f_from_formula

        # 如果活跃标签数为0，帧大小也应为0，以快速结束
        if num_active_tags == 0:
            self._frame_size = 0
            
        # 模拟标签根据哈希函数分配到各个时隙
        self._slots_in_frame = [[] for _ in range(self._frame_size)]
        for tag in self.active_tags:
            slot_index = hash(tag.id + str(self._seed)) % self._frame_size
            self._slots_in_frame[slot_index].append(tag)
        
        # 计算读写器广播指示器向量的比特数
        # 论文公式 (8): f_i * ceil(log2(x+1))
        indicator_bits = self._frame_size * math.ceil(math.log2(self.x + 1))
        reader_bits = CONSTANTS.READER_CMD_BASE_BITS + indicator_bits

        return AlgorithmStepResult(
            operation_type='CUMI_SETUP',
            reader_bits=reader_bits,
            tag_bits=0,
            operation_description=f"CUMI Frame Start: N={num_active_tags}, f={self._frame_size}"
        )

    def _process_slot(self) -> AlgorithmStepResult:
        """
        处理当前帧的一个时隙，判断其是否可用，并进行识别。
        """
        tags_in_slot = self._slots_in_frame[self._current_slot]
        self._current_slot += 1
        j = len(tags_in_slot)

        # 每个时隙的查询都有一个基础命令开销
        reader_bits = CONSTANTS.READER_CMD_BASE_BITS
        
        # 步骤 1: 判断时隙是否“可用” (usable)
        if not (1 <= j <= self.x):
            return AlgorithmStepResult('CUMI_SKIPPED_SLOT', reader_bits, 0, f"Slot {self._current_slot-1}: Skipped (j={j})")
        
        # 步骤 2: 检查内部碰撞 (internal collision)
        d_values = {hash(t.id + str(self._seed)) % j for t in tags_in_slot}
        if len(d_values) != j:
            # 发生内部碰撞，时隙不可用
            return AlgorithmStepResult('CUMI_COLLISION_SLOT', reader_bits, 0, f"Slot {self._current_slot-1}: Internal Collision (j={j})")
        
        # 步骤 3: 时隙可用，进行识别
        tag_bits = float(j)
        
        actual_signal = 0
        for tag in tags_in_slot:
            if tag.is_present:
                d = hash(tag.id + str(self._seed)) % j
                actual_signal |= (1 << d)

        # 【结果追踪修正】
        # 逐个判断该时隙内的标签状态，并分别记录在场和丢失
        for tag in tags_in_slot:
            d = hash(tag.id + str(self._seed)) % j
            if (actual_signal >> d) & 1:
                self.identified_present_ids.add(tag.id)
            else:
                self.identified_missing_ids.add(tag.id)
            self._tags_identified_in_frame.add(tag)

        return AlgorithmStepResult('CUMI_USABLE_SLOT', reader_bits, tag_bits, f"Slot {self._current_slot-1}: Usable (j={j})")


# ==============================================================================
# 2. ECUMI 算法实现 (适配新框架)
# ==============================================================================
class ECUMIAlgo(CUMIAlgo):
    """
    实现了论文中的 ECUMI 协议。
    ECUMI 是 CUMI 的增强版，它通过缩短指示器向量的长度和固定标签响应长度，
    以牺牲少量标签响应时间为代价，换取了总体识别效率的显著提升。
    它继承自 CUMIAlgo，并重写了关键的逻辑部分。
    """
    def __init__(self, x_prime: int = 10):
        """
        ECUMI 算法构造函数。
        
        Args:
            x_prime (int): ECUMI 协议中可用时隙的最大标签数，也是标签响应的比特长度。
                         论文指出 x_prime 越大效率越高，但增长会放缓，推荐值为 10。
        """
        super().__init__()
        self.x_prime = x_prime
        # ECUMI 使用 x_prime 而不是 x
        if hasattr(self, 'x'):
            del self.x 

    def _start_new_frame(self) -> AlgorithmStepResult:
        """
        ECUMI 的准备新帧方法。主要区别在于帧大小计算和指示器向量长度。
        """
        if not self.active_tags:
            return AlgorithmStepResult(operation_type="FINISHED")

        self._frame_in_progress = True
        self._current_slot = 0
        self._tags_identified_in_frame.clear()
        self._seed = random.random()

        num_active_tags = len(self.active_tags)

        # 【死循环修复】
        # 同样为 ECUMI 修复小规模标签集下的帧大小计算问题
        f_from_formula = max(1, int(0.2341 * num_active_tags))
        if f_from_formula == 1 and num_active_tags > self.x_prime:
            self._frame_size = num_active_tags
        else:
            self._frame_size = f_from_formula
        
        if num_active_tags == 0:
            self._frame_size = 0
        
        self._slots_in_frame = [[] for _ in range(self._frame_size)]
        for tag in self.active_tags:
            # 避免当 frame_size 为 0 时出现 Modulo by zero 错误
            if self._frame_size > 0:
                slot_index = hash(tag.id + str(self._seed)) % self._frame_size
                self._slots_in_frame[slot_index].append(tag)
        
        # ECUMI 的指示器向量长度为 f-bit，比 CUMI 短得多
        indicator_bits = float(self._frame_size)
        reader_bits = CONSTANTS.READER_CMD_BASE_BITS + indicator_bits
        
        return AlgorithmStepResult(
            operation_type='ECUMI_SETUP',
            reader_bits=reader_bits,
            tag_bits=0,
            operation_description=f"ECUMI Frame Start: N={num_active_tags}, f={self._frame_size}"
        )

    def _process_slot(self) -> AlgorithmStepResult:
        """
        ECUMI 的处理时隙方法。
        主要区别在于可用时隙判断、标签响应比特数和内部哈希的模数。
        """
        tags_in_slot = self._slots_in_frame[self._current_slot]
        self._current_slot += 1
        j = len(tags_in_slot)

        reader_bits = CONSTANTS.READER_CMD_BASE_BITS
        
        # 步骤 1: 判断时隙是否“可用”
        if not (1 <= j <= self.x_prime):
            return AlgorithmStepResult('ECUMI_SKIPPED_SLOT', reader_bits, 0, f"Slot {self._current_slot-1}: Skipped (j={j})")
        
        # 步骤 2: 计算通信开销
        num_present_tags = sum(1 for t in tags_in_slot if t.is_present)
        tag_bits = float(self.x_prime) if num_present_tags > 0 else 0.0

        # 步骤 3: 检查内部碰撞
        d_values = {hash(t.id + str(self._seed)) % self.x_prime for t in tags_in_slot}
        if len(d_values) != j:
            # 即使内部碰撞，标签也已经响应了，消耗了信道时间。
            return AlgorithmStepResult('ECUMI_COLLISION_SLOT', reader_bits, tag_bits, f"Slot {self._current_slot-1}: Internal Collision (j={j})")
        
        # 步骤 4: 时隙可用且无内部碰撞，进行识别
        actual_signal = 0
        for tag in tags_in_slot:
            if tag.is_present:
                d = hash(tag.id + str(self._seed)) % self.x_prime
                actual_signal |= (1 << d)
        
        # 【结果追踪修正】
        for tag in tags_in_slot:
            d = hash(tag.id + str(self._seed)) % self.x_prime
            if (actual_signal >> d) & 1:
                self.identified_present_ids.add(tag.id)
            else:
                self.identified_missing_ids.add(tag.id)
            self._tags_identified_in_frame.add(tag)
        
        return AlgorithmStepResult('ECUMI_USABLE_SLOT', reader_bits, tag_bits, f"Slot {self._current_slot-1}: Usable (j={j})")
