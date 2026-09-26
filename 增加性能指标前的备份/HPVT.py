# 核心创新点: HPVT 算法将传统碰撞树的确定性前缀分组与现代并行协议的高效随机验证相结合。
# 它不像IIP那样对所有标签进行全局随机散列，也不像CPT那样需要复杂的离线计算来寻找最优分裂位。HPVT使用简单的前缀进行分层“剪枝”，
# 然后在有必要时，才对规模可控的子集启动高效的并行验证，形成一种宏观确定性剪枝、微观并行化处理的新模式。
# -*- coding: utf-8 -*-

from collections import deque
from typing import List, Set, Tuple

# 假设以下类已在您的新框架文件中定义
# from fair_simulation_framework_v2 import MissingTagAlgorithmInterface, AlgorithmStepResult, Tag
# 我们在此处重新定义，以确保代码块的独立性
# （在您的实际项目中，请移除这些重复的定义）
from Framework import *

# ==============================================================================
# HPVT 算法实现 (已适配新框架)
# ==============================================================================

class _HPVT_Task:
    """HPVT算法内部使用的任务对象。"""
    def __init__(self, prefix: str, tags_in_scope: List['Tag']):
        self.prefix = prefix
        self.tags_in_scope = tags_in_scope

class _AlohaFrameContext:
    """用于存储当前正在执行的ALOHA帧的上下文信息。"""
    def __init__(self, task: _HPVT_Task, frame_size: int, random_seed: int):
        self.original_task = task
        self.frame_size = frame_size
        self.random_seed = random_seed
        self.current_slot = 0
        self.expected_slots = [[] for _ in range(frame_size)]
        self.actual_responses = [[] for _ in range(frame_size)]
        
        for tag in task.tags_in_scope:
            slot_index = hash(tag.id + str(random_seed)) % frame_size
            self.expected_slots[slot_index].append(tag)
            if tag.is_present:
                self.actual_responses[slot_index].append(tag)


class HPVTAlgo(MissingTagAlgorithmInterface):
    """
    分层并行验证树 (HPVT) 算法，已适配新的动态时间计算框架。
    """
    
    def initialize(self, expected_tags: List[Tag]):
        """初始化算法，创建顶层任务并加入队列。"""
        self.expected_tags_db = expected_tags
        self.found_present_ids = set()
        self.found_missing_ids = set()
        self.task_queue = deque()
        
        self.STATE_IDLE = 0
        self.STATE_AWAITING_VERIFICATION = 1
        self.STATE_IN_ALOHA_FRAME = 2
        self.current_state = self.STATE_IDLE
        
        self.pending_task = None
        self.active_aloha_frame = None

        if not self.expected_tags_db:
            return

        tags_0 = [t for t in self.expected_tags_db if t.id.startswith('0')]
        tags_1 = [t for t in self.expected_tags_db if t.id.startswith('1')]
        
        if tags_0: self.task_queue.append(_HPVT_Task(prefix='0', tags_in_scope=tags_0))
        if tags_1: self.task_queue.append(_HPVT_Task(prefix='1', tags_in_scope=tags_1))

    def is_finished(self) -> bool:
        """当主任务队列和所有中间状态都为空时，算法才算结束。"""
        return not self.task_queue and self.current_state == self.STATE_IDLE

    def get_results(self) -> Tuple[Set[str], Set[str]]:
        """直接返回逐步确认的集合。"""
        # 在一个严谨的框架中，最后的清理逻辑应在仿真器端完成
        # 但为保持与您原代码一致，我们暂时保留此处的清理逻辑
        all_ids_in_db = {t.id for t in self.expected_tags_db}
        unconfirmed_ids = all_ids_in_db - self.found_present_ids - self.found_missing_ids
        if unconfirmed_ids:
            self.found_missing_ids.update(unconfirmed_ids)
        return self.found_present_ids, self.found_missing_ids

    def perform_step(self) -> 'AlgorithmStepResult':
        """执行一个微步骤，返回该步骤的通信开销信息。"""
        if self.is_finished():
            return AlgorithmStepResult(operation_type="finished")

        if self.current_state == self.STATE_IN_ALOHA_FRAME:
            return self._perform_aloha_slot()
        
        if self.current_state == self.STATE_AWAITING_VERIFICATION:
            return self._setup_parallel_verification()

        if self.task_queue:
            return self._perform_group_probe()

        return AlgorithmStepResult(operation_type="idle")

    def _perform_group_probe(self) -> 'AlgorithmStepResult':
        """执行“分组探测”微步骤，返回其通信比特信息。"""
        current_task = self.task_queue.popleft()
        
        # 计算读写器发送的比特数（基础命令 + 前缀长度）
        reader_bits = self.config['READER_CMD_BASE_BITS'] + len(current_task.prefix)
        
        responding_tags = [t for t in current_task.tags_in_scope if t.is_present]
        tag_bits = 0
        
        if not responding_tags:
            # 高效剪枝：分支为空，所有标签确定为丢失
            tags_in_scope_ids = {t.id for t in current_task.tags_in_scope}
            self.found_missing_ids.update(tags_in_scope_ids)
            op_desc = f"HPVT Probe on '{current_task.prefix}': Idle. Pruned."
        else:
            # 探测非空：标签会响应一个短信号
            tag_bits = self.config.get('TAG_SHORT_RESP_BITS', 1)
            self.current_state = self.STATE_AWAITING_VERIFICATION
            self.pending_task = current_task
            op_desc = f"HPVT Probe on '{current_task.prefix}': Active."
            
        return AlgorithmStepResult(
            operation_type='PROBE',
            reader_bits=reader_bits,
            tag_bits=tag_bits,
            operation_description=op_desc
        )

    def _setup_parallel_verification(self) -> 'AlgorithmStepResult':
        """执行“并行验证准备”：广播ALOHA帧头。这是一个纯读写器操作。"""
        current_task = self.pending_task
        self.pending_task = None

        n_subset = len(current_task.tags_in_scope)
        frame_size = n_subset if n_subset > 0 else 1
        random_seed = random.randint(0, 10000)

        self.active_aloha_frame = _AlohaFrameContext(current_task, frame_size, random_seed)
        self.current_state = self.STATE_IN_ALOHA_FRAME

        # 计算读写器发送的比特数（基础命令 + 种子和帧长等参数）
        reader_bits = self.config['READER_CMD_BASE_BITS'] + 32  # 假设种子和帧长共32位
        op_desc = f"HPVT ALOHA Setup for '{current_task.prefix}': f={frame_size}."
        
        return AlgorithmStepResult(
            operation_type='ALOHA_SETUP',
            reader_bits=reader_bits,
            tag_bits=0, # 标签在此步骤不响应
            operation_description=op_desc
        )

    def _perform_aloha_slot(self) -> 'AlgorithmStepResult':
        """执行“并行验证单时隙处理”。这主要是标签的响应行为。"""
        frame_ctx = self.active_aloha_frame
        slot_idx = frame_ctx.current_slot
        
        expected_in_slot = frame_ctx.expected_slots[slot_idx]
        responded_in_slot = frame_ctx.actual_responses[slot_idx]
        
        # 确定本时隙的逻辑结果
        if len(responded_in_slot) == 1:
            self.found_present_ids.add(responded_in_slot[0].id)
        elif len(expected_in_slot) == 1 and not responded_in_slot:
            self.found_missing_ids.add(expected_in_slot[0].id)
        
        # 计算标签响应的比特数
        # 空闲(0), 成功(1), 碰撞(>1)
        tag_bits = len(responded_in_slot) * self.config.get('TAG_SHORT_RESP_BITS', 1)

        # 移动到下一个时隙并检查是否结束
        frame_ctx.current_slot += 1
        if frame_ctx.current_slot >= frame_ctx.frame_size:
            self._finalize_aloha_frame()

        return AlgorithmStepResult(
            operation_type='ALOHA_SLOT',
            reader_bits=0, # 单个时隙的读者开销已在SETUP中计算
            tag_bits=tag_bits,
            operation_description=f"ALOHA slot {slot_idx+1}/{frame_ctx.frame_size}"
        )

    def _finalize_aloha_frame(self):
        """在ALOHA帧结束后，处理碰撞，创建新任务。此方法不直接产生时间开销。"""
        frame_ctx = self.active_aloha_frame
        
        collided_tags = set()
        for i in range(frame_ctx.frame_size):
            if len(frame_ctx.actual_responses[i]) > 1:
                for tag in frame_ctx.expected_slots[i]:
                    collided_tags.add(tag)

        resolved_ids = set()
        for i in range(frame_ctx.frame_size):
             if len(frame_ctx.actual_responses[i]) == 1:
                 resolved_ids.add(frame_ctx.actual_responses[i][0].id)
             elif len(frame_ctx.expected_slots[i]) == 1 and not frame_ctx.actual_responses[i]:
                 resolved_ids.add(frame_ctx.expected_slots[i][0].id)

        all_ids_in_frame_task = {t.id for t in frame_ctx.original_task.tags_in_scope}
        collided_ids = {t.id for t in collided_tags}
        
        implicitly_missing_ids = all_ids_in_frame_task - resolved_ids - collided_ids
        self.found_missing_ids.update(implicitly_missing_ids)

        if collided_tags:
            prefix = frame_ctx.original_task.prefix
            # 确保前缀不会超过ID总长度
            if len(prefix) < self.config.get('BINARY_LENGTH', 96):
                prefix_0 = prefix + '0'
                prefix_1 = prefix + '1'
                tags_0 = [t for t in collided_tags if t.id.startswith(prefix_0)]
                tags_1 = [t for t in collided_tags if t.id.startswith(prefix_1)]

                if tags_0: self.task_queue.append(_HPVT_Task(prefix=prefix_0, tags_in_scope=tags_0))
                if tags_1: self.task_queue.append(_HPVT_Task(prefix=prefix_1, tags_in_scope=tags_1))
        
        self.active_aloha_frame = None
        self.current_state = self.STATE_IDLE


# ==============================================================================
# 5. 主程序入口
# ==============================================================================

if __name__ == '__main__':
    # 1. 定义全局配置 (采用动态时间模型)
    global_config = {
        'TOTAL_TAGS': 2000,
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

    # 2. 运行仿真
    print("--- 运行基线轮询算法 ---")
    baseline_results = run_missing_tag_simulation(
        global_config=global_config,
        algorithm_class=BaselinePollingAlgo,
        algorithm_specific_config={}
    )
    
    print("\n--- 运行HPVT算法 (动态时间模型) ---")
    hpvt_results = run_missing_tag_simulation(
        global_config=global_config,
        algorithm_class=HPVTAlgo,
        algorithm_specific_config={}
    )

    # 3. 打印结果
    print("\n" + "="*50)
    print("                      仿 真 结 果 对 比")
    print("="*50)
    print_results(baseline_results)
    print_results(hpvt_results)
    print("="*50)

    # 4. 性能对比总结
    if baseline_results and hpvt_results:
        baseline_time = baseline_results.get('total_protocol_time_us', float('inf'))
        hpvt_time = hpvt_results.get('total_protocol_time_us', float('inf'))

        if baseline_time > 1 and hpvt_time > 1:
            reduction_vs_base = (1 - hpvt_time / baseline_time) * 100
            print(f"\n时间效率: HPVT 相比基线轮询，执行时间减少了 {reduction_vs_base:.2f}%")
        else:
            print("\n无法进行有意义的时间效率对比。")