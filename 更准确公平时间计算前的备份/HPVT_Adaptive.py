import math
import random
from collections import deque, Counter
import time

from Framework import *

# ==============================================================================
# 算法实现区域 (包含所有版本的算法)
# ==============================================================================

class _HPVT_Task:
    """HPVT算法内部使用的任务对象。"""
    def __init__(self, prefix: str, tags_in_scope: list['Tag']):
        self.prefix = prefix
        self.tags_in_scope = tags_in_scope

class _AlohaFrameContext:
    """用于存储当前正在执行的ALOHA帧的上下文信息。"""
    def __init__(self, task: '_HPVT_Task', frame_size: int, random_seed: int):
        self.original_task = task
        self.frame_size = frame_size
        self.random_seed = random_seed
        self.current_slot = 0
        self.expected_slots = [[] for _ in range(frame_size)]
        self.actual_responses = [[] for _ in range(frame_size)]
        
        # !!核心Bug修复点1!!: 增加本帧内部的解析结果缓存
        self.resolved_present_ids_in_frame = set()
        self.resolved_missing_ids_in_frame = set()
        
        for tag in task.tags_in_scope:
            slot_index = hash(tag.id + str(random_seed)) % frame_size
            self.expected_slots[slot_index].append(tag)
            if tag.is_present:
                self.actual_responses[slot_index].append(tag)

class HPVTAlgo_SOTA_Inspired(MissingTagAlgorithmInterface):
    """
    HPVT - SOTA Inspired 版本 (Bug修复)
    """
    POLLING_THRESHOLD = 2
    MICRO_ALOHA_THRESHOLD = 5

    def initialize(self, expected_tags: list['Tag']):
        self.expected_tags_db = expected_tags
        self.found_present_ids = set()
        self.found_missing_ids = set()
        self.task_queue = deque()
        
        self.STATE_IDLE, self.STATE_AWAITING_VERIFICATION, self.STATE_IN_ALOHA_FRAME, self.STATE_POLLING_COLLISIONS = 0, 1, 2, 3
        self.current_state = self.STATE_IDLE
        
        self.pending_task = None
        self.active_aloha_frame = None
        self.polling_queue = deque()

        if not self.expected_tags_db: return

        tags_0 = [t for t in self.expected_tags_db if t.id.startswith('0')]
        tags_1 = [t for t in self.expected_tags_db if t.id.startswith('1')]
        
        if tags_0: self.task_queue.append(_HPVT_Task(prefix='0', tags_in_scope=tags_0))
        if tags_1: self.task_queue.append(_HPVT_Task(prefix='1', tags_in_scope=tags_1))

    def is_finished(self) -> bool:
        return not self.task_queue and not self.polling_queue and self.current_state == self.STATE_IDLE

    def get_results(self) -> tuple[set[str], set[str]]:
        # !!核心Bug修复点2!!: get_results 不应修改状态，只做报告
        # 最后的检查应该在仿真器中进行，而不是算法内部
        # 确保算法逻辑本身能识别所有标签
        return self.found_present_ids, self.found_missing_ids

    def perform_step(self) -> 'AlgorithmStepResult':
        if self.is_finished():
            return AlgorithmStepResult(0, 0, 0, "finished")

        state_map = {
            self.STATE_POLLING_COLLISIONS: self._perform_collision_polling_step,
            self.STATE_IN_ALOHA_FRAME: self._perform_aloha_slot,
            self.STATE_AWAITING_VERIFICATION: self._setup_parallel_verification,
        }
        if self.current_state in state_map:
            return state_map[self.current_state]()

        if self.polling_queue:
            self.current_state = self.STATE_POLLING_COLLISIONS
            return self._perform_collision_polling_step()

        if self.task_queue:
            return self._perform_group_probe()

        # 如果队列都为空，但状态不是IDLE，强制设为IDLE，避免卡死
        if self.current_state != self.STATE_IDLE:
             self.current_state = self.STATE_IDLE

        return AlgorithmStepResult(0, 0, 0, "waiting")

    def _perform_group_probe(self) -> 'AlgorithmStepResult':
        # ... 此方法逻辑无变化 ...
        current_task = self.task_queue.popleft()
        
        reader_bits_probe = self.config['READER_CMD_BASE_BITS'] + len(current_task.prefix)
        time_delta_probe = (reader_bits_probe / self.config['BITS_PER_MICROSECOND']) + self.config.get('T_l_SLOT_TIME_US', 800.0)

        responding_tags = [t for t in current_task.tags_in_scope if t.is_present]

        if not responding_tags:
            tags_in_scope_ids = {t.id for t in current_task.tags_in_scope}
            self.found_missing_ids.update(tags_in_scope_ids)
            op_desc = f"SOTA Probe '{current_task.prefix}': Idle. Pruned."
            return AlgorithmStepResult(time_delta_probe, reader_bits_probe, 0, op_desc)

        if len(current_task.tags_in_scope) == 1:
            singleton_tag = current_task.tags_in_scope[0]
            self.found_present_ids.add(singleton_tag.id)
            op_desc = f"SOTA Probe '{current_task.prefix}': Singleton Confirmed."
            return AlgorithmStepResult(time_delta_probe, reader_bits_probe, 1, op_desc)

        self.current_state = self.STATE_AWAITING_VERIFICATION
        self.pending_task = current_task
        self.pending_task.actual_responders_count = len(responding_tags)
        op_desc = f"SOTA Probe '{current_task.prefix}': Active ({len(responding_tags)} responders)."
        return AlgorithmStepResult(time_delta_probe, reader_bits_probe, 1, op_desc)

    def _setup_parallel_verification(self) -> 'AlgorithmStepResult':
        # ... 此方法逻辑无变化，但注意它创建了带有缓存的 _AlohaFrameContext ...
        current_task = self.pending_task
        self.pending_task = None
        n_actual = getattr(current_task, 'actual_responders_count', len(current_task.tags_in_scope))
        frame_size = n_actual if n_actual > 0 else 1
        random_seed = random.randint(0, 10000)
        self.active_aloha_frame = _AlohaFrameContext(current_task, frame_size, random_seed)
        self.current_state = self.STATE_IN_ALOHA_FRAME
        reader_bits_setup = self.config['READER_CMD_BASE_BITS'] + 32
        time_delta_setup = reader_bits_setup / self.config['BITS_PER_MICROSECOND']
        op_desc = f"SOTA ALOHA Setup for '{current_task.prefix}': f={frame_size}."
        return AlgorithmStepResult(time_delta_setup, reader_bits_setup, 0, op_desc)

    def _perform_aloha_slot(self) -> 'AlgorithmStepResult':
        frame_ctx = self.active_aloha_frame
        slot_idx = frame_ctx.current_slot
        
        expected_in_slot = frame_ctx.expected_slots[slot_idx]
        responded_in_slot = frame_ctx.actual_responses[slot_idx]
        
        # !!核心Bug修复点3!!: 结果写入本帧的缓存，而不是全局集合
        if len(responded_in_slot) == 1:
            frame_ctx.resolved_present_ids_in_frame.add(responded_in_slot[0].id)
        elif len(expected_in_slot) == 1 and not responded_in_slot:
            frame_ctx.resolved_missing_ids_in_frame.add(expected_in_slot[0].id)
        
        time_delta_slot = self.config.get('T_s_SLOT_TIME_US', 400.0)
        tag_bits_slot = len(responded_in_slot) * self.config.get('TAG_SHORT_RESP_BITS', 1)

        frame_ctx.current_slot += 1
        if frame_ctx.current_slot >= frame_ctx.frame_size:
            self._finalize_aloha_frame()

        return AlgorithmStepResult(time_delta_slot, 0, tag_bits_slot, f"ALOHA slot {slot_idx+1}/{frame_ctx.frame_size}")

    def _finalize_aloha_frame(self):
        frame_ctx = self.active_aloha_frame

        # 1. 首先，将本帧内确定的结果提交到全局集合
        self.found_present_ids.update(frame_ctx.resolved_present_ids_in_frame)
        self.found_missing_ids.update(frame_ctx.resolved_missing_ids_in_frame)

        # 2. 然后，找出本帧的碰撞标签
        collided_tags = {tag for slot in frame_ctx.actual_responses if len(slot) > 1 for tag in slot}
        collided_ids = {t.id for t in collided_tags}

        # 3. !!核心Bug修复点4!!: 现在基于正确的、局限于本帧的作用域来计算隐式丢失
        all_ids_in_frame_task = {t.id for t in frame_ctx.original_task.tags_in_scope}
        resolved_ids_this_frame = frame_ctx.resolved_present_ids_in_frame.union(frame_ctx.resolved_missing_ids_in_frame)
        
        implicitly_missing_ids = all_ids_in_frame_task - resolved_ids_this_frame - collided_ids
        self.found_missing_ids.update(implicitly_missing_ids)

        # 4. 最后，基于正确的碰撞集合进行分级处理 (这部分逻辑不变)
        if collided_tags:
            n_collided = len(collided_tags)
            if n_collided <= self.POLLING_THRESHOLD:
                for tag in collided_tags: self.polling_queue.append(tag)
                self.current_state = self.STATE_POLLING_COLLISIONS
            elif n_collided <= self.MICRO_ALOHA_THRESHOLD:
                new_task = _HPVT_Task(prefix=f"micro_aloha_{n_collided}", tags_in_scope=list(collided_tags))
                self.task_queue.appendleft(new_task)
            else:
                optimal_bit = self._find_optimal_split_bit(collided_tags, self.config['BINARY_LENGTH'])
                tags_0 = [t for t in collided_tags if t.id[optimal_bit] == '0']
                tags_1 = [t for t in collided_tags if t.id[optimal_bit] == '1']
                if tags_0: self.task_queue.appendleft(_HPVT_Task(prefix=f"split_on_bit_{optimal_bit}=0", tags_in_scope=tags_0))
                if tags_1: self.task_queue.appendleft(_HPVT_Task(prefix=f"split_on_bit_{optimal_bit}=1", tags_in_scope=tags_1))
        
        self.active_aloha_frame = None
        if self.current_state != self.STATE_POLLING_COLLISIONS:
            self.current_state = self.STATE_IDLE

    def _find_optimal_split_bit(self, tags: set['Tag'], binary_length: int) -> int:
        # ... 此方法逻辑无变化 ...
        best_bit = 0
        min_diff = len(tags)
        for bit_index in range(binary_length):
            counts = Counter(tag.id[bit_index] for tag in tags)
            diff = abs(counts['0'] - counts['1'])
            if diff < min_diff:
                min_diff = diff
                best_bit = bit_index
                if min_diff == 0: return best_bit
        return best_bit

    def _perform_collision_polling_step(self) -> 'AlgorithmStepResult':
        # ... 此方法逻辑无变化 ...
        if not self.polling_queue:
            self.current_state = self.STATE_IDLE
            return AlgorithmStepResult(0, 0, 0, "polling finished")
        tag_to_poll = self.polling_queue.popleft()
        self.found_present_ids.add(tag_to_poll.id)
        time_delta = self.config['T_tag_SLOT_TIME_US']
        reader_bits = self.config['READER_CMD_BASE_BITS'] + self.config['BINARY_LENGTH']
        tag_bits = self.config['BINARY_LENGTH']
        op_desc = f"Collision Polling for {tag_to_poll.id[:10]}... -> Present"
        if not self.polling_queue:
            self.current_state = self.STATE_IDLE
        return AlgorithmStepResult(time_delta, reader_bits, tag_bits, op_desc)

# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == '__main__':
    print("开始缺失标签识别算法的最终对比仿真...")

    # --- 1. 定义全局仿真配置 ---
    # 使用与之前相同，但更具挑战性的参数
    simulation_parameters = {
        'TOTAL_TAGS': 5000,
        'MISSING_RATE': 0.1,     # 10% 的标签丢失
        'BINARY_LENGTH': 96,
        'DATA_RATE_BPS': 160000.0,
        
        'T_s_SLOT_TIME_US': 400.0,
        'T_l_SLOT_TIME_US': 800.0,
        'T_tag_SLOT_TIME_US': 2400.0,
        
        'READER_CMD_BASE_BITS': 37,
        'TAG_SHORT_RESP_BITS': 1,
    }
    simulation_parameters['BITS_PER_MICROSECOND'] = simulation_parameters['DATA_RATE_BPS'] / 1.0e6

    # --- 2. 运行所有版本的算法进行对比 ---
    
    # 重新引入之前的算法类定义（或从外部文件导入）
    # 为简洁起见，这里假设它们已定义
    # from previous_code import BaselinePollingAlgo, HPVTAlgo, HPVTAlgoOptimized

    # 为了让此文件独立运行，我们在此处包含所有算法定义
    # (在实际项目中，您会把类放在不同的文件中)
    # 定义 BaselinePollingAlgo, HPVTAlgo, HPVTAlgoOptimized... (此处省略以节省空间)
    # 请从之前的交互中复制这些类的代码到此处

    # 运行最终的 SOTA Inspired 算法
    print("\n>>> 正在运行 HPVTAlgo_SOTA_Inspired (最终版本)...")
    sota_results = run_missing_tag_simulation(
        global_config=simulation_parameters,
        algorithm_class=HPVTAlgo_SOTA_Inspired,
        algorithm_specific_config={}
    )

    # --- 3. 打印结果 ---
    print("\n" + "="*60)
    print("                 终 极 仿 真 结 果")
    print("="*60)
    
    # print_results(baseline_results)
    # print_results(hpvt_original_results)
    # print_results(hpvt_optimized_results)
    print_results(sota_results)
    
    print("\n\n--- 性能提升分析 ---")
    # 此处可以添加与其他算法结果的详细对比
    # 例如：
    # if sota_results and hpvt_optimized_results:
    #     optimized_time = hpvt_optimized_results.get('total_protocol_time_us', float('inf'))
    #     sota_time = sota_results.get('total_protocol_time_us', float('inf'))
    #     if optimized_time > 0 and sota_time > 0:
    #         reduction = (1 - sota_time / optimized_time) * 100
    #         print(f"SOTA 版本相比优化版，时间效率进一步提升了: {reduction:.2f}%")
        
    print("\n仿真执行结束。")

