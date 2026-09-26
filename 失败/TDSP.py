import math
import random
from collections import deque
import time

from Framework import *

# --- 算法内部数据结构 ---
class _Task:
    def __init__(self, tags: list['Tag'], prefix: str = ""):
        self.tags_in_scope = tags
        self.prefix = prefix # 仅用于Pruning引擎

class _AlohaContext:
    def __init__(self, tags: list['Tag'], frame_size: int, random_seed: int, task=None):
        self.tags_in_scope = tags
        self.task = task
        self.frame_size = frame_size
        self.random_seed = random_seed
        self.current_slot = 0
        self.expected_slots = [[] for _ in range(frame_size)]
        self.actual_responses = [[] for _ in range(frame_size)]
        self.empty_slot_count = 0
        for tag in tags:
            slot_index = hash(tag.id + str(random_seed)) % frame_size
            self.expected_slots[slot_index].append(tag)
            if tag.is_present: self.actual_responses[slot_index].append(tag)

class TSDPAlgo(MissingTagAlgorithmInterface):
    """
    "诊断-执行"双阶段协议 (Two-Stage Diagnostic Protocol, TSDP)
    """
    def initialize(self, expected_tags: list['Tag']):
        self.expected_tags_db = expected_tags
        self.found_present_ids, self.found_missing_ids = set(), set()
        
        # --- 状态机定义 ---
        self.STATE_DIAGNOSIS_SETUP, self.STATE_DIAGNOSIS_FRAME = 0, 1
        self.STATE_CPT_ENGINE, self.STATE_PRUNING_ENGINE = 10, 20
        self.STATE_PRUNING_AWAIT_ALOHA, self.STATE_PRUNING_IN_ALOHA = 21, 22
        self.STATE_FINISHED = 99
        self.current_state = self.STATE_DIAGNOSIS_SETUP if self.expected_tags_db else self.STATE_FINISHED

        # --- 引擎所需变量 ---
        self.task_queue = deque()
        self.active_aloha_frame = None
        self.pending_task = None
        
        # --- 可调参数 ---
        self.DIAGNOSIS_FRAME_SIZE = 512
        # 当诊断帧的空闲率超过40%时（近似对应丢失率>50%），我们认为丢失率较高
        self.EMPTY_RATE_THRESHOLD = self.config.get('EMPTY_RATE_THRESHOLD', 0.4) 

    def is_finished(self) -> bool: return self.current_state == self.STATE_FINISHED
    def get_results(self) -> tuple[set[str], set[str]]: return self.found_present_ids, self.found_missing_ids

    def perform_step(self) -> 'AlgorithmStepResult':
        if self.is_finished():
            all_ids = {t.id for t in self.expected_tags_db} - self.found_present_ids - self.found_missing_ids
            if all_ids: self.found_missing_ids.update(all_ids)
            return AlgorithmStepResult(0, 0, 0, "Finished")

        # --- 状态机调度 ---
        if self.current_state == self.STATE_DIAGNOSIS_SETUP: return self._setup_diagnosis_frame()
        if self.current_state == self.STATE_DIAGNOSIS_FRAME: return self._perform_diagnosis_slot()
        if self.current_state == self.STATE_CPT_ENGINE: return self._cpt_engine_step()
        if self.current_state == self.STATE_PRUNING_ENGINE: return self._pruning_engine_step()
        if self.current_state == self.STATE_PRUNING_AWAIT_ALOHA: return self._setup_pruning_aloha()
        if self.current_state == self.STATE_PRUNING_IN_ALOHA: return self._perform_pruning_aloha_slot()
        
        return AlgorithmStepResult(0, 0, 0, "Waiting")

    def _setup_diagnosis_frame(self) -> 'AlgorithmStepResult':
        self.active_aloha_frame = _AlohaContext(self.expected_tags_db, self.DIAGNOSIS_FRAME_SIZE, random.randint(0, 10000))
        self.current_state = self.STATE_DIAGNOSIS_FRAME
        reader_bits = self.config.get('READER_CMD_BASE_BITS', 37) + 32
        time_delta = reader_bits / self.config.get('BITS_PER_MICROSECOND', 0.16)
        return AlgorithmStepResult(time_delta, reader_bits, 0, f"Diagnosis Setup: f={self.DIAGNOSIS_FRAME_SIZE}")

    def _perform_diagnosis_slot(self) -> 'AlgorithmStepResult':
        ctx = self.active_aloha_frame
        if not ctx.actual_responses[ctx.current_slot]: ctx.empty_slot_count += 1
        ctx.current_slot += 1
        if ctx.current_slot >= ctx.frame_size: self._finalize_diagnosis()
        return AlgorithmStepResult(self.config.get('T_s_SLOT_TIME_US', 400.0), 0, 0, "Diagnosis Slot")

    def _finalize_diagnosis(self):
        ctx = self.active_aloha_frame
        empty_rate = ctx.empty_slot_count / ctx.frame_size
        
        if empty_rate < self.EMPTY_RATE_THRESHOLD:
            self.current_state = self.STATE_CPT_ENGINE
            self.task_queue.append(_Task(self.expected_tags_db))
            # print(f"诊断结果: 低丢失率 (空闲率 {empty_rate:.2%}). 启动 CPT 引擎。")
        else:
            self.current_state = self.STATE_PRUNING_ENGINE
            tags_0 = [t for t in self.expected_tags_db if t.id.startswith('0')]
            tags_1 = [t for t in self.expected_tags_db if t.id.startswith('1')]
            if tags_0: self.task_queue.append(_Task(tags_0, '0'))
            if tags_1: self.task_queue.append(_Task(tags_1, '1'))
            # print(f"诊断结果: 高丢失率 (空闲率 {empty_rate:.2%}). 启动 Pruning 引擎。")
        self.active_aloha_frame = None

    def _cpt_engine_step(self) -> 'AlgorithmStepResult':
        if not self.task_queue:
            self.current_state = self.STATE_FINISHED
            return AlgorithmStepResult(0,0,0,"CPT Engine Finished")
        current_task = self.task_queue.popleft()
        tags = current_task.tags_in_scope
        
        if len(tags) <= 4:
            total_time, total_bits_r, total_bits_t = 0, 0, 0
            for tag in tags:
                bits_r = self.config.get('BINARY_LENGTH', 96)
                time = (bits_r / self.config.get('BITS_PER_MICROSECOND', 0.16)) + self.config.get('T1_RTcal', 25.0)
                if tag.is_present:
                    self.found_present_ids.add(tag.id)
                    bits_t = 1
                    time += (bits_t / self.config.get('BITS_PER_MICROSECOND', 0.16)) + self.config.get('T2_TRcal', 25.0)
                else: self.found_missing_ids.add(tag.id); bits_t = 0
                total_time += time; total_bits_r += bits_r; total_bits_t += bits_t
            return AlgorithmStepResult(total_time, total_bits_r, total_bits_t, "CPT Small Group Polling")
        
        split_bit_idx = self._find_best_split_bit(tags)
        reader_bits = self.config.get('READER_CMD_BASE_BITS', 37) + 8
        time_delta = (reader_bits / self.config.get('BITS_PER_MICROSECOND', 0.16)) + self.config.get('T_l_SLOT_TIME_US', 800.0)
        tags_0 = [t for t in tags if t.id[split_bit_idx] == '0']
        tags_1 = [t for t in tags if t.id[split_bit_idx] == '1']
        active_0 = any(t.is_present for t in tags_0)
        active_1 = any(t.is_present for t in tags_1)
        if active_0: self.task_queue.appendleft(_Task(tags_0))
        else: self.found_missing_ids.update(t.id for t in tags_0)
        if active_1: self.task_queue.appendleft(_Task(tags_1))
        else: self.found_missing_ids.update(t.id for t in tags_1)
        return AlgorithmStepResult(time_delta, reader_bits, int(active_0) + int(active_1), f"CPT Split on bit {split_bit_idx}")
    
    def _find_best_split_bit(self, tags: list['Tag']) -> int:
        if not tags or len(tags) < 2: return 0
        tag_id_len = len(tags[0].id)
        min_diff, best_bit_idx = float('inf'), 0
        for bit_idx in range(0, tag_id_len, 8):
            count_0 = sum(1 for tag in tags if tag.id[bit_idx] == '0')
            diff = abs(len(tags) - 2 * count_0)
            if diff < min_diff: min_diff, best_bit_idx = diff, bit_idx
            if min_diff == 0: break
        return best_bit_idx

    def _pruning_engine_step(self) -> 'AlgorithmStepResult':
        if not self.task_queue:
            self.current_state = self.STATE_FINISHED
            return AlgorithmStepResult(0,0,0,"Pruning Engine Finished")
        current_task = self.task_queue.popleft()
        is_active = any(t.is_present for t in current_task.tags_in_scope)
        reader_bits = self.config.get('READER_CMD_BASE_BITS', 37) + len(current_task.prefix)
        time_delta = (reader_bits / self.config.get('BITS_PER_MICROSECOND', 0.16)) + self.config.get('T_l_SLOT_TIME_US', 800.0)
        if not is_active:
            self.found_missing_ids.update(t.id for t in current_task.tags_in_scope)
            return AlgorithmStepResult(time_delta, reader_bits, 0, f"Pruning Probe '{current_task.prefix}': Idle.")
        else:
            self.current_state = self.STATE_PRUNING_AWAIT_ALOHA
            self.pending_task = current_task
            return AlgorithmStepResult(time_delta, reader_bits, 1, f"Pruning Probe '{current_task.prefix}': Active.")
    
    def _setup_pruning_aloha(self) -> 'AlgorithmStepResult':
        current_task = self.pending_task
        self.pending_task = None
        n_subset = len(current_task.tags_in_scope)
        frame_size = max(1, n_subset)
        self.active_aloha_frame = _AlohaContext(current_task.tags_in_scope, frame_size, random.randint(0, 10000), task=current_task)
        self.current_state = self.STATE_PRUNING_IN_ALOHA
        reader_bits = self.config.get('READER_CMD_BASE_BITS', 37) + 32
        time_delta = reader_bits / self.config.get('BITS_PER_MICROSECOND', 0.16)
        return AlgorithmStepResult(time_delta, reader_bits, 0, f"Pruning ALOHA Setup: f={frame_size}.")

    def _perform_pruning_aloha_slot(self) -> 'AlgorithmStepResult':
        ctx = self.active_aloha_frame
        slot, n_resp = ctx.current_slot, len(ctx.actual_responses[ctx.current_slot])
        n_exp, exp_tags = len(ctx.expected_slots[slot]), ctx.expected_slots[slot]
        if n_resp == 1: self.found_present_ids.add(ctx.actual_responses[slot][0].id)
        elif n_exp == 1 and n_resp == 0: self.found_missing_ids.add(exp_tags[0].id)
        ctx.current_slot += 1
        if ctx.current_slot >= ctx.frame_size: self._finalize_pruning_aloha()
        return AlgorithmStepResult(self.config.get('T_s_SLOT_TIME_US', 400.0), 0, n_resp, "Pruning ALOHA Slot")

    def _finalize_pruning_aloha(self):
        ctx = self.active_aloha_frame
        collided = {tag for i in range(ctx.frame_size) if len(ctx.actual_responses[i]) > 1 for tag in ctx.expected_slots[i]}
        resolved = {r[0].id for r in ctx.actual_responses if len(r) == 1} | {e[0].id for e, r in zip(ctx.expected_slots, ctx.actual_responses) if len(e) == 1 and not r}
        self.found_missing_ids.update({t.id for t in ctx.task.tags_in_scope} - resolved - {t.id for t in collided})
        if collided:
            prefix = ctx.task.prefix
            tags_0 = [t for t in collided if t.id.startswith(prefix + '0')]
            tags_1 = [t for t in collided if t.id.startswith(prefix + '1')]
            if tags_1: self.task_queue.appendleft(_Task(tags_1, prefix + '1'))
            if tags_0: self.task_queue.appendleft(_Task(tags_0, prefix + '0'))
        self.active_aloha_frame, self.current_state = None, self.STATE_PRUNING_ENGINE


# --- Main Runner ---
def print_results(results: dict):
    """一个辅助函数，用于格式化并打印单次仿真的结果。"""
    if not results: print("仿真未能返回有效结果。"); return
    print(f"\n--- 结果汇总: {results.get('algorithm_name', 'N/A')} ---")
    print("-" * 45)
    total_time_s = results.get('total_protocol_time_us', 0) / 1e6
    print(f"协议总耗时: {total_time_s:.4f} 秒")
    total_bits = results.get('total_reader_bits', 0) + results.get('total_tag_bits', 0)
    print(f"通信总开销: {total_bits / 1024:.1f} Kbits")
    print("-" * 45)

if __name__ == '__main__':
    print("开始 TSDP 算法的独立性能测试...")

    # --- 1. 定义全局仿真配置 ---
    simulation_parameters = {
        'TOTAL_TAGS': 10000,
        'BINARY_LENGTH': 96,
        'DATA_RATE_BPS': 160000.0,
        'T_s_SLOT_TIME_US': 400.0,
        'T_l_SLOT_TIME_US': 800.0,
        'T_tag_SLOT_TIME_US': 2400.0,
        'T1_RTcal': 25.0,
        'T2_TRcal': 25.0,
        'READER_CMD_BASE_BITS': 37,
        'TAG_SHORT_RESP_BITS': 1,
    }
    simulation_parameters['BITS_PER_MICROSECOND'] = simulation_parameters['DATA_RATE_BPS'] / 1.0e6

    # --- 2. 运行仿真 ---
    # 定义一个场景进行测试，例如低丢失率
    low_missing_rate_config = simulation_parameters.copy()
    low_missing_rate_config['MISSING_RATE'] = 0.1 # 10% 丢失率

    # 定义另一个场景，高丢失率
    high_missing_rate_config = simulation_parameters.copy()
    high_missing_rate_config['MISSING_RATE'] = 0.8 # 80% 丢失率

    print("\n" + "="*50)
    print("场景一: 低丢失率 (10%)")
    print("="*50)
    tsdp_results_low = run_missing_tag_simulation(
        global_config=low_missing_rate_config,
        algorithm_class=TSDPAlgo,
        algorithm_specific_config={}
    )
    print_results(tsdp_results_low)


    print("\n" + "="*50)
    print("场景二: 高丢失率 (80%)")
    print("="*50)
    tsdp_results_high = run_missing_tag_simulation(
        global_config=high_missing_rate_config,
        algorithm_class=TSDPAlgo,
        algorithm_specific_config={}
    )
    print_results(tsdp_results_high)
        
    print("\n仿真执行结束。")