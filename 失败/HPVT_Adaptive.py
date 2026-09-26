# 自适应策略增强的HPVT算法
# 增强后的算法工作流程
# 初始化：算法启动，将根节点的两个分支 ('0' 和 '1') 作为初始任务放入队列，并设定好 ALOHA_THRESHOLD (例如 64)。

# 分组探测：算法从队列中取出一个任务（例如，前缀为 '0' 的分组）。

# 如果探测无响应：高效剪枝，将 '0' 分支下的所有标签标记为丢失。
# 如果探测有响应：进入决策点。
# 决策1：检查该分组内标签总数。假设为 1000。因为 1000 > 64，算法决定继续分裂。
# 它会创建 '00' 和 '01' 两个新任务，并将它们放回队列的前端（appendleft），实现深度优先的策略。
# 算法会继续处理 '00'，再次进行探测和决策。这个过程会一直持续，直到某个分支下的标签总数小于等于64。
# 决策2：假设算法现在处理到前缀 '001011'，其下有 50 个标签。因为 50 <= 64，算法决定启动并行验证。
# 它会为这50个标签启动一次高效的 ALOHA 帧。
# 并行验证：

# 在 ALOHA 帧中，大部分标签会通过单例时隙被识别为在场或丢失。
# 少数发生碰撞的标签会被识别出来。
# 循环：

# ALOHA 结束后，如果还有碰撞的标签，算法会为这些（数量已经变得很少的）碰撞标签再次创建更深层的树状任务，并放回队列。
# 算法回到第2步，继续处理队列中的任务，直至队列为空。
# 通过这种方式，增强后的 HPVT 算法在面对低丢失率、大规模分组时，能够通过低成本的树状分裂快速缩小问题规模，只有当问题规模缩小到“甜蜜点”（小于等于阈值）时，才调用并行验证这一“杀手锏”来终结问题，从而在所有场景下都能实现高性能。
# HPVT_Enhanced.py
import math
import random
from collections import deque

# 假设以下类已在框架中定义
from Framework import *


class _HPVT_Task:
    """HPVT的任务现在只包含标签，不再需要前缀"""
    def __init__(self, tags_in_scope: list['Tag']):
        self.tags_in_scope = tags_in_scope

class _AlohaFrameContext:
    def __init__(self, tags_in_scope: list['Tag'], frame_size: int, random_seed: int):
        self.tags_in_scope = tags_in_scope
        self.frame_size = frame_size
        self.random_seed = random_seed
        self.current_slot = 0
        self.expected_slots = [[] for _ in range(frame_size)]
        self.actual_responses = [[] for _ in range(frame_size)]
        for tag in tags_in_scope:
            slot_index = hash(tag.id + str(random_seed)) % frame_size
            self.expected_slots[slot_index].append(tag)
            if tag.is_present: self.actual_responses[slot_index].append(tag)

class HPVT_AdaptiveAlgo(MissingTagAlgorithmInterface):
    """
    分层并行验证树 (HPVT) 算法。
    !!V4最终增强版: "最优分裂" + "自适应并行验证"!!
    """
    def initialize(self, expected_tags: list['Tag']):
        self.expected_tags_db = expected_tags
        self.found_present_ids, self.found_missing_ids = set(), set()
        self.task_queue = deque()
        self.STATE_IDLE, self.STATE_AWAITING_VERIFICATION, self.STATE_IN_ALOHA_FRAME = 0, 1, 2
        self.current_state = self.STATE_IDLE
        self.pending_task = None
        self.active_aloha_frame = None
        self.ALOHA_THRESHOLD = self.config.get('ALOHA_THRESHOLD', 64)

        if self.expected_tags_db:
            self.task_queue.append(_HPVT_Task(self.expected_tags_db))

    def is_finished(self) -> bool: return not self.task_queue and self.current_state == self.STATE_IDLE
    def get_results(self) -> tuple[set[str], set[str]]: return self.found_present_ids, self.found_missing_ids

    def perform_step(self) -> 'AlgorithmStepResult':
        if self.is_finished():
            all_ids = {t.id for t in self.expected_tags_db} - self.found_present_ids - self.found_missing_ids
            if all_ids: self.found_missing_ids.update(all_ids)
            return AlgorithmStepResult(0, 0, 0, "finished")

        if self.current_state == self.STATE_IN_ALOHA_FRAME: return self._perform_aloha_slot()
        if self.current_state == self.STATE_AWAITING_VERIFICATION: return self._setup_aloha_frame()
        if self.task_queue: return self._process_task()
        return AlgorithmStepResult(0, 0, 0, "waiting")

    # --- 核心调度逻辑 ---
    def _process_task(self) -> 'AlgorithmStepResult':
        current_task = self.task_queue.popleft()
        tags, n_tags = current_task.tags_in_scope, len(current_task.tags_in_scope)
        
        # 决策点：根据任务规模决定是分裂还是并行验证
        if n_tags > self.ALOHA_THRESHOLD:
            return self._perform_balanced_split(current_task)
        else:
            # 在进入并行验证前，先做一次快速的整体探测
            is_active = any(t.is_present for t in tags)
            reader_bits = self.config.get('READER_CMD_BASE_BITS', 37)
            time_delta = (reader_bits / self.config.get('BITS_PER_MICROSECOND', 0.16)) + self.config.get('T_s_SLOT_TIME_US', 400.0)

            if not is_active:
                self.found_missing_ids.update(t.id for t in tags)
                return AlgorithmStepResult(time_delta, reader_bits, 0, "Final Probe: Idle. Pruned.")
            
            self.current_state = self.STATE_AWAITING_VERIFICATION
            self.pending_task = current_task
            return AlgorithmStepResult(time_delta, reader_bits, 1, "Final Probe: Active. Init ALOHA.")


    # --- V4新增：最优分裂逻辑 ---
    def _find_best_split_bit(self, tags: list['Tag']) -> int:
        """从标签ID中找到能最均衡地划分当前集合的比特位索引"""
        if not tags or len(tags) < 2: return 0
        
        tag_id_len = len(tags[0].id)
        min_diff = float('inf')
        best_bit_idx = 0
        
        # 为了效率，我们只检查一部分代表性的比特位
        # 您可以根据需要调整检查的步长，例如 range(0, tag_id_len, 4)
        for bit_idx in range(0, tag_id_len, 8): # 优化检查步长
            count_0 = sum(1 for tag in tags if tag.id[bit_idx] == '0')
            diff = abs(len(tags) - 2 * count_0)
            if diff < min_diff:
                min_diff = diff
                best_bit_idx = bit_idx
                if min_diff == 0: # 已经完美均衡，提前退出
                    break
        return best_bit_idx

    def _perform_balanced_split(self, task: _HPVT_Task) -> 'AlgorithmStepResult':
        """执行一次基于最优分裂位的树状分解"""
        tags = task.tags_in_scope
        split_bit_idx = self._find_best_split_bit(tags)

        # 模拟读写器广播查询（例如，一个包含分裂位索引的命令）
        reader_bits = self.config.get('READER_CMD_BASE_BITS', 37) + math.ceil(math.log2(len(tags[0].id)))
        time_delta = (reader_bits / self.config.get('BITS_PER_MICROSECOND', 0.16)) + self.config.get('T_l_SLOT_TIME_US', 800.0)
        
        tags_0 = [t for t in tags if t.id[split_bit_idx] == '0']
        tags_1 = [t for t in tags if t.id[split_bit_idx] == '1']
        
        active_0 = any(t.is_present for t in tags_0)
        active_1 = any(t.is_present for t in tags_1)
        
        # 根据子集的响应情况创建新任务
        if active_0:
            self.task_queue.appendleft(_HPVT_Task(tags_0))
        else: # 如果探测到0分组是空闲的，则该组全部丢失
            self.found_missing_ids.update(t.id for t in tags_0)

        if active_1:
            self.task_queue.appendleft(_HPVT_Task(tags_1))
        else: # 如果探测到1分组是空闲的，则该组全部丢失
            self.found_missing_ids.update(t.id for t in tags_1)

        return AlgorithmStepResult(time_delta, reader_bits, int(active_0) + int(active_1), f"Split on bit {split_bit_idx}")
        
    def _setup_aloha_frame(self) -> 'AlgorithmStepResult':
        """执行“并行验证准备”微步骤，采用理论最优帧长。"""
        current_task = self.pending_task
        self.pending_task = None
        tags_in_scope = current_task.tags_in_scope
        n_subset = len(tags_in_scope)
        
        # 使用理论最优负载因子 f=N
        frame_size = max(1, n_subset)

        self.active_aloha_frame = _AlohaFrameContext(tags_in_scope, frame_size, random.randint(0, 10000))
        self.current_state = self.STATE_IN_ALOHA_FRAME
        reader_bits = self.config.get('READER_CMD_BASE_BITS', 37) + 32
        time_delta = reader_bits / self.config.get('BITS_PER_MICROSECOND', 0.16)
        return AlgorithmStepResult(time_delta, reader_bits, 0, f"ALOHA Setup: f={frame_size}.")

    def _perform_aloha_slot(self) -> 'AlgorithmStepResult':
        ctx = self.active_aloha_frame
        slot, n_resp = ctx.current_slot, len(ctx.actual_responses[ctx.current_slot])
        n_exp, exp_tags = len(ctx.expected_slots[slot]), ctx.expected_slots[slot]
        if n_resp == 1: self.found_present_ids.add(ctx.actual_responses[slot][0].id)
        elif n_exp == 1 and n_resp == 0: self.found_missing_ids.add(exp_tags[0].id)
        
        ctx.current_slot += 1
        if ctx.current_slot >= ctx.frame_size: self._finalize_aloha_frame()
        return AlgorithmStepResult(self.config.get('T_s_SLOT_TIME_US', 400.0), 0, n_resp, "ALOHA Slot")

    def _finalize_aloha_frame(self):
        ctx = self.active_aloha_frame
        all_tags_in_frame = ctx.tags_in_scope
        
        resolved_ids = {r[0].id for r in ctx.actual_responses if len(r) == 1} | {e[0].id for e, r in zip(ctx.expected_slots, ctx.actual_responses) if len(e) == 1 and not r}
        collided_tags = {tag for i in range(ctx.frame_size) if len(ctx.actual_responses[i]) > 1 for tag in ctx.expected_slots[i]}
        
        self.found_missing_ids.update({t.id for t in all_tags_in_frame} - resolved_ids - {t.id for t in collided_tags})
        
        if collided_tags:
            self.task_queue.appendleft(_HPVT_Task(list(collided_tags)))
        
        self.active_aloha_frame = None
        self.current_state = self.STATE_IDLE
# --- 主程序调用入口 ---

if __name__ == '__main__':
    print("开始缺失标签识别算法的对比仿真...")

    # --- 1. 定义全局仿真配置 ---
    simulation_parameters = {
        'TOTAL_TAGS': 10000,
        'MISSING_RATE': 0.01, # 1% 的标签丢失
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
    
    # 运行基线轮询算法作为对比基准
    print("\n>>> 正在运行 BaselinePollingAlgo...")
    baseline_results = run_missing_tag_simulation(
        global_config=simulation_parameters,
        algorithm_class=BaselinePollingAlgo,
        algorithm_specific_config={}
    )

    # 运行您新设计的增强版HPVT算法
    print("\n>>> 正在运行 HPVT_AdaptiveAlgo (自适应版)...")
    hpvt_results = run_missing_tag_simulation(
        global_config=simulation_parameters,
        algorithm_class=HPVT_AdaptiveAlgo,
        algorithm_specific_config={}
    )

    # --- 3. 统一打印和对比结果 ---
    print("\n" + "="*50)
    print("                   仿 真 结 果 对 比")
    print("="*50)
    
    # 为了便于比较，我们先对结果进行单位转换
    if baseline_results:
        baseline_results['total_protocol_time_us'] = baseline_results.get('total_protocol_time_us', 0)
    if hpvt_results:
        hpvt_results['total_protocol_time_us'] = hpvt_results.get('total_protocol_time_us', 0)

    print_results(baseline_results)
    print_results(hpvt_results)
    
    print("\n\n--- 性能对比总结 ---")
    if baseline_results and hpvt_results:
        baseline_time = baseline_results.get('total_protocol_time_us', float('inf'))
        hpvt_time = hpvt_results.get('total_protocol_time_us', float('inf'))

        if baseline_time > 0 and hpvt_time > 0 and hpvt_time != float('inf'):
            time_reduction_hpvt = (1 - hpvt_time / baseline_time) * 100
            print(f"时间效率: HPVT 算法相比基线轮询，执行时间减少了 {time_reduction_hpvt:.2f}%")
        else:
            print("无法计算时间效率对比。")
    else:
        print("部分算法未能成功运行，无法进行性能对比。")
        
    print("\n仿真执行结束。")