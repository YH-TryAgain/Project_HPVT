# 核心创新点: HPVT 算法将传统碰撞树的确定性前缀分组与现代并行协议的高效随机验证相结合。
# 它不像IIP那样对所有标签进行全局随机散列，也不像CPT那样需要复杂的离线计算来寻找最优分裂位。HPVT使用简单的前缀进行分层“剪枝”，
# 然后在有必要时，才对规模可控的子集启动高效的并行验证，形成一种宏观确定性剪枝、微观并行化处理的新模式。
import math
import random
from collections import deque

# 假设以下类已在框架中定义
from Framework import *


class _HPVT_Task:
    """HPVT算法内部使用的任务对象。"""
    def __init__(self, prefix: str, tags_in_scope: list['Tag']):
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
        
        # 预计算所有标签在帧中的位置
        for tag in task.tags_in_scope:
            slot_index = hash(tag.id + str(random_seed)) % frame_size
            self.expected_slots[slot_index].append(tag)
            if tag.is_present:
                self.actual_responses[slot_index].append(tag)


class HPVTAlgo(MissingTagAlgorithmInterface):
    """
    分层并行验证树 (Hierarchical Parallel Verification Tree - HPVT) 算法。
    """
    
    def initialize(self, expected_tags: list['Tag']):
        """初始化算法，创建顶层任务并加入队列。"""
        self.expected_tags_db = expected_tags
        self.found_present_ids = set()
        self.found_missing_ids = set()
        self.task_queue = deque()
        
        # !!新增!!: 用于管理精细化步骤的状态机
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

    def get_results(self) -> tuple[set[str], set[str]]:
        """直接返回逐步确认的集合。"""
        return self.found_present_ids, self.found_missing_ids

    def perform_step(self) -> 'AlgorithmStepResult':
        """!!已修改!!: 执行一个真正的“微步骤”，每次调用只处理一个不可再分的操作。"""
        if self.is_finished():
            # 最后的安全检查：将所有仍未确认的标签标记为丢失
            # 在理想逻辑下，unconfirmed_ids 应该为空
            all_ids_in_db = {t.id for t in self.expected_tags_db}
            unconfirmed_ids = all_ids_in_db - self.found_present_ids - self.found_missing_ids
            if unconfirmed_ids:
                self.found_missing_ids.update(unconfirmed_ids)
            return AlgorithmStepResult(0, 0, 0, "finished")

        if self.current_state == self.STATE_IN_ALOHA_FRAME:
            return self._perform_aloha_slot()
        
        if self.current_state == self.STATE_AWAITING_VERIFICATION:
            return self._setup_parallel_verification()

        # 默认状态: STATE_IDLE
        if self.task_queue:
            return self._perform_group_probe()

        return AlgorithmStepResult(0, 0, 0, "waiting")

    def _perform_group_probe(self) -> 'AlgorithmStepResult':
        """执行“分组探测”微步骤。"""
        current_task = self.task_queue.popleft()
        responding_tags = [t for t in current_task.tags_in_scope if t.is_present]
        
        reader_bits_probe = self.config['READER_CMD_BASE_BITS'] + len(current_task.prefix)
        time_delta_probe = (reader_bits_probe / self.config['BITS_PER_MICROSECOND']) + self.config.get('T_l_SLOT_TIME_US', 800.0)

        if not responding_tags:
            # 高效剪枝：该分支所有标签都确定为丢失。状态不变，继续处理下一个任务。
            tags_in_scope_ids = {t.id for t in current_task.tags_in_scope}
            self.found_missing_ids.update(tags_in_scope_ids)
            op_desc = f"HPVT Probe on '{current_task.prefix}': Idle. Pruned."
            return AlgorithmStepResult(time_delta_us=time_delta_probe, reader_bits=reader_bits_probe, tag_bits=0, operation_description=op_desc)
        else:
            # 探测非空：进入等待验证状态，为下一步准备。
            self.current_state = self.STATE_AWAITING_VERIFICATION
            self.pending_task = current_task
            op_desc = f"HPVT Probe on '{current_task.prefix}': Active."
            return AlgorithmStepResult(time_delta_us=time_delta_probe, reader_bits=reader_bits_probe, tag_bits=1, operation_description=op_desc)

    def _setup_parallel_verification(self) -> 'AlgorithmStepResult':
        """执行“并行验证准备”微步骤：仅广播ALOHA帧头，不处理时隙。"""
        current_task = self.pending_task
        self.pending_task = None

        n_subset = len(current_task.tags_in_scope)
        frame_size = n_subset if n_subset > 0 else 1
        random_seed = random.randint(0, 10000)

        # 创建并存储ALOHA帧的上下文
        self.active_aloha_frame = _AlohaFrameContext(current_task, frame_size, random_seed)
        self.current_state = self.STATE_IN_ALOHA_FRAME

        # 此步骤的开销只是发送帧头命令
        reader_bits_setup = self.config['READER_CMD_BASE_BITS'] + 32  # 种子和帧长
        time_delta_setup = reader_bits_setup / self.config['BITS_PER_MICROSECOND']
        op_desc = f"HPVT ALOHA Setup for '{current_task.prefix}': f={frame_size}."
        
        return AlgorithmStepResult(time_delta_us=time_delta_setup, reader_bits=reader_bits_setup, tag_bits=0, operation_description=op_desc)

    def _perform_aloha_slot(self) -> 'AlgorithmStepResult':
        """执行“并行验证单时隙处理”微步骤。"""
        frame_ctx = self.active_aloha_frame
        slot_idx = frame_ctx.current_slot
        
        # 分析当前时隙的结果
        expected_in_slot = frame_ctx.expected_slots[slot_idx]
        responded_in_slot = frame_ctx.actual_responses[slot_idx]
        
        if len(responded_in_slot) == 1:
            self.found_present_ids.add(responded_in_slot[0].id)
        elif len(expected_in_slot) == 1 and not responded_in_slot:
            self.found_missing_ids.add(expected_in_slot[0].id)
        
        # 此步骤的开销只是一个短时隙的时间
        time_delta_slot = self.config.get('T_s_SLOT_TIME_US', 400.0)
        tag_bits_slot = len(responded_in_slot) * self.config.get('TAG_SHORT_RESP_BITS', 1)

        # 移动到下一个时隙
        frame_ctx.current_slot += 1

        # 检查ALOHA帧是否已处理完毕
        if frame_ctx.current_slot >= frame_ctx.frame_size:
            self._finalize_aloha_frame()

        return AlgorithmStepResult(time_delta_us=time_delta_slot, reader_bits=0, tag_bits=tag_bits_slot, operation_description=f"ALOHA slot {slot_idx+1}/{frame_ctx.frame_size}")

    def _finalize_aloha_frame(self):
        """在ALOHA帧所有时隙处理完毕后，进行清理和后续任务创建。"""
        frame_ctx = self.active_aloha_frame
        
        collided_tags = set()
        # 识别该帧中所有发生碰撞的标签
        for i in range(frame_ctx.frame_size):
            if len(frame_ctx.actual_responses[i]) > 1: # 碰撞
                for tag in frame_ctx.expected_slots[i]:
                    collided_tags.add(tag)

        # 识别该帧中所有已解决的标签（单例响应或单例空闲）
        resolved_ids = set()
        for i in range(frame_ctx.frame_size):
             if len(frame_ctx.actual_responses[i]) == 1:
                 resolved_ids.add(frame_ctx.actual_responses[i][0].id)
             elif len(frame_ctx.expected_slots[i]) == 1 and not frame_ctx.actual_responses[i]:
                 resolved_ids.add(frame_ctx.expected_slots[i][0].id)

        # !!关键逻辑!!: 识别“隐式”丢失的标签
        # 这些标签在帧范围内，但既未解决，也未碰撞（例如，在有多个期望标签但无响应的时隙中）
        all_ids_in_frame_task = {t.id for t in frame_ctx.original_task.tags_in_scope}
        collided_ids = {t.id for t in collided_tags}
        
        implicitly_missing_ids = all_ids_in_frame_task - resolved_ids - collided_ids
        self.found_missing_ids.update(implicitly_missing_ids)

        # 为碰撞的标签创建新任务
        if collided_tags:
            prefix = frame_ctx.original_task.prefix
            prefix_0 = prefix + '0'
            prefix_1 = prefix + '1'
            tags_0 = [t for t in collided_tags if t.id.startswith(prefix_0)]
            tags_1 = [t for t in collided_tags if t.id.startswith(prefix_1)]

            if tags_0: self.task_queue.append(_HPVT_Task(prefix=prefix_0, tags_in_scope=tags_0))
            if tags_1: self.task_queue.append(_HPVT_Task(prefix=prefix_1, tags_in_scope=tags_1))
        
        # 清理状态，返回IDLE，准备处理下一个任务
        self.active_aloha_frame = None
        self.current_state = self.STATE_IDLE


if __name__ == '__main__':
    print("开始缺失标签识别算法的对比仿真...")

    # --- 1. 定义全局仿真配置 ---
    #    这里的参数可以根据您的需要进行调整
    simulation_parameters = {
        'TOTAL_TAGS': 20000,
        'MISSING_RATE': 0.1, # 1% 的标签丢失
        'BINARY_LENGTH': 96,
        'DATA_RATE_BPS': 160000.0,
        
        # 物理层和链路层时间常量 (µs)
        'T_s_SLOT_TIME_US': 400.0,    # 0.4ms (短时隙)
        'T_l_SLOT_TIME_US': 800.0,    # 0.8ms (长时隙)
        'T_tag_SLOT_TIME_US': 2400.0, # 2.4ms (ID时隙)
        
        # 框架所需的其他详细参数
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

    # 运行您新设计的HPVT算法
    print("\n>>> 正在运行 HPVTAlgo...")
    hpvt_results = run_missing_tag_simulation(
        global_config=simulation_parameters,
        algorithm_class=HPVTAlgo,
        algorithm_specific_config={}
    )

    # --- 3. 统一打印和对比结果 ---
    print("\n" + "="*50)
    print("              仿 真 结 果 对 比")
    print("="*50)
    
    print_results(baseline_results)
    print_results(hpvt_results)
    
    print("\n\n--- 性能对比总结 ---")
    if baseline_results and hpvt_results:
        baseline_time = baseline_results.get('total_protocol_time_us', float('inf'))
        hpvt_time = hpvt_results.get('total_protocol_time_us', float('inf'))

        if baseline_time > 0 and hpvt_time > 0:
            time_reduction_hpvt = (1 - hpvt_time / baseline_time) * 100
            print(f"时间效率: HPVT 算法相比基线轮询，执行时间减少了 {time_reduction_hpvt:.2f}%")
        else:
            print("无法计算时间效率对比。")
    else:
        print("部分算法未能成功运行，无法进行性能对比。")
        
    print("\n仿真执行结束。")