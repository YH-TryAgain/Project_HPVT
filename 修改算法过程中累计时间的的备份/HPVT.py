# 核心创新点: HPVT 算法将传统碰撞树的确定性前缀分组与现代并行协议的高效随机验证相结合。
# 它不像IIP那样对所有标签进行全局随机散列，也不像CPT那样需要复杂的离线计算来寻找最优分裂位。HPVT使用简单的前缀进行分层“剪枝”，
# 然后在有必要时，才对规模可控的子集启动高效的并行验证，形成一种宏观确定性剪枝、微观并行化处理的新模式。
import math
from collections import deque
from Framework_Update import *

# 假设以下类已在框架中定义，这里仅为类型提示和清晰性
# from missing_tag_simulation_framework import (
#     MissingTagAlgorithmInterface,
#     AlgorithmStepResult,
#     Tag
# )

class _HPVT_Task:
    """HPVT算法内部使用的任务对象。"""
    def __init__(self, prefix: str, tags_in_scope: list['Tag']):
        self.prefix = prefix
        self.tags_in_scope = tags_in_scope

class HPVTAlgo(MissingTagAlgorithmInterface):
    """
    分层并行验证树 (Hierarchical Parallel Verification Tree - HPVT) 算法。
    
    该算法基于用户原创的RCT思路演进而来，结合了确定性前缀分组
    与高效的并行ALOHA验证机制。
    """
    
    def initialize(self, expected_tags: list['Tag']):
        """
        初始化算法，创建顶层任务并加入队列。
        """
        self.expected_tags_db = expected_tags
        self.found_present_ids = set()
        self.task_queue = deque()

        if not self.expected_tags_db:
            return

        # 步骤 A: 初始化 - 创建顶层任务 (前缀 '0' 和 '1')
        tags_0 = [t for t in self.expected_tags_db if t.id.startswith('0')]
        tags_1 = [t for t in self.expected_tags_db if t.id.startswith('1')]
        
        if tags_0:
            self.task_queue.append(_HPVT_Task(prefix='0', tags_in_scope=tags_0))
        if tags_1:
            self.task_queue.append(_HPVT_Task(prefix='1', tags_in_scope=tags_1))

    def is_finished(self) -> bool:
        """当任务队列为空时，表示所有标签状态都已确定。"""
        return not self.task_queue

    def get_results(self) -> tuple[set[str], set[str]]:
        """返回最终识别结果。"""
        all_ids_in_db = {t.id for t in self.expected_tags_db}
        missing_ids = all_ids_in_db - self.found_present_ids
        return self.found_present_ids, missing_ids

    def perform_step(self) -> 'AlgorithmStepResult':
        """
        执行一个步骤：处理一个任务，可能包含“分组探测”和“并行验证”。
        """
        if self.is_finished():
            return AlgorithmStepResult(0, 0, 0, "finished")

        current_task = self.task_queue.popleft()
        
        # --- 步骤 B.1: 分组探测 (Group Probe) ---
        # 模拟读写器广播 Query(prefix)，并等待在一个长时隙中的响应
        responding_tags = [t for t in current_task.tags_in_scope if t.is_present]
        
        # 计算该探测步骤的开销
        reader_bits_probe = self.config['READER_CMD_BASE_BITS'] + len(current_task.prefix)
        time_delta_probe = (reader_bits_probe / self.config['BITS_PER_MICROSECOND']) + self.config.get('T_l_SLOT_TIME_US', 800.0)

        if not responding_tags:
            # 高效剪枝：探测结果为空闲，该前缀下的所有标签都确定为丢失。
            # 无需进行后续操作，直接返回本次探测的开销。
            op_desc = f"HPVT Probe on '{current_task.prefix}': Idle. Pruned {len(current_task.tags_in_scope)} tags."
            return AlgorithmStepResult(
                time_delta_us=time_delta_probe,
                reader_bits=reader_bits_probe,
                tag_bits=0, # 无响应
                operation_description=op_desc
            )

        # --- 步骤 B.2: 并行验证 (Parallel Verification) ---
        # 探测结果非空，需要对该子集进行ALOHA帧验证
        
        # 为该子集运行一轮并行的ALOHA帧
        subset_tags = current_task.tags_in_scope
        n_subset = len(subset_tags)
        # 使用一个简单的负载因子来确定帧长，这里用1.0作为示例
        frame_size = n_subset
        if frame_size == 0: frame_size = 1

        random_seed = random.randint(0, 10000) # 为本轮ALOHA生成随机种子

        # 模拟读写器预计算和标签响应
        expected_slots = [[] for _ in range(frame_size)]
        actual_responses = [[] for _ in range(frame_size)]
        
        for tag in subset_tags:
            slot_index = hash(tag.id + str(random_seed)) % frame_size
            expected_slots[slot_index].append(tag)
            if tag.is_present:
                actual_responses[slot_index].append(tag)

        # 分析ALOHA结果
        identified_in_frame = []
        unresolved_in_frame = []
        
        for i in range(frame_size):
            if len(actual_responses[i]) == 1:
                # 单例时隙：成功识别一个在场标签
                identified_in_frame.append(actual_responses[i][0])
            elif len(expected_slots[i]) == 1 and not actual_responses[i]:
                # 期望单例但实际为空：成功识别一个丢失标签
                # 这个标签的状态已确定，无需再处理
                pass
            elif len(actual_responses[i]) > 1:
                # 碰撞时隙：将这些标签加入待下一轮递归处理的列表
                unresolved_in_frame.extend(expected_slots[i])

        # 更新已确认在场的标签
        for tag in identified_in_frame:
            self.found_present_ids.add(tag.id)

        # --- 步骤 B.3: 递归创建新任务 ---
        if unresolved_in_frame:
            # 对所有碰撞的标签，按下一位前缀创建新的、更深的任务
            prefix_0 = current_task.prefix + '0'
            prefix_1 = current_task.prefix + '1'
            tags_0 = [t for t in unresolved_in_frame if t.id.startswith(prefix_0)]
            tags_1 = [t for t in unresolved_in_frame if t.id.startswith(prefix_1)]

            if tags_0: self.task_queue.append(_HPVT_Task(prefix=prefix_0, tags_in_scope=tags_0))
            if tags_1: self.task_queue.append(_HPVT_Task(prefix=prefix_1, tags_in_scope=tags_1))

        # --- 计算并返回总开销 ---
        # ALOHA帧的开销：一个带参数的查询 + f个短时隙
        reader_bits_aloha = self.config['READER_CMD_BASE_BITS'] + 32 # 假设种子和帧长用32位
        time_aloha = (reader_bits_aloha / self.config['BITS_PER_MICROSECOND']) + (frame_size * self.config.get('T_s_SLOT_TIME_US', 400.0))
        tag_bits_aloha = sum(len(slot) for slot in actual_responses) * self.config.get('TAG_SHORT_RESP_BITS', 1)

        # 总开销 = 分组探测开销 + 并行验证开销
        total_time = time_delta_probe + time_aloha
        total_reader_bits = reader_bits_probe + reader_bits_aloha
        total_tag_bits = tag_bits_aloha
        
        op_desc = f"HPVT Step on '{current_task.prefix}': Probe + ALOHA(f={frame_size}). New tasks: {len(self.task_queue)}."

        return AlgorithmStepResult(
            time_delta_us=total_time,
            reader_bits=total_reader_bits,
            tag_bits=total_tag_bits,
            operation_description=op_desc
        )


if __name__ == '__main__':
    print("开始缺失标签识别算法的对比仿真...")

    # --- 1. 定义全局仿真配置 ---
    #    这里的参数可以根据您的需要进行调整
    simulation_parameters = {
        'TOTAL_TAGS': 50000,
        'MISSING_RATE': 0.01, # 1% 的标签丢失
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