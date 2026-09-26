import math
import random

from Framework import *
class _IIPFrameContext:
    """用于存储当前正在执行的 IIP ALOHA 帧的上下文信息。"""
    def __init__(self, unverified_tags: list['Tag'], frame_size: int, random_seed: int, final_check_mode: bool):
        self.frame_size = frame_size
        self.random_seed = random_seed
        self.final_check_mode = final_check_mode
        self.current_slot = 0
        
        # --- 预计算 ---
        # 1. 计算期望时隙和 pre-frame 向量
        self.expected_slots = [[] for _ in range(frame_size)]
        for tag in unverified_tags:
            slot_index = hash(tag.id + str(random_seed)) % frame_size
            self.expected_slots[slot_index].append(tag)
        
        self.pre_frame_vector = [1 if len(tags_in_slot) > 1 else 0 for tags_in_slot in self.expected_slots]
        
        # 2. 计算真实响应
        self.actual_responses = [[] for _ in range(frame_size)]
        present_unverified_tags = [tag for tag in unverified_tags if tag.is_present]
        
        for tag in present_unverified_tags:
            slot_index = hash(tag.id + str(self.random_seed)) % frame_size
            should_respond = True
            
            if not self.final_check_mode and self.pre_frame_vector[slot_index] == 1:
                if hash(tag.id + str(self.random_seed) + "second_hash") % 2 == 0:
                    should_respond = False
            
            if should_respond:
                self.actual_responses[slot_index].append(tag)
        
        self.tags_that_responded_count = sum(len(s) for s in self.actual_responses)


class IIPAlgo(MissingTagAlgorithmInterface):
    """
    论文 'Identifying the Missing Tags in a Large RFID System' 中提出的
    迭代式无ID协议 (Iterative ID-free Protocol, IIP) 的实现。
    (已修正为微步化执行模式)
    """
    
    def initialize(self, expected_tags: list['Tag']):
        """初始化算法状态。"""
        self.expected_tags_db = expected_tags
        self.unverified_tags = list(self.expected_tags_db)
        self.found_present_ids = set()
        self.found_missing_ids = set()
        
        self.random_seed = 0
        self.optimal_rho = 1.516
        self.final_check_mode = False
        
        self.STATE_IDLE = 0
        self.STATE_IN_FRAME = 1
        self.current_state = self.STATE_IDLE
        self.active_frame_context = None

    def is_finished(self) -> bool:
        """当不再有未验证的标签且状态机空闲时，算法结束。"""
        return not self.unverified_tags and self.current_state == self.STATE_IDLE

    def get_results(self) -> tuple[set[str], set[str]]:
        """返回逐步确认的集合，以支持时间线统计。"""
        return self.found_present_ids, self.found_missing_ids

    def perform_step(self) -> 'AlgorithmStepResult':
        """执行一个微步骤，要么是帧设置，要么是单时隙处理。"""
        if self.is_finished():
            all_ids_in_db = {t.id for t in self.expected_tags_db}
            unconfirmed_ids = all_ids_in_db - self.found_present_ids - self.found_missing_ids
            self.found_missing_ids.update(unconfirmed_ids)
            return AlgorithmStepResult(0, 0, 0, "finished")
        
        if self.current_state == self.STATE_IN_FRAME:
            return self._perform_iip_slot()
        
        return self._setup_iip_frame()

    def _setup_iip_frame(self) -> 'AlgorithmStepResult':
        """执行“帧设置”微步骤：计算并广播帧头向量。"""
        n_star = len(self.unverified_tags)
        if n_star == 0:
             self.current_state = self.STATE_IDLE
             return AlgorithmStepResult(0,0,0, "all verified")

        # !!已修改!!: 修正帧长计算逻辑，以避免“碰撞僵局”
        # 论文中的最优负载因子适用于大N*，当N*很小时，会导致帧长过小而必然碰撞
        frame_size = int(round(n_star / self.optimal_rho)) if self.optimal_rho > 0 else n_star
        # 增加保护机制：当剩余标签很少时，确保帧长至少不小于标签数，以给标签留出足够的空间
        if n_star > 0:
            frame_size = max(frame_size, n_star)
        if frame_size == 0: 
            frame_size = 1

        self.active_frame_context = _IIPFrameContext(self.unverified_tags, frame_size, self.random_seed, self.final_check_mode)
        self.current_state = self.STATE_IN_FRAME
        
        t_tag_slot = self.config.get('T_tag_SLOT_TIME_US', 2400.0)
        time_for_vectors = 2 * math.ceil(frame_size / 96.0) * t_tag_slot
        reader_bits = 2 * frame_size
        
        op_desc = f"IIP Frame Setup: N*={n_star}, f={frame_size}"
        return AlgorithmStepResult(time_delta_us=time_for_vectors, reader_bits=reader_bits, tag_bits=0, operation_description=op_desc)

    def _perform_iip_slot(self) -> 'AlgorithmStepResult':
        """执行“单时隙处理”微步骤。"""
        frame_ctx = self.active_frame_context
        slot_idx = frame_ctx.current_slot
        
        responded_in_slot = frame_ctx.actual_responses[slot_idx]
        expected_in_slot = frame_ctx.expected_slots[slot_idx]
        
        if len(responded_in_slot) == 1:
            identified_tag = responded_in_slot[0]
            if identified_tag.id not in self.found_present_ids:
                 self.found_present_ids.add(identified_tag.id)
        elif len(responded_in_slot) == 0 and len(expected_in_slot) == 1:
            missing_tag = expected_in_slot[0]
            if missing_tag.id not in self.found_missing_ids:
                self.found_missing_ids.add(missing_tag.id)
        
        t_s_slot = self.config.get('T_s_SLOT_TIME_US', 400.0)
        tag_bits_slot = len(responded_in_slot) * self.config.get('TAG_SHORT_RESP_BITS', 1)

        frame_ctx.current_slot += 1
        
        if frame_ctx.current_slot >= frame_ctx.frame_size:
            self._finalize_iip_frame()
            
        op_desc = f"IIP Slot {slot_idx+1}/{frame_ctx.frame_size}"
        return AlgorithmStepResult(time_delta_us=t_s_slot, reader_bits=0, tag_bits=tag_bits_slot, operation_description=op_desc)

    def _finalize_iip_frame(self):
        """在一轮IIP帧结束后，更新状态。"""
        frame_ctx = self.active_frame_context
        
        resolved_ids_this_frame = set()
        for i in range(frame_ctx.frame_size):
            if len(frame_ctx.actual_responses[i]) == 1:
                resolved_ids_this_frame.add(frame_ctx.actual_responses[i][0].id)
            elif len(frame_ctx.actual_responses[i]) == 0 and len(frame_ctx.expected_slots[i]) == 1:
                resolved_ids_this_frame.add(frame_ctx.expected_slots[i][0].id)
        
        if resolved_ids_this_frame:
            self.final_check_mode = False
            self.unverified_tags = [t for t in self.unverified_tags if t.id not in resolved_ids_this_frame]
        else:
            if frame_ctx.tags_that_responded_count == 0 and self.unverified_tags:
                if not self.final_check_mode:
                    self.final_check_mode = True
                else:
                    for tag in self.unverified_tags:
                        self.found_missing_ids.add(tag.id)
                    self.unverified_tags = []
        
        self.random_seed += 1
        self.current_state = self.STATE_IDLE
        self.active_frame_context = None

if __name__ == '__main__':
    print("开始IIP算法的独立仿真测试...")

    # --- 1. 定义全局仿真配置 ---
    #    这里的参数可以根据您的需要进行调整
    simulation_parameters = {
        'TOTAL_TAGS': 10000,
        'MISSING_RATE': 0, # 固定丢失率为50%进行测试
        'BINARY_LENGTH': 96,
        'DATA_RATE_BPS': 160000.0,
        
        # 物理层和链路层时间常量 (µs)
        'T_s_SLOT_TIME_US': 400.0,    # 短时隙, IIP 主要使用
        'T_l_SLOT_TIME_US': 800.0,    # 长时隙
        'T_tag_SLOT_TIME_US': 2400.0, # ID时隙或用于向量传输
        
        # 框架所需的其他详细参数
        'T1_RTcal': 25.0,
        'T2_TRcal': 25.0,
        'READER_CMD_BASE_BITS': 37,
        'TAG_SHORT_RESP_BITS': 1,
    }
    simulation_parameters['BITS_PER_MICROSECOND'] = simulation_parameters['DATA_RATE_BPS'] / 1.0e6

    # --- 2. 运行仿真 ---
    
    # 为了对比，首先运行一个简单的基线算法
    # 注意: BaselinePollingAlgo需要存在于您的框架或此文件中
    print("\n>>> 开始运行 BaselinePollingAlgo...")
    baseline_results = run_missing_tag_simulation(
        global_config=simulation_parameters,
        algorithm_class=BaselinePollingAlgo,
        algorithm_specific_config={}
    )

    # 运行您要测试的IIP算法
    print("\n>>> 开始运行 IIPAlgo...")
    iip_results = run_missing_tag_simulation(
        global_config=simulation_parameters,
        algorithm_class=IIPAlgo,
        algorithm_specific_config={}
    )

    # --- 3. 统一打印和对比结果 ---
    print("\n" + "="*50)
    print("                  仿 真 结 果 对 比")
    print("="*50)
    
    # 假设您有一个 print_results 函数来格式化输出
    print_results(baseline_results)
    print_results(iip_results)
    
    print("\n--- 性能对比总结 ---")
    if baseline_results and iip_results:
        baseline_time = baseline_results.get('total_protocol_time_us', float('inf'))
        iip_time = iip_results.get('total_protocol_time_us', float('inf'))

        if baseline_time > 0 and iip_time > 0 and iip_time != float('inf'):
            time_reduction = (1 - iip_time / baseline_time) * 100
            print(f"时间效率: IIP算法相比基线轮询，执行时间减少了 {time_reduction:.2f}%")
        else:
            print("无法计算时间效率对比。")
    else:
        print("部分算法未能成功运行，无法进行性能对比。")
        
    print("\nIIP算法独立测试执行结束。")