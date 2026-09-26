import math
import random

from Framework_Update import *

# 假设以下类已在框架中定义，这里仅为类型提示和清晰性
# from missing_tag_simulation_framework import (
#     MissingTagAlgorithmInterface,
#     AlgorithmStepResult,
#     Tag
# )

class IIPAlgo(MissingTagAlgorithmInterface):
    """
    论文 'Identifying the Missing Tags in a Large RFID System' 中提出的
    迭代式无ID协议 (Iterative ID-free Protocol, IIP) 的实现。
    
    本实现严格遵循 MissingTagAlgorithmInterface 接口，可直接被仿真框架调用。
    """
    
    def initialize(self, expected_tags: list['Tag']):
        """
        初始化算法状态。
        在每个仿真开始时被框架调用，用于重置所有内部变量。
        """
        # 1. 存储“应有名单”
        self.expected_tags_db = expected_tags
        
        # 2. 算法的核心状态：一个动态变化的、尚未被确认为“在场”的标签列表
        #    初始时，所有标签都未被验证。
        self.unverified_tags = list(self.expected_tags_db)
        
        # 3. 用于存储已被协议确认为“在场”的标签ID集合
        self.found_present_ids = set()
        
        # 4. 用于哈希的随机种子，每轮迭代更新以改变标签到时隙的映射
        self.random_seed = 0
        
        # 5. 根据论文分析，使用最优负载因子来确定每轮的帧大小
        self.optimal_rho = 1.516
        
        # 6. 特殊终止状态标志，用于处理协议末尾没有新标签被发现的情况
        self.final_check_mode = False

    def is_finished(self) -> bool:
        """
        向框架报告算法是否已完成。
        当不再有未验证的标签时，说明所有在场标签都已找到，算法结束。
        """
        return not self.unverified_tags

    def get_results(self) -> tuple[set[str], set[str]]:
        """
        在仿真结束后，向框架返回最终的识别结果。
        """
        all_ids_in_db = {t.id for t in self.expected_tags_db}
        # 核心逻辑：在完整的ID清单中，减去所有被协议确认为“在场”的ID，
        # 剩下的就一定是“丢失”的ID。
        missing_ids = all_ids_in_db - self.found_present_ids
        return self.found_present_ids, missing_ids

    def perform_step(self) -> 'AlgorithmStepResult':
        """
        执行一轮IIP协议的帧通信。这是算法的核心逻辑，会被框架反复调用。
        """
        if self.is_finished():
            return AlgorithmStepResult(0, 0, 0, "finished")

        # 当前待验证的标签数量，即论文中的 N*
        n_star = len(self.unverified_tags)
        
        # 1. 确定本轮帧大小 f
        # f = N* / ρ
        if self.optimal_rho <= 0:
            frame_size = n_star # 避免除零错误
        else:
            frame_size = int(round(n_star / self.optimal_rho))
        
        # 确保帧大小至少为1，避免在剩余少量标签时帧大小为0
        if frame_size == 0:
            frame_size = 1

        # --- 步骤A: 读写器侧的预计算 ---

        # 2. 预计算每个时隙的期望占用情况
        #    算法本身不知道标签是否在场，所以它基于所有“未验证”的标签进行计算。
        expected_slots = [[] for _ in range(frame_size)]
        for tag in self.unverified_tags:
            # 使用Python内置的hash函数模拟论文中的 H(id, r)
            # 这里的哈希函数需要稳定且对于相同的 (id, seed) 给出相同的结果
            slot_index = hash(tag.id + str(self.random_seed)) % frame_size
            expected_slots[slot_index].append(tag)
        
        # 3. 根据期望占用情况，生成 pre-frame vector
        #    如果一个时隙期望有超过1个标签映射过来，则标记为碰撞时隙('1')
        pre_frame_vector = [1 if len(tags_in_slot) > 1 else 0 for tags_in_slot in expected_slots]
        
        # --- 步骤B: 模拟标签侧的响应 ---
        
        # 4. 模拟真实在场的标签进行响应
        actual_responses = [[] for _ in range(frame_size)]
        # 关键：只有真实在场(is_present=True)且未被验证的标签才会参与响应
        present_unverified_tags = [tag for tag in self.unverified_tags if tag.is_present]
        
        tags_that_responded = 0
        for tag in present_unverified_tags:
            slot_index = hash(tag.id + str(self.random_seed)) % frame_size
            should_respond = True
            
            # 如果是碰撞时隙，则有50%概率不响应以试图解决碰撞
            if pre_frame_vector[slot_index] == 1:
                # 使用另一个哈希函数模拟论文中的 H'(id, r)，这里用不同的盐值实现
                if hash(tag.id + str(self.random_seed) + "second_hash") % 2 == 0:
                    should_respond = False
            
            # 论文中提到的特殊终止条件处理
            if self.final_check_mode:
                # 在最终检查模式下，所有剩余标签都必须响应，不再随机跳过
                should_respond = True

            if should_respond:
                actual_responses[slot_index].append(tag)
                tags_that_responded += 1

        # --- 步骤C: 读写器侧处理结果并更新状态 ---

        # 5. 扫描所有时隙，识别所有单例时隙(singleton slot)中的标签
        newly_identified_tags = set()
        for slot in actual_responses:
            if len(slot) == 1:
                identified_tag = slot[0]
                newly_identified_tags.add(identified_tag)

        # 6. 更新算法内部状态
        if newly_identified_tags:
            self.final_check_mode = False # 只要有进展，就退出最终检查模式
            for tag in newly_identified_tags:
                self.found_present_ids.add(tag.id)
            # 从未验证列表中移除已确认在场的标签
            self.unverified_tags = [t for t in self.unverified_tags if t.id not in self.found_present_ids]
        else:
            # 如果本轮没有任何新标签被识别
            if tags_that_responded == 0 and n_star > 0:
                # 触发论文中描述的特殊终止检查逻辑
                if not self.final_check_mode:
                    # 进入最终检查模式，下一轮将强制所有剩余标签响应
                    self.final_check_mode = True
                else:
                    # 如果在最终检查模式下仍然没有任何响应，
                    # 说明所有剩余未验证的标签都确定是丢失的。
                    # 此时可以安全地清空未验证列表，从而结束算法。
                    self.unverified_tags = []
            
        # 7. 计算本轮的开销，并返回给框架
        # 根据论文公式: T_frame = 2 * ceil(f/96) * t_tag + f * t_s
        # 2 * ceil(f/96) * t_tag 是 pre-frame 和 post-frame 向量的传输时间
        # f * t_s 是帧内所有短响应时隙的总时间
        t_tag_slot = self.config.get('T_tag_SLOT_TIME_US', 2400.0)
        t_s_slot = self.config.get('T_s_SLOT_TIME_US', 400.0)
        
        # 时间开销
        time_for_vectors = 2 * math.ceil(frame_size / 96.0) * t_tag_slot
        time_for_slots = frame_size * t_s_slot
        time_delta = time_for_vectors + time_for_slots
        
        # 通信开销
        # 读写器发送pre-frame和post-frame两个向量，总比特数为 2*f
        reader_bits = 2 * frame_size
        # 标签在所有非空时隙中响应了1比特
        tag_bits = tags_that_responded * self.config.get('TAG_SHORT_RESP_BITS', 1)

        # 为下一轮迭代更新随机种子
        self.random_seed += 1
        
        op_desc = f"IIP 帧: N*={n_star}, f={frame_size}, 新识别={len(newly_identified_tags)}"

        # 返回标准化的结果对象
        return AlgorithmStepResult(
            time_delta_us=time_delta,
            reader_bits=reader_bits,
            tag_bits=tag_bits,
            operation_description=op_desc
        )

if __name__ == '__main__':
    print("开始缺失标签识别算法的对比仿真...")

    # --- 1. 定义全局仿真配置 ---
    #    这里的参数参考了论文中的设定，例如 N=50000
    simulation_parameters = {
        'TOTAL_TAGS': 50000,
        'MISSING_RATE': 0.01, # 1% 的标签丢失
        'BINARY_LENGTH': 96,
        'DATA_RATE_BPS': 160000.0,
        
        # 物理层和链路层时间常量 (µs)
        # 严格遵循论文中的时间设定
        'T_s_SLOT_TIME_US': 400.0,    # 0.4ms
        'T_tag_SLOT_TIME_US': 2400.0, # 2.4ms
        
        # 以下参数用于更细粒度的自定义算法，对于IIP和Baseline可能不是必须的，
        # 但框架提供了它们以支持更多类型的算法。
        'T1_RTcal': 25.0,
        'T2_TRcal': 25.0,
        'READER_CMD_BASE_BITS': 37,
        'TAG_SHORT_RESP_BITS': 1,
    }
    # 框架会自动根据BPS计算BITS_PER_MICROSECOND，这里为了清晰也计算一下
    simulation_parameters['BITS_PER_MICROSECOND'] = simulation_parameters['DATA_RATE_BPS'] / 1.0e6

    # --- 2. 运行仿真 ---
    
    # 运行基线轮询算法作为对比基准
    print("\n>>> 开始运行 BaselinePollingAlgo...")
    baseline_results = run_missing_tag_simulation(
        global_config=simulation_parameters,
        algorithm_class=BaselinePollingAlgo,
        algorithm_specific_config={}
    )

    # 运行论文中的IIP算法
    print("\n>>> 开始运行 IIPAlgo...")
    iip_results = run_missing_tag_simulation(
        global_config=simulation_parameters,
        algorithm_class=IIPAlgo,
        algorithm_specific_config={}
    )

    # --- 3. 统一打印和对比结果 ---
    print("\n" + "="*50)
    print("           仿 真 结 果 对 比")
    print("="*50)
    
    print_results(baseline_results)
    print_results(iip_results)
    
    print("\n--- 性能对比总结 ---")
    baseline_time = baseline_results.get('total_protocol_time_us', float('inf'))
    iip_time = iip_results.get('total_protocol_time_us', float('inf'))
    
    if baseline_time > 0 and iip_time > 0:
        time_reduction = (1 - iip_time / baseline_time) * 100
        print(f"时间效率: IIP算法相比基线轮询算法，执行时间减少了 {time_reduction:.2f}%")
    else:
        print("无法计算时间效率对比。")