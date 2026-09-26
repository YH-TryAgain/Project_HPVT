# -*- coding: utf-8 -*-

import random
import time

# ==============================================================================
# 0. 仿真配置和常量
# ==============================================================================

# --- 场景配置 ---
DEFAULT_TOTAL_TAGS = 50000        # 数据库中的标签总数 (N)
DEFAULT_MISSING_RATE = 0.01       # 标签丢失率 (α)，例如 1%
DEFAULT_BINARY_LENGTH = 96        # 标签ID长度 (bits)

# --- 物理层和链路层时间常量 (µs) ---
# 这些参数与之前的框架保持一致，用于精确计算时间开销
DEFAULT_DATA_RATE_BPS = 160000.0
DEFAULT_BITS_PER_MICROSECOND = DEFAULT_DATA_RATE_BPS / 1.0e6
DEFAULT_T1_RTcal = 25.0           # 读写器 -> 标签 周转时间
DEFAULT_T2_TRcal = 25.0           # 标签 -> 读写器 周转时间
DEFAULT_READER_CMD_BASE_BITS = 37 # 读写器命令的基础比特数
DEFAULT_TAG_SHORT_RESP_BITS = 1   # 标签短响应的比特数 (例如，1比特表示'我在')

# ==============================================================================
# 1. 通用类定义
# ==============================================================================

class Tag:
    """表示一个RFID标签及其在盘点场景中的状态。"""

    def __init__(self, identityCode: str, is_present: bool = True):
        """
        初始化标签。
        Args:
            identityCode (str): 标签的唯一ID (二进制字符串)。
            is_present (bool): 标签是否真实存在于现场 (Ground Truth)。
                               算法无法直接访问此状态。
        """
        self.id: str = identityCode
        self.is_present: bool = is_present  # 这是标签的真实状态，供仿真器评估使用

        # 算法对标签状态的判断结果，初始为未知
        self.status_by_algo: str = 'UNKNOWN' # 可能的状态: 'UNKNOWN', 'PRESENT', 'MISSING'

    def __repr__(self):
        """返回标签对象的字符串表示，方便调试。"""
        id_short = self.id[-6:] if len(self.id) > 6 else self.id
        ground_truth = "在场" if self.is_present else "丢失"
        return f"Tag(...{id_short}, 真实状态={ground_truth}, 算法判断={self.status_by_algo})"

class AlgorithmStepResult:
    """封装单次算法执行步骤的结果。"""

    def __init__(self, time_delta_us: float = 0.0,
                 reader_bits: float = 0.0,
                 tag_bits: float = 0.0,
                 operation_description: str = 'idle'):
        """
        Args:
            time_delta_us (float): 此步骤消耗的协议时间 (µs)。
            reader_bits (float): 此步骤中读写器发送的总比特数。
            tag_bits (float): 此步骤中标签响应的总比特数。
            operation_description (str): 对此步骤操作的描述，用于调试或统计。
        """
        self.time_delta_us: float = time_delta_us
        self.reader_bits: float = reader_bits
        self.tag_bits: float = tag_bits
        self.operation_description: str = operation_description

# ==============================================================================
# 2. 算法接口定义
# ==============================================================================

class MissingTagAlgorithmInterface:
    """定义所有缺失标签识别算法必须实现的接口。"""

    def __init__(self, config: dict):
        """
        初始化算法。
        Args:
            config (dict): 包含所有仿真配置参数的字典。
        """
        self.config: dict = config
        self.expected_tags_db: list[Tag] = [] # 算法知道的“应有名单”

    def initialize(self, expected_tags: list[Tag]):
        """
        在每次仿真运行开始时调用，用于重置或初始化算法的内部状态。
        Args:
            expected_tags (list[Tag]): 仿真器生成的“应有名单” (包含所有N个标签对象)。
                                      算法可以读取此列表中的所有 `tag.id`。
        """
        self.expected_tags_db = expected_tags
        raise NotImplementedError("算法必须实现 initialize 方法。")

    def perform_step(self) -> AlgorithmStepResult:
        """
        执行算法的一个逻辑步骤（例如一次查询、一个时隙的处理等）。
        这是算法的核心实现。
        Returns:
            AlgorithmStepResult: 包含该步骤结果（耗时、比特数等）的对象。
        """
        raise NotImplementedError("算法必须实现 perform_step 方法。")

    def is_finished(self) -> bool:
        """
        指示算法是否已经完成了它的所有识别任务。
        当此方法返回 True 时，仿真循环将结束。
        Returns:
            bool: 如果算法已完成，则为 True，否则为 False。
        """
        raise NotImplementedError("算法必须实现 is_finished 方法。")

    def get_results(self) -> tuple[set[str], set[str]]:
        """
        在仿真结束后调用，用于获取算法的最终判断结果。
        Returns:
            一个元组 (found_present_ids, found_missing_ids)
            found_present_ids (set[str]): 算法判断为“在场”的标签ID集合。
            found_missing_ids (set[str]): 算法判断为“丢失”的标签ID集合。
        """
        raise NotImplementedError("算法必须实现 get_results 方法。")


# ==============================================================================
# 3. 核心工具函数
# ==============================================================================

def random_id(length: int) -> str:
    """生成指定长度的随机二进制字符串ID。"""
    if length <= 0: return ""
    return ''.join(random.choice('01') for _ in range(length))

def generate_scenario(config: dict) -> list[Tag]:
    """
    根据配置生成仿真场景（完整的标签列表及其真实状态）。
    """
    total_tags = config.get('TOTAL_TAGS', DEFAULT_TOTAL_TAGS)
    missing_rate = config.get('MISSING_RATE', DEFAULT_MISSING_RATE)
    binary_length = config.get('BINARY_LENGTH', DEFAULT_BINARY_LENGTH)

    num_missing = int(total_tags * missing_rate)
    num_present = total_tags - num_missing

    id_set = set()
    while len(id_set) < total_tags:
        id_set.add(random_id(binary_length))

    all_ids = list(id_set)
    random.shuffle(all_ids)

    scenario_tags = []
    # 创建在场的标签
    for i in range(num_present):
        scenario_tags.append(Tag(all_ids[i], is_present=True))
    # 创建丢失的标签
    for i in range(num_present, total_tags):
        scenario_tags.append(Tag(all_ids[i], is_present=False))

    print(f"场景生成完毕: 总标签数={total_tags}, 其中在场={num_present}, 丢失={num_missing}")
    return scenario_tags

# ==============================================================================
# 4. 仿真核心逻辑
# ==============================================================================

def run_missing_tag_simulation(
    global_config: dict,
    algorithm_class,
    algorithm_specific_config: dict
) -> dict:
    """
    标准化的缺失标签识别仿真框架核心函数。
    """
    sim_start_wall_clock = time.time()

    # --- 准备结果字典 ---
    result_dict = {
        'algorithm_name': algorithm_class.__name__,
        'total_protocol_time_us': 0.0,
        'total_reader_bits': 0.0,
        'total_tag_bits': 0.0,
        # 准确性指标
        'true_positives': 0,  # 找到了且真丢失 (TP)
        'false_positives': 0, # 找到了但其实在场 (FP, 误报)
        'false_negatives': 0, # 没找到但其实丢失 (FN, 漏报)
        'true_negatives': 0,  # 没找且确实在场 (TN)
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'miss_rate': 0.0,
        'false_alarm_rate': 0.0,
        'config_summary': {**global_config, **algorithm_specific_config}
    }

    # --- 1. 生成仿真场景 ---
    scenario_tags = generate_scenario(global_config)
    if not scenario_tags:
        print("警告: 未能生成标签场景，仿真提前结束。")
        return result_dict

    # --- 2. 初始化算法 ---
    # 合并配置，传递给算法实例
    effective_algo_config = {**global_config, **algorithm_specific_config}
    algo_instance = algorithm_class(effective_algo_config)
    algo_instance.initialize(scenario_tags)

    print(f"开始仿真 [{algo_instance.__class__.__name__}]...")

    # --- 3. 主仿真循环 ---
    current_protocol_time_us = 0.0
    step_count = 0
    while not algo_instance.is_finished():
        step_result = algo_instance.perform_step()

        if step_result.time_delta_us < 0:
            print(f"错误: 算法返回了无效的负时间增量: {step_result.time_delta_us}µs！停止仿真。")
            break
        
        # 累加时间和开销
        current_protocol_time_us += step_result.time_delta_us
        result_dict['total_reader_bits'] += step_result.reader_bits
        result_dict['total_tag_bits'] += step_result.tag_bits
        step_count += 1
        
        # 防止无限循环的保护机制
        if step_count > global_config['TOTAL_TAGS'] * 100 and global_config['TOTAL_TAGS'] > 0:
             print(f"错误: 仿真步骤过多 ({step_count})，可能存在无限循环。强制终止。")
             break


    result_dict['total_protocol_time_us'] = current_protocol_time_us
    print(f"仿真结束 [{algo_instance.__class__.__name__}]. 协议总耗时: {current_protocol_time_us / 1e6:.4f} s")

    # --- 4. 获取算法结果并进行最终统计 ---
    predicted_present_ids, predicted_missing_ids = algo_instance.get_results()

    ground_truth_present_ids = {t.id for t in scenario_tags if t.is_present}
    ground_truth_missing_ids = {t.id for t in scenario_tags if not t.is_present}

    # 计算 TP, FP, FN, TN
    tp = len(predicted_missing_ids.intersection(ground_truth_missing_ids))
    fp = len(predicted_missing_ids.intersection(ground_truth_present_ids))
    fn = len(ground_truth_missing_ids.difference(predicted_missing_ids))
    tn = len(ground_truth_present_ids.difference(predicted_missing_ids))

    result_dict['true_positives'] = tp
    result_dict['false_positives'] = fp
    result_dict['false_negatives'] = fn
    result_dict['true_negatives'] = tn

    # 计算衍生指标
    if (tp + fp) > 0:
        result_dict['precision'] = tp / (tp + fp)
    if (tp + fn) > 0:
        result_dict['recall'] = tp / (tp + fn)
    if result_dict['precision'] > 0 and result_dict['recall'] > 0:
        result_dict['f1_score'] = 2 * (result_dict['precision'] * result_dict['recall']) / (result_dict['precision'] + result_dict['recall'])
    
    if len(ground_truth_missing_ids) > 0:
        result_dict['miss_rate'] = fn / len(ground_truth_missing_ids) # 漏报率
    if len(predicted_missing_ids) > 0:
        result_dict['false_alarm_rate'] = fp / len(predicted_missing_ids) # 误报率

    result_dict['total_wall_clock_time_s'] = time.time() - sim_start_wall_clock
    return result_dict


# ==============================================================================
# 5. 示例算法实现 (基线轮询算法)
# ==============================================================================

class BaselinePollingAlgo(MissingTagAlgorithmInterface):
    """一个简单的基线算法，逐一轮询数据库中的所有标签ID。"""

    def initialize(self, expected_tags: list[Tag]):
        self.expected_tags_db = expected_tags
        self.tags_to_check = list(self.expected_tags_db) # 创建一个待检查列表的副本
        self.results = {} # 用于存储每个ID的检查结果: 'PRESENT' 或 'MISSING'

    def is_finished(self) -> bool:
        # 如果待检查列表为空，则任务完成
        return not self.tags_to_check

    def get_results(self) -> tuple[set[str], set[str]]:
        present_ids = {tag_id for tag_id, status in self.results.items() if status == 'PRESENT'}
        missing_ids = {tag_id for tag_id, status in self.results.items() if status == 'MISSING'}
        return present_ids, missing_ids

    def perform_step(self) -> AlgorithmStepResult:
        if self.is_finished():
            return AlgorithmStepResult(0, 0, 0, "finished")

        # 从待检查列表中取出一个标签进行检查
        tag_to_poll = self.tags_to_check.pop(0)

        # 模拟一次读写器轮询操作：广播一个完整的标签ID
        reader_bits = self.config['BINARY_LENGTH']
        # 加上命令本身的基础开销
        reader_bits += self.config.get('READER_CMD_BASE_BITS', DEFAULT_READER_CMD_BASE_BITS)
        
        # 计算读写器发送命令的时间
        time_reader_tx = reader_bits / self.config['BITS_PER_MICROSECOND']
        time_turnaround1 = self.config['T1_RTcal']
        
        time_delta = time_reader_tx + time_turnaround1
        tag_bits = 0

        # 检查这个标签是否真实在场，以决定是否有响应
        if tag_to_poll.is_present:
            # 如果在场，标签会响应一个简短信号
            tag_bits = self.config.get('TAG_SHORT_RESP_BITS', DEFAULT_TAG_SHORT_RESP_BITS)
            time_tag_tx = tag_bits / self.config['BITS_PER_MICROSECOND']
            time_turnaround2 = self.config['T2_TRcal']
            time_delta += time_tag_tx + time_turnaround2
            self.results[tag_to_poll.id] = 'PRESENT'
            op_desc = f"轮询 {tag_to_poll.id[-6:]}... 成功响应"
        else:
            # 如果丢失，读写器会等待超时
            # 注意：这里的超时模型可以更精细，但为了简化，我们假设一个固定的等待时间
            time_delta += self.config['T1_RTcal'] # 模拟读写器等待响应的最小时间
            self.results[tag_to_poll.id] = 'MISSING'
            op_desc = f"轮询 {tag_to_poll.id[-6:]}... 无响应 (丢失)"

        return AlgorithmStepResult(
            time_delta_us=time_delta,
            reader_bits=reader_bits,
            tag_bits=tag_bits,
            operation_description=op_desc
        )
def print_results(results: dict):
    """
    一个辅助函数，用于格式化并打印单次仿真的结果。
    """
    if not results:
        print("仿真未能返回有效结果。")
        return
        
    print(f"\n--- 结果汇总: {results.get('algorithm_name', 'N/A')} ---")
    print("-" * 35)
    print(f"协议总耗时: {results.get('total_protocol_time_us', 0) / 1e6:.4f} 秒")
    total_bits = results.get('total_reader_bits', 0) + results.get('total_tag_bits', 0)
    print(f"通信总开销: {total_bits:.0f} bits (读写器: {results.get('total_reader_bits', 0):.0f}, 标签: {results.get('total_tag_bits', 0):.0f})")
    print("-" * 35)
    print("准确性评估 (针对'丢失'标签的识别):")
    print(f"  - TP (正确找到丢失): {results.get('true_positives', 0)}")
    print(f"  - FP (将在场误报为丢失): {results.get('false_positives', 0)}")
    print(f"  - FN (将丢失漏报为在场): {results.get('false_negatives', 0)}")
    print(f"  - TN (正确判断在场): {results.get('true_negatives', 0)}")
    print("-" * 35)
    print(f"精确率 (Precision): {results.get('precision', 0):.4f}")
    print(f"召回率 (Recall): {results.get('recall', 0):.4f}")
    print(f"F1 分数: {results.get('f1_score', 0):.4f}")
    print("-" * 35)
    print(f"仿真墙上时钟耗时: {results.get('total_wall_clock_time_s', 0):.3f} 秒")
