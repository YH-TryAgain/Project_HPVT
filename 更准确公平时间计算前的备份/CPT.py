# Paper name : Revisiting_RFID_Missing_Tag_Identification.pdf
import math
import random
from collections import deque

# 假设以下类已在框架中定义
from Framework import *

class _CPTNode:
    """CPT树的内部节点类，用于构建树形结构。"""
    def __init__(self):
        self.is_leaf = False
        # --- 对于内部节点 ---
        self.split_bit_index = -1  # 用于分裂当前节点标签集合的最佳比特位索引
        self.child_0 = None        # 对应比特值为0的分支
        self.child_1 = None        # 对应比特值为1的分支
        # --- 对于叶子节点 ---
        self.tags = []             # 该叶子节点包含的标签列表 (最多2个)
        # --- 用于查询的辅助信息 ---
        self.path_from_root = []   # 从根节点到此节点的路径描述


class CPTAlgo(MissingTagAlgorithmInterface):
    """
    论文 'Revisiting RFID Missing Tag Identification' 中提出的
    碰撞分区树 (Collision-Partition Tree, CPT) 算法的实现。
    """
    
    def initialize(self, expected_tags: list['Tag']):
        """
        初始化算法。此方法在仿真开始时被框架调用，完成CPT算法的所有准备工作。
        """
        self.expected_tags_db = expected_tags
        # !!已修改!!: 分别追踪确认在场和确认丢失的标签集合
        self.found_present_ids = set()
        self.found_missing_ids = set()
        
        self.pseudo_id_map = {}   # 存储 {original_id: pseudo_id}
        self.pseudo_id_len = 0
        
        if not self.expected_tags_db:
            self.query_queue = deque()
            return
            
        # 步骤一：生成伪ID
        self._generate_pseudo_ids()

        # 步骤二：构建CPT树
        self.root = self._build_cpt_recursive(self.expected_tags_db, [])
        
        # 步骤三：准备查询队列
        self.query_queue = deque()
        self._collect_leaf_queries(self.root)

    def is_finished(self) -> bool:
        """当查询队列为空时，代表所有叶子节点都已盘点完毕，算法结束。"""
        return not self.query_queue

    def get_results(self) -> tuple[set[str], set[str]]:
        """
        !!已修改!!: 向框架返回当前已经确认的在场和丢失标签集合。
        这个修改是让时间线统计正常工作的关键。
        """
        return self.found_present_ids, self.found_missing_ids

    def perform_step(self) -> 'AlgorithmStepResult':
        """
        执行一个查询步骤，即处理查询队列中的一个叶子节点。
        """
        if self.is_finished():
            # 在最后一步，将所有剩余未确认的标签都标记为丢失
            all_ids_in_db = {t.id for t in self.expected_tags_db}
            unconfirmed_ids = all_ids_in_db - self.found_present_ids - self.found_missing_ids
            self.found_missing_ids.update(unconfirmed_ids)
            return AlgorithmStepResult(0, 0, 0, "finished")

        # 从队列中取出一个查询任务（叶子节点）
        leaf_node = self.query_queue.popleft()
        
        # --- 模拟读写器查询 ---
        path = leaf_node.path_from_root
        
        # 计算读写器命令的比特数
        bits_for_path_len = math.ceil(math.log2(self.pseudo_id_len)) if self.pseudo_id_len > 1 else 1
        reader_bits = len(path) * (bits_for_path_len + 1)
        
        if len(leaf_node.tags) == 2:
            tag0_pid = self.pseudo_id_map[leaf_node.tags[0].id]
            tag1_pid = self.pseudo_id_map[leaf_node.tags[1].id]
            for i in range(self.pseudo_id_len):
                if tag0_pid[i] != tag1_pid[i]:
                    reader_bits += bits_for_path_len
                    break
        
        # --- 模拟标签响应 ---
        responding_tags = [tag for tag in leaf_node.tags if tag.is_present]
        responding_ids = {t.id for t in responding_tags}
        
        # !!已修改!!: 更新确认在场和确认丢失的集合
        # 1. 响应的标签被确认为在场
        self.found_present_ids.update(responding_ids)
        
        # 2. 在这个叶子中，所有应该响应但没有响应的标签，被确认为丢失
        all_tags_in_leaf_ids = {t.id for t in leaf_node.tags}
        missing_in_this_leaf = all_tags_in_leaf_ids - responding_ids
        self.found_missing_ids.update(missing_in_this_leaf)
            
        # --- 计算开销 ---
        tag_bits = len(responding_tags) * self.config.get('TAG_SHORT_RESP_BITS', 1)
        
        time_reader_tx = reader_bits / self.config['BITS_PER_MICROSECOND']
        time_response_slot = self.config.get('T_l_SLOT_TIME_US', 800.0)
        time_delta = time_reader_tx + time_response_slot

        op_desc = f"CPT 查询: 路径深度={len(path)}, 叶子大小={len(leaf_node.tags)}, 响应数={len(responding_tags)}"
        
        return AlgorithmStepResult(
            time_delta_us=time_delta,
            reader_bits=reader_bits,
            tag_bits=tag_bits,
            operation_description=op_desc
        )

    # --- CPT 算法内部辅助方法 ---

    def _generate_pseudo_ids(self):
        """为所有标签生成唯一的、较短的伪ID。"""
        n = len(self.expected_tags_db)
        if n == 0: return
        
        self.pseudo_id_len = math.ceil(2 * math.log2(n)) if n > 1 else 1
        
        while True:
            self.pseudo_id_map.clear()
            pseudo_id_set = set()
            collision_detected = False
            for tag in self.expected_tags_db:
                pseudo_hash = hash(tag.id + str(random.random()))
                pseudo_id = format(pseudo_hash & ((1 << self.pseudo_id_len) - 1), f'0{self.pseudo_id_len}b')
                
                if pseudo_id in pseudo_id_set:
                    collision_detected = True
                    break
                pseudo_id_set.add(pseudo_id)
                self.pseudo_id_map[tag.id] = pseudo_id
            
            if not collision_detected:
                break
    
    def _build_cpt_recursive(self, tags: list['Tag'], current_path: list) -> _CPTNode:
        """递归地构建CPT树。"""
        node = _CPTNode()
        node.path_from_root = current_path

        if len(tags) <= 2:
            node.is_leaf = True
            node.tags = tags
            return node

        best_split_bit_index = -1
        min_difference = float('inf')

        for bit_idx in range(self.pseudo_id_len):
            count_0 = sum(1 for tag in tags if self.pseudo_id_map[tag.id][bit_idx] == '0')
            count_1 = len(tags) - count_0
            
            difference = abs(count_0 - count_1)
            if difference < min_difference:
                min_difference = difference
                best_split_bit_index = bit_idx
        
        node.split_bit_index = best_split_bit_index
        
        tags_0 = [t for t in tags if self.pseudo_id_map[t.id][best_split_bit_index] == '0']
        tags_1 = [t for t in tags if self.pseudo_id_map[t.id][best_split_bit_index] == '1']
        
        node.child_0 = self._build_cpt_recursive(tags_0, current_path + [(best_split_bit_index, '0')])
        node.child_1 = self._build_cpt_recursive(tags_1, current_path + [(best_split_bit_index, '1')])
        
        return node
    
    def _collect_leaf_queries(self, node: _CPTNode):
        """通过深度优先遍历，将所有叶子节点收集到查询队列中。"""
        if node is None:
            return
        if node.is_leaf:
            if node.tags:
                self.query_queue.append(node)
            return
        
        self._collect_leaf_queries(node.child_0)
        self._collect_leaf_queries(node.child_1)


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
        # 严格遵循论文中的时间设定，为CPT算法增加长时隙 T_l
        'T_s_SLOT_TIME_US': 400.0,    # 0.4ms (短时隙)
        'T_l_SLOT_TIME_US': 800.0,    # 0.8ms (长时隙, CPT需要)
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
    print("\n>>> 开始运行 BaselinePollingAlgo...")
    baseline_results = run_missing_tag_simulation(
        global_config=simulation_parameters,
        algorithm_class=BaselinePollingAlgo,
        algorithm_specific_config={}
    )

    # 运行新实现的CPT算法
    print("\n>>> 开始运行 CPTAlgo...")
    cpt_results = run_missing_tag_simulation(
        global_config=simulation_parameters,
        algorithm_class=CPTAlgo,
        algorithm_specific_config={}
    )

        # --- 3. 统一打印和对比结果 ---
    print("\n" + "="*50)
    print("           仿 真 结 果 对 比")
    print("="*50)
    
    print_results(baseline_results)
    print_results(cpt_results)
    
    print("\n--- 性能对比总结 ---")
    baseline_time = baseline_results.get('total_protocol_time_us', float('inf'))
    cpt_time = cpt_results.get('total_protocol_time_us', float('inf'))

    if baseline_time > 0 and cpt_time > 0:
        time_reduction_cpt = (1 - cpt_time / baseline_time) * 100
        print(f"时间效率: CPT 算法相比基线轮询，执行时间减少了 {time_reduction_cpt:.2f}%")
