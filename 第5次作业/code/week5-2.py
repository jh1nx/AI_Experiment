import numpy as np
import heapq
import time
from numba import njit

@njit
def find_zero_position(board: np.ndarray, value: int) -> int:
    """查找空格在数组中的位置"""
    indices = np.where(board == value)[0]
    return indices[0]

# def calculate_manhattan_distance(board: np.ndarray, target: np.ndarray) -> int:
#     """计算曼哈顿距离"""
#     target_positions = {}
#     for i in range(16):
#         target_positions[target[i]] = i
    
#     distance = 0
#     for i in range(16):
#         tile_value = board[i]
#         if tile_value == 0:
#             continue
#         correct_pos = target_positions[tile_value]
#         current_row, current_col = i // 4, i % 4
#         correct_row, correct_col = correct_pos // 4, correct_pos % 4
#         distance += abs(current_row - correct_row) + abs(current_col - correct_col)
    
#     return distance

# def calculate_euclidean_distance(board: np.ndarray, target: np.ndarray) -> float:
#     """计算欧几里得距离"""
#     target_positions = {}
#     for i in range(16):
#         target_positions[target[i]] = i
    
#     distance = 0
#     for i in range(16):
#         tile_value = board[i]
#         if tile_value == 0:
#             continue
#         correct_pos = target_positions[tile_value]
#         current_row, current_col = i // 4, i % 4
#         correct_row, correct_col = correct_pos // 4, correct_pos % 4
#         distance += np.sqrt((current_row - correct_row)**2 + (current_col - correct_col)**2)
    
#     return distance

@njit
def calculate_manhattan_distance(board, target_positions):
    """计算曼哈顿距离(使用NumPy加速)"""
    distance = 0
    for i in range(16):
        tile_value = board[i]
        if tile_value == 0:
            continue
        target_pos = target_positions[tile_value]
        current_row, current_col = i // 4, i % 4
        target_row, target_col = target_pos // 4, target_pos % 4
        distance += abs(current_row - target_row) + abs(current_col - target_col)
    
    return distance

@njit
def calculate_manhattan_with_conflicts(current_board, target_board, target_positions):
    """计算带冲突的曼哈顿距离"""
    base_distance = calculate_manhattan_distance(current_board, target_positions)
    conflict_penalty = 0
    
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            tile_value = current_board[idx]
            if tile_value == 0:
                continue
            
            target_pos = target_positions[tile_value]
            target_row, target_col = target_pos // 4, target_pos % 4
            
            # 检查行冲突
            if i == target_row:
                for k in range(j+1, 4):
                    idx2 = i * 4 + k
                    tile_value2 = current_board[idx2]
                    if tile_value2 != 0:
                        if tile_value2 < 16:  # 防止数组越界
                            target_pos2 = target_positions[tile_value2]
                            target_row2, target_col2 = target_pos2 // 4, target_pos2 % 4
                            if target_row2 == i and target_col2 < target_col:
                                conflict_penalty += 2
            
            # 检查列冲突
            if j == target_col:
                for k in range(i+1, 4):
                    idx2 = k * 4 + j
                    tile_value2 = current_board[idx2]
                    if tile_value2 != 0:
                        if tile_value2 < 16:  # 防止数组越界
                            target_pos2 = target_positions[tile_value2]
                            target_row2, target_col2 = target_pos2 // 4, target_pos2 % 4
                            if target_col2 == j and target_row2 < target_row:
                                conflict_penalty += 2
    
    return base_distance + conflict_penalty


class PuzzleNode:
    __slots__ = ['board', 'parent_node', 'parent', 'g', 'h', 'f', '_hash', 'empty_pos']   # 使用__slots__来优化内存使用
    
    def __init__(self, board, parent_node=None, parent=None, empty_pos=None):
        self.board = board
        self.parent_node = parent_node
        self.parent = parent
        self.g = 0  # 从起始节点到当前节点的实际代价
        self.h = 0  # 启发式估计值
        self.f = 0  # f = g + h
        self._hash = self._calculate_hash()
        if empty_pos is not None:
            self.empty_pos = empty_pos
        else:
            self.empty_pos = find_zero_position(self.board, 0)
    
    def _calculate_hash(self):  
        hash_value = 0
        for i in range(16):
            hash_value |= int(self.board[i]) << (i * 4)
        return hash_value
    
    def __lt__(self, other):
        if self.f == other.f:
            return self.h < other.h
        return self.f < other.f
    
    def __hash__(self):
        return self._hash
    
    def __eq__(self, other):
        if not isinstance(other, PuzzleNode):
            return False
        return self._hash == other._hash and np.array_equal(self.board, other.board)
    
    def get_neighbors(self):
        """获取所有可能的后继节点"""
        neighbor_nodes = []
        row, col = self.empty_pos // 4, self.empty_pos % 4
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in directions:
            new_row, new_col = row + dx, col + dy
            
            if not (0 <= new_row < 4 and 0 <= new_col < 4):
                continue
            
            new_pos = new_row * 4 + new_col
            new_board = self.board.copy()
            tile_value = new_board[new_pos]
            new_board[self.empty_pos] = tile_value
            new_board[new_pos] = 0
            neighbor_nodes.append(PuzzleNode(new_board, tile_value, self, new_pos))
        
        return neighbor_nodes

class AStarSolver:
    def __init__(self, initial_state, heuristic_func, max_depth=70):
        self.initial_state = np.array(initial_state, dtype=np.int32).flatten()
        self.goal_state = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0], dtype=np.int8)
        self.heuristic_func = heuristic_func
        self.max_depth = max_depth
        self.goal_positions = np.zeros(16, dtype=np.int32)
        for i in range(16):
            self.goal_positions[self.goal_state[i]] = i
    
    def solve(self):
        """A*搜索算法实现"""
        open_list = []
        open_dict = {}
        closed_set = set()
        goal_hash = hash(PuzzleNode(self.goal_state))

        heuristic = lambda state: calculate_manhattan_with_conflicts(state, self.goal_state, self.goal_positions)

        start_node = PuzzleNode(self.initial_state, None, None)
        start_node.h = heuristic(self.initial_state)
        start_node.f = start_node.h
        
        node_hash = hash(start_node)
        heapq.heappush(open_list, start_node)
        open_dict[node_hash] = start_node
        
        nodes_expanded = 0
        max_queue_size = 1
        
        while open_list:
            current = heapq.heappop(open_list)
            current_hash = hash(current)
            
            if current_hash in closed_set:
                continue
                
            if current_hash in open_dict:
                del open_dict[current_hash]

            if current_hash == goal_hash:
                solution_path = []
                node_ptr = current
                while node_ptr and node_ptr.parent_node is not None:
                    solution_path.append(node_ptr.parent_node)
                    node_ptr = node_ptr.parent
                return solution_path[::-1]
            
            closed_set.add(current_hash)
            nodes_expanded += 1

            if current.g >= self.max_depth:
                continue

            for neighbor in current.get_neighbors():
                neighbor_hash = hash(neighbor)

                if neighbor_hash in closed_set:
                    continue

                new_g = current.g + 1
                
                if new_g > self.max_depth:
                    continue

                if neighbor_hash in open_dict and open_dict[neighbor_hash].g <= new_g:
                    continue

                neighbor.g = new_g
                neighbor.h = heuristic(neighbor.board)
                neighbor.f = neighbor.g + neighbor.h
                
                open_dict[neighbor_hash] = neighbor
                heapq.heappush(open_list, neighbor)
                
                if len(open_list) > max_queue_size:
                    max_queue_size = len(open_list)

    def __call__(self):
        return self.solve()

if __name__ == "__main__":
    start_time = time.time()
    
    initial_puzzle = np.array([0,5,15,14,7,9,6,13,1,2,12,10,8,11,4,3], dtype=np.int8)
    
    # 预热JIT编译
    calculate_manhattan_distance(initial_puzzle, np.zeros(16, dtype=np.int8))
    
    print("开始进行搜索")
    
    solver = AStarSolver(initial_puzzle, 70)
    
    solution = solver()
    print(f"成功解决15puzzle问题，需要的步数为: {len(solution)}")
    print(solution)
    
    end_time = time.time()
    print(f"执行时间: {end_time - start_time:.2f}秒")