import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

class TspSolution:
    def __init__(self, route, fitness):
        self.route = route  
        self.fitness = fitness  
    
    def __lt__(self, other):
        return self.fitness < other.fitness
    
    def __gt__(self, other):
        return self.fitness > other.fitness
    
    def __ge__(self, other):
        return self.fitness >= other.fitness
    
    def __eq__(self, other):
        return self.fitness == other.fitness

def load_tsp_data(file_path):
    """读取TSP问题数据文件，提取城市坐标"""
    header_line = 0
    # 查找城市坐标数据的起始位置
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if line.startswith('NODE_COORD_SECTION'):
                header_line = i + 1
    
    # 读取数据并转换为numpy数组
    df = pd.read_csv(file_path, skiprows=header_line, header=None, sep=' ')
    df = df.dropna()  
    cities_data = df.to_numpy()
    for i in range(len(cities_data)):
        cities_data[i][1] = float(cities_data[i][1])
        cities_data[i][2] = float(cities_data[i][2])
    cities_data = cities_data[:, 1:]  
    return cities_data

def calculate_fitness(cities_data, route):
    """计算路径的适应度值"""
    # 根据路径顺序重新排列城市坐标
    ordered_cities = np.array([cities_data[route[i]] for i in range(len(route))])
    total_distance = 0.0
    for i in range(len(route) - 1):
        total_distance += np.linalg.norm(ordered_cities[i] - ordered_cities[i+1])
    total_distance += np.linalg.norm(ordered_cities[-1] - ordered_cities[0])
    fitness = 1 / (total_distance + 1e-6)
    return fitness

def select_best_solution(candidates):
    sorted_candidates = np.sort(candidates)[::-1]  # 按适应度降序排序
    return sorted_candidates[0]

def tournament_selection(solutions_population, tournament_size):
    """锦标赛选择法:从种群中随机选择几个个体，然后选出其中最好的一个"""
    selected_solutions = np.random.choice(solutions_population, size=tournament_size, replace=False)
    selected_solutions = np.array([TspSolution(solution.route, solution.fitness) 
                                 for i, solution in enumerate(selected_solutions)], 
                                dtype=object)
    return select_best_solution(selected_solutions)

def order_crossover(parent1_route, parent2_route):
    """顺序交叉算子:在保持路径有效性的前提下，结合两个父代路径的特征"""
    i = np.random.randint(0, len(parent1_route))
    j = np.random.randint(0, len(parent1_route))
    while i == j:
        j = np.random.randint(0, len(parent1_route))
    if i > j:
        i, j = j, i  # 确保i是起点，j是终点
    
    city_mapping = {}
    for k in range(i, j+1):
        city_mapping[parent2_route[k]] = parent1_route[k]
    
    child_route = np.zeros((len(parent1_route),), dtype=int)
    for k in range(len(parent1_route)):
        if k < i or k > j:
            child_route[k] = parent1_route[k]
        else:
            child_route[k] = city_mapping[parent2_route[k]]
    return child_route

def apply_crossover(cities_data, solutions_population, population_size=100, crossover_rate=0.8):
    """对整个种群应用交叉操作"""
    new_solutions = np.zeros((population_size,), dtype=object)
    for i in range(population_size):
        if np.random.rand() < crossover_rate:
            # 随机选择两个父代进行交叉
            parent1 = solutions_population[np.random.randint(population_size)]
            parent2 = solutions_population[np.random.randint(population_size)]
            child_route = order_crossover(parent1.route, parent2.route)
            new_solutions[i] = TspSolution(child_route, calculate_fitness(cities_data, child_route))
        else:
            # 不进行交叉，直接复制个体
            new_solutions[i] = solutions_population[np.random.randint(population_size)]
    return new_solutions

def apply_inversion_mutation(route, mutation_rate=0.1):
    """倒置变异:以一定概率随机选择路径的一段，将其倒置"""
    for i in range(len(route)):
        if np.random.rand() < mutation_rate:
            # 随机选择两个点
            i = np.random.randint(0, len(route))
            j = np.random.randint(0, len(route))
            while i == j:
                j = np.random.randint(0, len(route))
            if i > j:
                i, j = j, i 
            route[i:j+1] = route[i:j+1][::-1]
    return route

def apply_mutation(cities_data, solutions_population, population_size=100, mutation_rate=0.1):
    """对整个种群应用变异操作"""
    new_solutions = np.zeros((population_size,), dtype=object)
    for i in range(population_size):
        new_route = apply_inversion_mutation(solutions_population[i].route.copy(), mutation_rate)
        new_solutions[i] = TspSolution(new_route, calculate_fitness(cities_data, new_route))
    return new_solutions

def calculate_adaptive_mutation_rate(iteration, max_iterations):
    """自适应变异率计算:随着迭代进行，变异率逐渐降低，同时定期提高变异率以跳出局部最优"""
    initial_rate = 0.3  
    final_rate = 0.001  
    
    alpha = 4.0  # 退火参数
    progress = iteration / max_iterations
    
    # 每500代提高一次变异率
    if iteration % 500 == 0 and iteration > 0:
        return initial_rate * 0.5  
    return final_rate + (initial_rate - final_rate) * np.exp(-alpha * progress)

def perform_partial_reset(solutions_population, cities_data, reset_percentage=0.3):
    """部分种群重置:淘汰一部分较差的解，引入新的随机解增加多样性"""
    population_size = len(solutions_population)
    reset_count = int(population_size * reset_percentage)
    
    indices = np.argsort([solution.fitness for solution in solutions_population])
    keep_indices = indices[-(population_size-reset_count):]
    
    new_routes = np.array([np.random.permutation(len(cities_data)) for _ in range(reset_count)])
    new_solutions = np.array([TspSolution(route, calculate_fitness(cities_data, route)) 
                            for route in new_routes], dtype=object)
    
    # 组合保留的解和新生成的解
    new_population = np.zeros(population_size, dtype=object)
    new_population[:reset_count] = new_solutions
    new_population[reset_count:] = solutions_population[keep_indices]
    
    return new_population

def calculate_population_diversity(solutions_population):
    """计算种群多样性:使用适应度值的变异系数作为多样性度量"""
    fitnesss = np.array([solution.fitness for solution in solutions_population])
    return np.std(fitnesss) / np.mean(fitnesss)

def genetic_algorithm_for_tsp(file_path, population_size=100, tournament_size=5, 
                             max_generations=10000, elite_count=10):
    """使用遗传算法求解TSP问题的主函数"""

    cities_data = load_tsp_data(file_path)
    
    initial_routes = np.array([np.random.permutation(len(cities_data)) for _ in range(population_size)])
    solutions_population = np.array([TspSolution(route, calculate_fitness(cities_data, route)) 
                                   for route in initial_routes], dtype=object)
    
    best_distances = np.zeros((max_generations,), dtype=float)
    
    for generation in range(max_generations):
        # 保留精英解
        elite_indices = np.argsort([solution.fitness for solution in solutions_population])[-elite_count:]
        elite_solutions = solutions_population[elite_indices].copy()
        
        # 记录当前最佳解对应的距离
        best_distances[generation] = 1 / np.max(solutions_population).fitness
        
        new_population = np.zeros((population_size,), dtype=object)
        for j in range(population_size):
            selected_solution = tournament_selection(solutions_population, tournament_size)
            new_population[j] = TspSolution(selected_solution.route, selected_solution.fitness)
        solutions_population = new_population
        
        solutions_population = apply_crossover(cities_data, solutions_population, population_size)
        
        current_mutation_rate = calculate_adaptive_mutation_rate(generation, max_generations)
        
        solutions_population = apply_mutation(cities_data, solutions_population, 
                                            population_size, current_mutation_rate)
        
        for k, elite in enumerate(elite_solutions):
            solutions_population[k] = elite
            
        if generation % 50 == 0:
            solutions_population = perform_partial_reset(solutions_population, cities_data, reset_percentage=0.3)
            
        if generation % 1000 == 0:
            print(f"第 {generation} 代: 最佳路径长度 = {best_distances[generation]}")
        
    return np.min(best_distances)  # 返回找到的最短距离

best_distance = genetic_algorithm_for_tsp(r"C:\Users\jhinx\Desktop\大学课件\人工智能作业\5-6 搜索算法\qa194.tsp",population_size=100,tournament_size=2,max_generations=30001,elite_count=10)