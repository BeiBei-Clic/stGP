import numpy as np
import pandas as pd
import operator
import random
from deap import base, creator, tools, gp, algorithms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
import os
import json
import functools

# 设置随机种子
random.seed(42)
np.random.seed(42)

# 全局变量，避免重复创建类
_fitness_created = False
_individual_created = False

def protected_div(left, right):
    """保护除法，避免除零错误"""
    if abs(right) < 1e-6:
        return 1.0
    result = left / right
    if math.isnan(result) or math.isinf(result):
        return 1.0
    return result

def protected_sqrt(x):
    """保护平方根，避免负数开方"""
    result = math.sqrt(abs(x))
    if math.isnan(result) or math.isinf(result):
        return 1.0
    return result

def protected_log(x):
    """保护对数，避免负数或零的对数"""
    result = math.log(abs(x) + 1e-6)
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result

def protected_exp(x):
    """保护指数，避免溢出"""
    x = max(min(x, 50), -50)  # 更严格的限制
    result = math.exp(x)
    if math.isnan(result) or math.isinf(result):
        return 1.0
    return result

def protected_pow(left, right):
    """保护幂运算"""
    if abs(left) < 1e-6 and right < 0:
        return 1.0
    if abs(right) > 10:  # 限制指数大小
        right = 10 if right > 0 else -10
    result = abs(left) ** right
    if math.isnan(result) or math.isinf(result):
        return 1.0
    return result

def protected_sin(x):
    """保护正弦函数"""
    result = math.sin(x)
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result

def protected_cos(x):
    """保护余弦函数"""
    result = math.cos(x)
    if math.isnan(result) or math.isinf(result):
        return 1.0
    return result

def load_and_preprocess_data(dataset_name, sample_size=1000):
    """加载并预处理数据"""
    print(f"正在加载数据集: {dataset_name}")
    
    # 构建数据文件路径
    data_path = f'/home/xyh/stGP/dataset/Feynman_with_units/{dataset_name}'
    
    # 读取指定数量的数据
    data = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            values = line.strip().split()
            if len(values) >= 2:  # 至少需要2列数据
                data.append([float(v) for v in values])
    
    data = np.array(data)
    
    # 检查数据维度
    if data.ndim == 1:
        raise ValueError(f"数据集 {dataset_name} 格式错误：只有一行数据")
    
    num_cols = data.shape[1]
    print(f"数据形状: {data.shape}, 列数: {num_cols}")
    
    if num_cols == 2:
        # 只有2列：输入和输出
        X = data[:, :1]  # 第一列作为输入特征
        y = data[:, 1]   # 第二列作为目标值
        # 为了保持一致性，复制第一列作为第二个特征
        X = np.column_stack([X.flatten(), X.flatten()])
    elif num_cols >= 3:
        # 3列或更多：前两列作为输入，最后一列作为输出
        X = data[:, :2]  # 前两列作为输入特征
        y = data[:, -1]  # 最后一列作为目标值
    else:
        raise ValueError(f"数据集 {dataset_name} 格式错误：列数不足")
    
    print(f"处理后数据形状: X={X.shape}, y={y.shape}")
    
    # 按8:2比例分割训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"训练集大小: {X_train_scaled.shape[0]}")
    print(f"测试集大小: {X_test_scaled.shape[0]}")
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y

def setup_gp():
    """设置遗传编程环境"""
    # 创建原语集
    pset = gp.PrimitiveSet("MAIN", 2)  # 2个输入变量
    
    # 添加基本运算符
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(protected_pow, 2)
    pset.addPrimitive(protected_sqrt, 1)
    pset.addPrimitive(protected_log, 1)
    pset.addPrimitive(protected_exp, 1)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(protected_sin, 1)
    pset.addPrimitive(protected_cos, 1)
    
    # 使用functools.partial替代lambda来避免警告
    pset.addEphemeralConstant("rand101", functools.partial(random.uniform, -2, 2))
    
    # 重命名参数
    pset.renameArguments(ARG0='x1', ARG1='x2')
    
    return pset

def create_fitness_and_individual(pset):
    """创建适应度和个体类型"""
    global _fitness_created, _individual_created
    
    # 只在第一次调用时创建类，避免重复创建警告
    if not _fitness_created:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        _fitness_created = True
    
    if not _individual_created:
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        _individual_created = True
    
    return creator.Individual

def setup_toolbox(pset, Individual):
    """设置工具箱"""
    toolbox = base.Toolbox()
    
    # 注册表达式生成器
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
    toolbox.register("individual", tools.initIterate, Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    
    return toolbox

def evaluate_individual(individual, toolbox, X_train, y_train):
    """评估个体的适应度（RMSE）"""
    # 编译个体为可调用函数
    func = toolbox.compile(expr=individual)
    
    # 计算预测值
    predictions = []
    for i in range(len(X_train)):
        try:
            pred = func(X_train[i, 0], X_train[i, 1])
            # 检查结果是否为有效数值
            if math.isnan(pred) or math.isinf(pred):
                pred = 0.0
            predictions.append(pred)
        except:
            predictions.append(0.0)
    
    predictions = np.array(predictions)
    
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_train, predictions))
    
    return rmse,

def run_genetic_programming(dataset_name, sample_size=1000, population_size=200, generations=300, random_seed=42):
    """运行遗传编程算法"""
    print(f"开始遗传编程算法... 数据集: {dataset_name}, 随机种子: {random_seed}")
    
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 加载和预处理数据
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = load_and_preprocess_data(dataset_name, sample_size)
    
    # 设置遗传编程
    pset = setup_gp()
    Individual = create_fitness_and_individual(pset)
    toolbox = setup_toolbox(pset, Individual)
    
    # 注册评估函数
    toolbox.register("evaluate", evaluate_individual, toolbox=toolbox, 
                    X_train=X_train, y_train=y_train)
    
    # 注册遗传算子
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    
    # 限制树的深度
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
    
    # 创建初始种群
    population = toolbox.population(n=population_size)
    
    # 创建名人堂保存最佳个体
    hof = tools.HallOfFame(1)
    
    # 设置统计信息
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    print("开始进化...")
    
    # 运行遗传算法
    population, logbook = algorithms.eaSimple(
        population, toolbox, 
        cxpb=0.5,      # 交叉概率
        mutpb=0.2,     # 变异概率
        ngen=generations,       # 进化代数
        stats=stats,
        halloffame=hof,
        verbose=True
    )
    
    # 获取最佳个体
    best_individual = hof[0]
    best_func = toolbox.compile(expr=best_individual)
    
    print(f"\n最佳个体: {best_individual}")
    print(f"最佳适应度 (训练RMSE): {best_individual.fitness.values[0]:.6f}")
    
    # 在测试集上评估
    test_predictions = []
    for i in range(len(X_test)):
        pred = best_func(X_test[i, 0], X_test[i, 1])
        if math.isnan(pred) or math.isinf(pred):
            pred = 0.0
        test_predictions.append(pred)
    
    test_predictions = np.array(test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    
    print(f"测试集RMSE: {test_rmse:.6f}")
    
    return {
        'best_individual': str(best_individual),
        'train_rmse': best_individual.fitness.values[0],
        'test_rmse': test_rmse,
        'logbook': logbook,
        'random_seed': random_seed
    }

def run_multiple_experiments(dataset_name, sample_size=1000, population_size=200, generations=300, num_runs=10):
    """对单个数据集运行多次实验"""
    print(f"\n开始对数据集 {dataset_name} 进行 {num_runs} 次实验...")
    
    results = []
    for seed in range(num_runs):
        print(f"\n--- 运行 {seed + 1}/{num_runs} (随机种子: {seed}) ---")
        result = run_genetic_programming(dataset_name, sample_size, population_size, generations, seed)
        results.append(result)
    
    # 计算统计信息
    test_rmses = [r['test_rmse'] for r in results]
    test_rmses.sort()
    
    # 计算分位数
    q25_idx = int(0.25 * len(test_rmses))
    q50_idx = int(0.50 * len(test_rmses))
    q75_idx = int(0.75 * len(test_rmses))
    
    # 获取对应分位数的结果
    selected_results = {
        'q25': results[test_rmses.index(test_rmses[q25_idx])],
        'q50': results[test_rmses.index(test_rmses[q50_idx])],
        'q75': results[test_rmses.index(test_rmses[q75_idx])]
    }
    
    # 添加统计信息
    stats = {
        'dataset_name': dataset_name,
        'sample_size': sample_size,
        'population_size': population_size,
        'generations': generations,
        'num_runs': num_runs,
        'test_rmse_stats': {
            'mean': np.mean(test_rmses),
            'std': np.std(test_rmses),
            'min': np.min(test_rmses),
            'max': np.max(test_rmses),
            'q25': test_rmses[q25_idx],
            'q50': test_rmses[q50_idx],
            'q75': test_rmses[q75_idx]
        },
        'selected_results': selected_results
    }
    
    print(f"\n数据集 {dataset_name} 实验完成:")
    print(f"测试RMSE - 平均值: {stats['test_rmse_stats']['mean']:.6f}, 标准差: {stats['test_rmse_stats']['std']:.6f}")
    print(f"分位数 - 25%: {stats['test_rmse_stats']['q25']:.6f}, 50%: {stats['test_rmse_stats']['q50']:.6f}, 75%: {stats['test_rmse_stats']['q75']:.6f}")
    
    return stats

def save_results(results, dataset_name):
    """保存实验结果到文件"""
    os.makedirs('/home/xyh/stGP/results', exist_ok=True)
    
    # 保存详细结果
    filename = f'/home/xyh/stGP/results/{dataset_name}_results.json'
    with open(filename, 'w', encoding='utf-8') as f:
        # 处理logbook对象，转换为可序列化的格式
        results_copy = results.copy()
        for key in ['q25', 'q50', 'q75']:
            if 'logbook' in results_copy['selected_results'][key]:
                logbook = results_copy['selected_results'][key]['logbook']
                # 只保存每一代的最佳适应度（最小值），不再保存generations字段
                results_copy['selected_results'][key]['logbook'] = {
                    'best_fitness': [record['min'] for record in logbook]
                }
        
        json.dump(results_copy, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {filename}")
    print(f"已保存三次实验的每代最佳适应度变化过程 (25%、50%、75%分位数)，不包含generations字段")

def main(sample_size=1000, population_size=200, generations=300):
    """主函数，运行所有数据集的实验"""
    # 指定要运行的数据集
    # datasets = ["I.6.2", "I.6.2a", "I.12.1", "I.12.5", "I.14.4", "I.25.13"]
    datasets=["I.6.2","I.6.2b","I.12.4","I.14.3","I.14.4","I.25.13"]
    
    print("开始批量实验...")
    print(f"数据集: {datasets}")
    print(f"超参数 - 样本数: {sample_size}, 种群大小: {population_size}, 迭代代数: {generations}")
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"处理数据集: {dataset}")
        print(f"{'='*60}")
        
        # 运行多次实验
        results = run_multiple_experiments(
            dataset_name=dataset,
            sample_size=sample_size,
            population_size=population_size,
            generations=generations,
            num_runs=10
        )
        
        # 保存结果
        save_results(results, dataset)
        
        # 存储到总结果中
        all_results[dataset] = results
    
    print(f"\n{'='*60}")
    print("所有实验完成！")
    print(f"{'='*60}")
    
    # 打印总结
    print("\n实验总结:")
    for dataset, results in all_results.items():
        stats = results['test_rmse_stats']
        print(f"{dataset}: 平均RMSE={stats['mean']:.6f} (±{stats['std']:.6f}), "
              f"分位数[{stats['q25']:.6f}, {stats['q50']:.6f}, {stats['q75']:.6f}]")
    
    return all_results

if __name__ == "__main__":
    # 可以通过修改这些参数来调试不同的超参数设置
    results = main(
        sample_size=100,    # 每个数据集使用的样本数
        population_size=200, # 种群个体数
        generations=300      # 迭代代数
    )