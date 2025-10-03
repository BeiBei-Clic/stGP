import numpy as np
import pandas as pd
import operator
import random
from deap import base, creator, tools, gp, algorithms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

# 设置随机种子
random.seed(42)
np.random.seed(42)

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

def load_and_preprocess_data():
    """加载并预处理数据"""
    print("正在加载数据...")
    
    # 读取前1000行数据
    data = []
    with open('/home/xyh/stGP/dataset/Feynman_with_units/I.6.2', 'r') as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            values = line.strip().split()
            if len(values) == 3:
                data.append([float(v) for v in values])
    
    data = np.array(data)
    X = data[:, :2]  # 前两列作为输入特征
    y = data[:, 2]   # 第三列作为目标值
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    
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
    
    # 添加常数
    pset.addEphemeralConstant("rand101", lambda: random.uniform(-2, 2))
    
    # 重命名参数
    pset.renameArguments(ARG0='x1', ARG1='x2')
    
    return pset

def create_fitness_and_individual(pset):
    """创建适应度和个体类型"""
    # 创建适应度类（最小化RMSE）
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    
    # 创建个体类
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    
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

def run_genetic_programming():
    """运行遗传编程算法"""
    print("开始遗传编程算法...")
    
    # 加载和预处理数据
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = load_and_preprocess_data()
    
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
    population = toolbox.population(n=200)
    
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
        ngen=300,       # 进化代数
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
        try:
            pred = best_func(X_test[i, 0], X_test[i, 1])
            if math.isnan(pred) or math.isinf(pred):
                pred = 0.0
            test_predictions.append(pred)
        except:
            test_predictions.append(0.0)
    
    test_predictions = np.array(test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    
    print(f"测试集RMSE: {test_rmse:.6f}")
    
    return best_individual, best_func, logbook

if __name__ == "__main__":
    best_individual, best_func, logbook = run_genetic_programming()