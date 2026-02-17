import argparse
import json
import math
import multiprocessing as mp
import operator
import os
import random
import functools
from pathlib import Path

import numpy as np
from deap import base, creator, tools, gp, algorithms
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sympy import simplify, sympify
from sympy.parsing.sympy_parser import parse_expr


# 全局变量，避免重复创建类
_fitness_created = False
_individual_created = False


def protected_div(left, right):
    if abs(right) < 1e-6:
        return 1.0
    result = left / right
    if math.isnan(result) or math.isinf(result):
        return 1.0
    return result


def protected_sqrt(x):
    result = math.sqrt(abs(x))
    if math.isnan(result) or math.isinf(result):
        return 1.0
    return result


def protected_log(x):
    result = math.log(abs(x) + 1e-6)
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result


def protected_exp(x):
    x = max(min(x, 50), -50)
    result = math.exp(x)
    if math.isnan(result) or math.isinf(result):
        return 1.0
    return result


def protected_pow(left, right):
    if abs(left) < 1e-6 and right < 0:
        return 1.0
    if abs(right) > 10:
        right = 10 if right > 0 else -10
    result = abs(left) ** right
    if math.isnan(result) or math.isinf(result):
        return 1.0
    return result


def protected_sin(x):
    result = math.sin(x)
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result


def protected_cos(x):
    result = math.cos(x)
    if math.isnan(result) or math.isinf(result):
        return 1.0
    return result


def load_feynman_csv(csv_path, max_input_points=None):
    """读取标准CSV格式Feynman数据集
    返回: (ground_truth, X, y)
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    ground_truth = lines[0].strip()

    data = []
    for line in lines[1:]:
        line = line.strip()
        if line:
            values = [float(v) for v in line.split()]
            data.append(values)

    data = np.array(data)

    if max_input_points is not None and len(data) > max_input_points:
        indices = np.random.choice(len(data), max_input_points, replace=False)
        data = data[indices]

    X = data[:, :-1]
    y = data[:, -1]

    return ground_truth, X, y


def get_result_path(csv_path, base_result_dir, dataset_dir):
    """计算结果保存路径，保持result/与dataset/结构一致
    例如: dataset/feynman/test_1.csv -> result/feynman/test_1_stgp.json
    """
    csv_path = Path(csv_path).resolve()
    dataset_dir = Path(dataset_dir).resolve()

    # 如果 dataset_dir 是文件，使用其父目录
    base_dir = dataset_dir if dataset_dir.is_dir() else dataset_dir.parent

    # 找到 dataset/ 目录，保持其后所有子目录结构
    parts = csv_path.parts
    try:
        dataset_idx = parts.index('dataset')
        relative_parts = parts[dataset_idx + 1:]  # 跳过 'dataset/'
    except ValueError:
        # 如果没有 dataset 目录，使用相对于 base_dir 的路径
        relative_path = csv_path.relative_to(base_dir)
        relative_parts = relative_path.parts

    # 去掉文件名的扩展名，添加 _stgp 后缀
    file_name = Path(relative_parts[-1]).stem
    result_parts = list(relative_parts[:-1]) + [f"{file_name}_stgp.json"]

    result_file = base_result_dir.joinpath(*result_parts)
    result_file.parent.mkdir(parents=True, exist_ok=True)

    return result_file


def setup_gp(n_variables):
    """设置遗传编程环境，支持可变变量数"""
    pset = gp.PrimitiveSet("MAIN", n_variables)

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

    pset.addEphemeralConstant("rand101", functools.partial(random.uniform, -2, 2))

    for i in range(n_variables):
        pset.renameArguments(**{f"ARG{i}": f"x_{i}"})

    return pset


def create_fitness_and_individual(pset):
    """创建适应度和个体类型"""
    global _fitness_created, _individual_created

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

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
    toolbox.register("individual", tools.initIterate, Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    return toolbox


def evalSymbReg(individual, toolbox, X, y):
    """评估个体适应度（RMSE），支持可变特征数量"""
    func = toolbox.compile(expr=individual)

    n_samples = X.shape[0]
    predictions = []

    for i in range(n_samples):
        args = X[i, :].tolist()
        try:
            pred = func(*args)
            if math.isnan(pred) or math.isinf(pred):
                pred = 0.0
            predictions.append(pred)
        except:
            predictions.append(0.0)

    predictions = np.array(predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))

    return rmse,


def compute_metrics(X, y, individual, toolbox):
    """计算RMSE和R²指标"""
    func = toolbox.compile(expr=individual)

    predictions = []
    for i in range(X.shape[0]):
        args = X[i, :].tolist()
        try:
            pred = func(*args)
            if math.isnan(pred) or math.isinf(pred):
                pred = 0.0
            predictions.append(pred)
        except:
            predictions.append(0.0)

    predictions = np.array(predictions)

    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)

    return rmse, r2


def simplify_expression(expr_str):
    """使用SymPy化简表达式"""
    try:
        expr = sympify(expr_str)
        simplified = simplify(expr)
        return str(simplified)
    except:
        return expr_str


def run_single_experiment(csv_path, seed, max_input_points, max_tree_height,
                          population_size, generations):
    """运行单次实验，返回包含结果的字典"""
    random.seed(seed)
    np.random.seed(seed)

    ground_truth, X, y = load_feynman_csv(csv_path, max_input_points)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    n_variables = X_train.shape[1]
    pset = setup_gp(n_variables)
    Individual = create_fitness_and_individual(pset)
    toolbox = setup_toolbox(pset, Individual)

    toolbox.register("evaluate", evalSymbReg, toolbox=toolbox,
                    X=X_train_scaled, y=y_train_scaled)

    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree_height))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree_height))

    population = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)

    evolution_curve = []

    def evolve_callback(gen):
        if hof:
            evolution_curve.append(hof[0].fitness.values[0])

    population, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=generations,
        stats=stats,
        halloffame=hof,
        verbose=False
    )

    for record in logbook:
        evolution_curve.append(record['min'])

    best_individual = hof[0]
    final_expression = str(best_individual)
    simplified_expression = simplify_expression(final_expression)

    train_rmse, train_r2 = compute_metrics(X_train_scaled, y_train_scaled, best_individual, toolbox)
    test_rmse, test_r2 = compute_metrics(X_test_scaled, y_test_scaled, best_individual, toolbox)

    return {
        'seed': seed,
        'final_expression': simplified_expression,
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'evolution_curve': [float(x) for x in evolution_curve]
    }


def cpu_worker(task_queue, result_queue, worker_id, max_input_points, max_tree_height,
               population_size, generations):
    """CPU工作进程"""
    while True:
        task = task_queue.get()
        if task is None:
            break

        csv_path, seed = task
        dataset_name = Path(csv_path).stem

        try:
            result = run_single_experiment(
                csv_path, seed, max_input_points, max_tree_height,
                population_size, generations
            )
            result_queue.put((dataset_name, csv_path, seed, result, None))
        except Exception as e:
            result_queue.put((dataset_name, csv_path, seed, None, str(e)))


def collect_results(result_queue, total_tasks, output_dir, input_path):
    """收集并保存结果"""
    from collections import defaultdict

    results = defaultdict(lambda: {'ground_truth': None, 'runs': []})
    completed = 0

    while completed < total_tasks:
        dataset_name, csv_path, seed, result, error = result_queue.get()
        completed += 1

        if error:
            print(f"[错误] {Path(csv_path).name} seed={seed}: {error}")
        else:
            ground_truth, _, _ = load_feynman_csv(csv_path)
            results[dataset_name]['ground_truth'] = ground_truth
            results[dataset_name]['runs'].append(result)

            result_file = get_result_path(csv_path, Path(output_dir), Path(input_path))

            output_data = {
                'dataset': dataset_name,
                'ground_truth': ground_truth,
                'runs': results[dataset_name]['runs']
            }

            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"[完成] {dataset_name} seed={seed} RMSE={result['test_rmse']:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='stGP标准符号回归实验')
    parser.add_argument('--dataset', type=str, required=True, help='数据集路径（文件夹或单个CSV文件）')
    parser.add_argument('--gpus', type=str, default='0', help='GPU编号列表，用于决定进程数（如: 0,1,2,3）')
    parser.add_argument('--num_seeds', type=int, default=10, help='实验重复次数')
    parser.add_argument('--max_input_points', type=int, default=100, help='训练数据点数量')
    parser.add_argument('--max_tree_height', type=int, default=5, help='表达式树最大高度')
    parser.add_argument('--population_size', type=int, default=200, help='种群大小')
    parser.add_argument('--generations', type=int, default=300, help='迭代次数')
    parser.add_argument('--output_dir', type=str, default='result', help='结果输出目录')
    args = parser.parse_args()

    input_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir)

    num_processes = len(args.gpus.split(','))

    csv_files = []
    if input_path.is_file() and input_path.suffix == '.csv':
        csv_files = [input_path]
    elif input_path.is_dir():
        csv_files = sorted(input_path.rglob('*.csv'))
    else:
        print(f"错误: 数据集路径 {args.dataset} 不存在或不是CSV文件/文件夹")
        return

    total_tasks = len(csv_files) * args.num_seeds

    print(f"数据集: {args.dataset}")
    print(f"CSV文件数: {len(csv_files)}")
    print(f"每个文件实验次数: {args.num_seeds}")
    print(f"总任务数: {total_tasks}")
    print(f"进程数: {num_processes}")
    print(f"参数: max_input_points={args.max_input_points}, max_tree_height={args.max_tree_height}, "
          f"population_size={args.population_size}, generations={args.generations}")

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    for csv_file in csv_files:
        for seed in range(args.num_seeds):
            task_queue.put((str(csv_file), seed))

    for _ in range(num_processes):
        task_queue.put(None)

    workers = []
    for i in range(num_processes):
        p = mp.Process(
            target=cpu_worker,
            args=(task_queue, result_queue, i, args.max_input_points,
                  args.max_tree_height, args.population_size, args.generations)
        )
        p.start()
        workers.append(p)

    collect_results(result_queue, total_tasks, output_dir, input_path)

    for p in workers:
        p.join()

    print("所有实验完成！")


if __name__ == "__main__":
    main()
