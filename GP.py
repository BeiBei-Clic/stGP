"""符号回归标准实验脚本：使用标准遗传编程算法在标准 CSV 格式数据集上进行评估"""
import random
import operator
import argparse
import json
import time
from pathlib import Path
from multiprocessing import get_context
from tqdm import tqdm
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from deap import gp, tools, base, creator


def get_pset(n_variables: int, include_extra_unary: bool = True) -> gp.PrimitiveSet:
    """创建原始集合（Primitive Set）

    Args:
        n_variables: 输入变量数量
        include_extra_unary: 是否包含额外的一元运算符

    Returns:
        配置好的原始集合
    """
    pset = gp.PrimitiveSet("MAIN", arity=n_variables)
    pset.renameArguments(**{f"ARG{i}": f"x{i}" for i in range(n_variables)})

    # 基本算术运算
    pset.addPrimitive(operator.add, arity=2, name="add")
    pset.addPrimitive(operator.sub, arity=2, name="sub")
    pset.addPrimitive(operator.mul, arity=2, name="mul")
    pset.addPrimitive(operator.neg, arity=1, name="neg")
    pset.addPrimitive(operator.abs, arity=1, name="abs")

    # 三角函数
    pset.addPrimitive(np.sin, arity=1, name="sin")
    pset.addPrimitive(np.cos, arity=1, name="cos")
    pset.addPrimitive(np.tan, arity=1, name="tan")

    # 反三角函数
    pset.addPrimitive(np.arcsin, arity=1, name="asin")
    pset.addPrimitive(np.arccos, arity=1, name="acos")
    pset.addPrimitive(np.arctan, arity=1, name="atan")

    # 双曲函数
    pset.addPrimitive(np.sinh, arity=1, name="sinh")
    pset.addPrimitive(np.cosh, arity=1, name="cosh")

    # 指数和对数
    pset.addPrimitive(np.exp, arity=1, name="exp")
    pset.addPrimitive(np.log, arity=1, name="log")
    pset.addPrimitive(np.log10, arity=1, name="log10")

    # 幂运算和平方根
    pset.addPrimitive(np.sqrt, arity=1, name="sqrt")

    # 常数
    pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))

    # 保护性除法
    def protected_div(left, right):
        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.divide(left, right)
            if isinstance(x, np.ndarray):
                x[np.isinf(x)] = 1
                x[np.isnan(x)] = 1
            elif np.isinf(x) or np.isnan(x):
                x = 1
        return x

    # 保护性幂运算
    def protected_pow(base, exp):
        with np.errstate(invalid='ignore'):
            result = np.power(base, exp)
            if isinstance(result, np.ndarray):
                result[np.isnan(result)] = 1
                result[np.isinf(result)] = 1
            elif np.isnan(result) or np.isinf(result):
                result = 1
        return result

    pset.addPrimitive(protected_div, arity=2, name="div")
    pset.addPrimitive(protected_pow, arity=2, name="pow")

    if include_extra_unary:
        pset.addPrimitive(np.square, arity=1, name="square")

    return pset


def setup_creator():
    """设置 DEAP creator（仅在首次调用时执行）"""
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    return creator


def eval_symb_reg(individual, pset, x_data, y_data):
    """评估符号回归表达式

    Args:
        individual: 个体（表达式树）
        pset: 原始集合
        x_data: 输入数据
        y_data: 目标值

    Returns:
        适应度值元组（MSE）
    """
    func = gp.compile(expr=individual, pset=pset)
    predictions = []
    for x in x_data:
        try:
            pred = func(*x)
            if np.isnan(pred) or np.isinf(pred):
                return (1e20,)
            predictions.append(pred)
        except:
            return (1e20,)

    mse = np.mean((np.array(predictions) - np.array(y_data).flatten()) ** 2)
    return (mse,)


def simplify_expression(individual):
    """使用 sympy 简化表达式"""
    expr_str = str(individual)
    try:
        from sympy import sympify, simplify
        sympy_expr = sympify(expr_str)
        return str(simplify(sympy_expr))
    except:
        return expr_str


class StandardSRDataset:
    """标准符号回归数据集（CSV格式）

    数据格式:
    - 第一行: 真实表达式字符串
    - 第2行开始: 样本数据，前n列是输入特征，最后一列是目标值
    """

    def __init__(self, csv_path: str, max_samples: int = 100, seed: int = 42):
        self.csv_path = csv_path
        self.max_samples = max_samples
        self.seed = seed

        with open(csv_path, 'r') as f:
            self.true_expr = f.readline().strip()

        data = np.loadtxt(csv_path, skiprows=1)

        X_full = data[:, :-1]
        y_full = data[:, -1]

        n_samples = len(y_full)
        if n_samples > max_samples:
            rng = np.random.RandomState(seed)
            indices = rng.choice(n_samples, size=max_samples, replace=False)
            self.X = X_full[indices]
            self.y = y_full[indices]
        else:
            self.X = X_full
            self.y = y_full

        self.n_features = self.X.shape[1]


class StandardGP:
    """标准遗传编程算法（支持进化曲线记录）"""

    def __init__(
        self,
        n_variables: int,
        config: dict = None,
    ):
        default_config = {
            "population_size": 200,
            "generations": 300,
            "crossover_prob": 0.7,
            "mutation_prob": 0.3,
            "tournament_size": 3,
            "max_tree_height": 6,
            "max_length": 20,
            "min_init_height": 1,
            "max_init_height": 4,
        }
        self.config = {**default_config, **(config or {})}
        self.n_variables = n_variables
        self.evolution_history = []
        self._max_len_limit = self.config["max_length"]

        self.pset = get_pset(n_variables, include_extra_unary=False)
        self.creator = setup_creator()
        self.toolbox = self._setup_toolbox()

    def _setup_toolbox(self):
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset,
                        min_=self.config["min_init_height"], max_=self.config["max_init_height"])
        toolbox.register("individual", tools.initIterate, self.creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=self.pset)
        toolbox.register("select", tools.selTournament, tournsize=self.config["tournament_size"])
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("evaluate", eval_symb_reg, pset=self.pset, x_data=None, y_data=None)

        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)

        toolbox.decorate("mate", gp.staticLimit(key=len, max_value=self._max_len_limit))
        toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=self._max_len_limit))
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.config["max_tree_height"]))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.config["max_tree_height"]))

        return toolbox

    def _evaluate_population(self, population):
        """评估种群中适应度无效的个体"""
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

    def _record_generation_stats(self, generation, mstats, population, verbose):
        """记录并可选地打印某一代的统计信息"""
        record = mstats.compile(population)
        self.evolution_history.append(float(record['fitness']['min']))
        if verbose:
            print(f"gen {generation}: fitness min={record['fitness']['min']:.4f}")

    def fit(self, X, y, verbose=True):

        x_data = [x for x in X]
        y_data = [[float(np.asarray(yi).reshape(-1)[0])] for yi in y]
        self._X_data = x_data
        self._y_data = y_data

        self.toolbox.unregister("evaluate")
        self.toolbox.register("evaluate", eval_symb_reg, pset=self.pset, x_data=x_data, y_data=y_data)

        population = self.toolbox.population(n=self.config["population_size"])
        hof = tools.HallOfFame(1)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        start_time = time.time()
        self.evolution_history = []

        self._evaluate_population(population)
        self._record_generation_stats(0, mstats, population, verbose)

        for g in range(1, self.config["generations"] + 1):
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            for i in range(1, len(offspring), 2):
                if random.random() < self.config["crossover_prob"]:
                    offspring[i - 1], offspring[i] = self.toolbox.mate(offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values, offspring[i].fitness.values

            for i in range(len(offspring)):
                if random.random() < self.config["mutation_prob"]:
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            self._evaluate_population(offspring)

            population[:] = offspring
            hof.update(population)

            self._record_generation_stats(g, mstats, population, verbose)

        if verbose:
            elapsed = time.time() - start_time
            print(f"GP 完成！耗时: {elapsed:.2f}秒, 最优: {hof[0]} (高度{len(hof[0])})")

        return hof[0]

    def evaluate(self, individual, X, y):
        from sklearn.metrics import r2_score
        func = self.toolbox.compile(expr=individual)
        y_pred = np.array([func(*x[:self.n_variables]) for x in X])
        valid_mask = ~np.isnan(y_pred)
        if valid_mask.sum() == 0:
            return -np.inf, np.inf
        y_valid = np.asarray(y[valid_mask]).flatten()
        y_pred_valid = y_pred[valid_mask]
        r2 = r2_score(y_valid, y_pred_valid)
        rmse = np.sqrt(np.mean((y_valid - y_pred_valid) ** 2))
        return r2, rmse


def run_single_experiment(
    csv_path: str,
    seed: int,
    population_size: int = 200,
    generations: int = 300,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.3,
    max_tree_height: int = 6,
    max_length: int = 20,
    min_init_height: int = 1,
    max_init_height: int = 4,
    tournament_size: int = 3,
    verbose: bool = False,
):
    """运行单次实验

    Args:
        csv_path: CSV 文件路径
        seed: 随机种子
        verbose: 是否打印详细信息

    Returns:
        dict: 实验结果
    """
    random.seed(seed)
    np.random.seed(seed)

    dataset = StandardSRDataset(csv_path)
    X, y = dataset.X, dataset.y
    true_expr = dataset.true_expr
    n_variables = dataset.n_features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    config = {
        "population_size": population_size,
        "generations": generations,
        "crossover_prob": crossover_prob,
        "mutation_prob": mutation_prob,
        "tournament_size": tournament_size,
        "max_tree_height": max_tree_height,
        "max_length": max_length,
        "min_init_height": min_init_height,
        "max_init_height": max_init_height,
    }

    gp_model = StandardGP(
        n_variables=n_variables,
        config=config,
    )

    if verbose:
        print(f"开始实验 (seed={seed})...")

    best_individual = gp_model.fit(X_train, y_train, verbose=verbose)

    train_r2, train_rmse = gp_model.evaluate(best_individual, X_train, y_train)
    test_r2, test_rmse = gp_model.evaluate(best_individual, X_test, y_test)

    simplified_expr = simplify_expression(best_individual)

    return {
        "seed": seed,
        "final_expression": str(simplified_expr),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "evolution_curve": gp_model.evolution_history,
    }


def worker_init():
    """初始化工作进程"""
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def worker_run_experiment(args):
    """工作进程：运行单个 CSV 文件的所有实验"""
    (csv_path, n_runs, population_size, generations, crossover_prob,
     mutation_prob, max_tree_height, max_length,
     min_init_height, max_init_height, tournament_size, gpu_id) = args

    # GPU 绑定（如果指定）
    if gpu_id is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    dataset = StandardSRDataset(csv_path)
    true_expr = dataset.true_expr

    results = {
        "dataset": Path(csv_path).stem,
        "ground_truth": true_expr,
        "runs": [],
    }

    for seed in range(n_runs):
        random.seed(seed)
        np.random.seed(seed)

        result = run_single_experiment(
            csv_path=csv_path,
            seed=seed,
            population_size=population_size,
            generations=generations,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            max_tree_height=max_tree_height,
            max_length=max_length,
            min_init_height=min_init_height,
            max_init_height=max_init_height,
            tournament_size=tournament_size,
            verbose=False,
        )
        results["runs"].append(result)

    return csv_path, results


def get_result_path(csv_path: str, base_result_dir: Path, input_path: Path) -> Path:
    csv_path = Path(csv_path)
    # 查找数据集根目录（dataset/）
    for parent in [csv_path] + list(csv_path.parents):
        if 'dataset' in parent.parts:
            idx = parent.parts.index('dataset')
            anchor_path = Path(*parent.parts[:idx+1])
            break
    else:
        anchor_path = input_path.parent
    relative_path = csv_path.relative_to(anchor_path)
    # 添加算法后缀 _gp
    result_name = relative_path.stem + "_gp" + relative_path.suffix
    return base_result_dir / relative_path.with_name(result_name).with_suffix(".json")


def run_batch_eval(
    data_path: str,
    n_runs: int = 10,
    population_size: int = 200,
    generations: int = 300,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.3,
    max_tree_height: int = 6,
    max_length: int = 20,
    min_init_height: int = 1,
    max_init_height: int = 4,
    tournament_size: int = 3,
    gpus: str = None,
):
    """批量运行 GP 评估

    Args:
        data_path: CSV 文件路径或文件夹路径
        n_runs: 每个文件的重复实验次数
        gpus: GPU ID 列表，逗号分隔（如 "0,1,2,3"）
    """
    data_path = Path(data_path)
    result_dir = Path("result")

    if data_path.is_file() and data_path.suffix == ".csv":
        csv_files = [data_path]
    elif data_path.is_dir():
        csv_files = list(data_path.glob("**/*.csv"))
    else:
        raise ValueError(f"无效的路径: {data_path}")

    print(f"找到 {len(csv_files)} 个 CSV 文件")

    # 解析 GPU 列表
    gpu_list = None
    if gpus is not None:
        gpu_list = [int(x.strip()) for x in gpus.split(",")]
        num_procs = len(gpu_list)
    else:
        num_procs = 1

    print(f"使用 {num_procs} 个进程")
    if gpu_list:
        print(f"GPU 列表: {gpu_list}")
    print(f"每个文件重复 {n_runs} 次实验")

    worker_args = []
    for i, csv_path in enumerate(csv_files):
        gpu_id = gpu_list[i % len(gpu_list)] if gpu_list else None
        worker_args.append(
            (
                str(csv_path), n_runs, population_size, generations, crossover_prob,
                mutation_prob, max_tree_height, max_length,
                min_init_height, max_init_height, tournament_size, gpu_id
            )
        )

    results = {}
    mp_ctx = get_context("spawn")
    with mp_ctx.Pool(processes=num_procs, initializer=worker_init) as pool:
        for csv_path, result in tqdm(
            pool.imap_unordered(worker_run_experiment, worker_args),
            total=len(csv_files),
            desc="评估进度"
        ):
            results[csv_path] = result

            # 保存结果
            result_path = get_result_path(csv_path, result_dir, data_path)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {result_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="符号回归标准实验脚本 - 使用标准遗传编程算法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 评估单个 CSV 文件
  python GP.py --dataset data/feynman/bonus_without_units/test_1.csv

  # 评估文件夹中的所有 CSV 文件
  python GP.py --dataset data/feynman/bonus_without_units/

  # 指定重复实验次数和最大树高度
  python GP.py --dataset data/feynman/bonus_without_units/ --num_seeds 5 --max_tree_height 6

  # 使用多 GPU 并行
  python GP.py --dataset data/feynman/bonus_without_units/ --gpus 0,1,2,3
        """
    )

    parser.add_argument("--dataset", type=str, required=True, help="CSV 文件路径或文件夹路径")
    parser.add_argument("--num_seeds", type=int, default=10, help="重复实验次数（默认：10）")
    parser.add_argument("--max_tree_height", type=int, default=6, help="最大树高度（默认：6）")
    parser.add_argument("--max_tree_size", type=int, default=20, help="最大表达式长度/节点数（默认：20）")
    parser.add_argument("--population_size", type=int, default=200, help="种群大小（默认：200）")
    parser.add_argument("--num_generations", type=int, default=300, help="进化代数（默认：300）")
    parser.add_argument("--crossover_prob", type=float, default=0.7, help="交叉概率（默认：0.7）")
    parser.add_argument("--mutation_prob", type=float, default=0.3, help="变异概率（默认：0.3）")
    parser.add_argument("--gpus", type=str, default=None, help="GPU ID 列表，逗号分隔（如 \"0,1,2,3\"）")

    args = parser.parse_args()

    run_batch_eval(
        data_path=args.dataset,
        n_runs=args.num_seeds,
        population_size=args.population_size,
        generations=args.num_generations,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        max_tree_height=args.max_tree_height,
        max_length=args.max_tree_size,
        gpus=args.gpus,
    )
