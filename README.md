# stGP - 标准遗传编程符号回归

基于 DEAP 实现的标准遗传编程（Strongly Typed Genetic Programming）符号回归算法。

## 环境配置

使用 uv 管理虚拟环境：

```bash
uv sync
```

## 对比试验

### 快速测试（单个数据集）

```bash
uv run python -u stgp_sr.py --dataset dataset/feynman/bonus_with_units/test_10.csv --gpus 0 --num_seeds 1 --max_input_points 100 --max_tree_height 5 --population_size 10 --generations 2
```

### 完整实验（批量处理文件夹）

```bash
uv run python -u stgp_sr.py --dataset dataset/feynman/bonus_without_units --gpus 0 --num_seeds 10 --max_input_points 100 --max_tree_height 5

uv run python -u stgp_sr.py --dataset dataset/feynman/bonus_without_units --gpus 0,1,2 --num_seeds 10 --max_input_points 100 --max_tree_height 5 --population_size 200 --generations 300
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集路径（文件夹或单个CSV文件） | 必需 |
| `--gpus` | GPU编号列表，用于决定进程数（如: 0,1,2,3） | `0` |
| `--num_seeds` | 实验重复次数 | `10` |
| `--max_input_points` | 训练数据点数量 | `100` |
| `--max_tree_height` | 表达式树最大高度 | `5` |
| `--population_size` | 种群大小 | `200` |
| `--generations` | 迭代次数 | `300` |
| `--output_dir` | 结果输出目录 | `result` |

## 数据集格式

标准 CSV 格式，第一行为 ground truth 表达式，后续行为空格分隔的数据：

```
arccos((cos(x2)-x1/x0)/(1-x1/x0*cos(x2)))
4.813264961310665 2.2020590253435524 1.0292507372882937 1.4948731649146594
5.869282961709513 1.1524489478734037 2.7161616511646143 2.7911824580112303
...
```

## 结果输出格式

结果保存为 JSON 文件，目录结构与输入数据集保持一致：

```json
{
  "dataset": "test_10",
  "ground_truth": "arccos((cos(x2)-x1/x0)/(1-x1/x0*cos(x2)))",
  "runs": [
    {
      "seed": 0,
      "final_expression": "acos((cos(x_2) - x_1/x_0) / (1.0 - x_1*cos(x_2)/x_0))",
      "train_rmse": 0.015,
      "test_rmse": 0.031,
      "train_r2": 0.998,
      "test_r2": 0.995,
      "evolution_curve": [0.328, 0.096, ...]
    }
  ]
}
```
