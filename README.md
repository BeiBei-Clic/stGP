## 对比试验

```bash
uv run python -u GP.py --dataset dataset/feynman/bonus_with_units --num_seeds 1
```

```bash
uv run python -u GP.py --dataset dataset/feynman/bonus_without_units --num_seeds 10 --max_tree_height 6 --max_tree_size 15 --gpus 0,1,2
```
