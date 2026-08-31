# GRPO Math Reasoning

基于 `Qwen2.5-0.5B-Instruct`、GSM8K 和 TRL 的数学推理 GRPO 强化学习项目。

本项目使用规则奖励评价答案正确性与 XML 输出格式，并使用 LoRA 降低 GRPO 训练的显存需求。

## 项目流程

1. 处理 GSM8K 数据集，生成训练集、验证集和测试集。
2. 加载本地 Qwen2.5-0.5B-Instruct 基座模型。
3. 每道题生成多个回答，并使用规则奖励函数打分。
4. 使用 GRPO 计算组内相对优势并更新 LoRA 参数。
5. 保存训练完成的 LoRA 适配器。

## 文件结构

```text
src/
├── prepare_data.py   # 下载、划分并转换 GSM8K
├── rewards.py        # 正确性与 XML 格式奖励
└── train_grpo.py     # 加载模型并执行 LoRA-GRPO 训练
requirements.txt
```

模型权重、处理后的数据和训练输出不会提交到 GitHub，需要在本地生成。

## 环境

- Linux / WSL2
- Python 3.11
- NVIDIA GPU（支持 BF16）
- PyTorch 2.10
- Transformers 5.3
- TRL 0.27.2

安装依赖：

```bash
pip install -r requirements.txt
```

请先将 `Qwen2.5-0.5B-Instruct` 放到：

```text
models/Qwen2.5-0.5B-Instruct
```

## 运行

处理数据：

```bash
python ./src/prepare_data.py
```

执行默认的短程 GRPO 训练：

```bash
python ./src/train_grpo.py
```

使用全部训练集并增加训练步数：

```bash
python ./src/train_grpo.py --train-samples 0 --max-steps 500
```

训练后的 LoRA 默认保存在：

```text
outputs/qwen2.5-0.5b-grpo-lora/final_adapter
```

