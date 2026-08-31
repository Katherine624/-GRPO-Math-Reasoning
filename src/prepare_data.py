"""第一步：处理 GSM8K 数据集，并保存为 GRPO 可以直接使用的格式。"""

import shutil#shutil 是 Python 自带的文件操作工具
from pathlib import Path

from datasets import DatasetDict, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]   #Path 用来表示和拼接文件路径。resolve把路径转换成完整的绝对路径。parents[2]获取上面第二层父目录。
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "gsm8k_grpo"

# 这个系统提示规定模型的输出形式，奖励函数也会检查这个形式。要求模型按照下面的格式回答：
SYSTEM_PROMPT = """请一步一步完成数学推理，并严格使用下面的格式回答：
<reasoning>
这里写推理过程
</reasoning>
<answer>
这里仅写最终答案
</answer>"""


def extract_reference_answer(raw_answer: str) -> str:
    """GSM8K 的标准答案在 #### 后面；这里只取最终答案作为训练标签。"""
    return raw_answer.split("####")[-1].strip()


def format_example(example: dict) -> dict:
    """把一条 GSM8K 样本转成 TRL 对话格式。"""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "answer": extract_reference_answer(example["answer"]),
    }


def main() -> None:
    # GSM8K 官方只有 train 和 test。
    raw = load_dataset("openai/gsm8k", "main")

    # 从官方 train 中固定抽 5% 做 validation；官方 test 始终留作最终测试。
    split = raw["train"].train_test_split(test_size=0.05, seed=42)
    processed = DatasetDict(
        {
            "train": split["train"].map(format_example, remove_columns=raw["train"].column_names),
            "validation": split["test"].map(format_example, remove_columns=raw["train"].column_names),
            "test": raw["test"].map(format_example, remove_columns=raw["test"].column_names),
        }
    )

    # 允许重复运行脚本：只覆盖这个脚本生成的 processed/gsm8k_grpo 目录。
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    processed.save_to_disk(OUTPUT_DIR)

    print(f"数据已保存到：{OUTPUT_DIR}")
    print(processed)
    print("\n处理后的一条样本：")
    print(processed["train"][0])


if __name__ == "__main__":
    main()

