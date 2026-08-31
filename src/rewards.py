"""GRPO 奖励函数：判断答案是否正确，以及输出格式是否符合要求。"""

import re
from typing import Any


# 模型必须尽量按这个 XML 结构回答。
# reasoning 放推理过程，answer 只放最终数字。
STRICT_PATTERN = re.compile(
    r"^\s*<reasoning>\s*.*?\s*</reasoning>\s*"
    r"<answer>\s*.*?\s*</answer>\s*$",
    re.DOTALL,
)

SOFT_PATTERN = re.compile(
    r"<reasoning>.*?</reasoning>.*?<answer>.*?</answer>",
    re.DOTALL,
)


def get_completion_texts(completions: list[Any]) -> list[str]:  #取出模型回答文字
    """把 TRL 传入的 completion 统一取成普通字符串。"""
    texts = []
    for completion in completions:
        # 对话数据通常长这样：[{"role": "assistant", "content": "..."}]
        if isinstance(completion, list) and completion:
            texts.append(str(completion[0].get("content", "")))
        else:
            texts.append(str(completion))
    return texts


def extract_xml_answer(text: str) -> str:
    """提取 <answer> 与 </answer> 之间的最终答案。"""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def normalize_answer(answer: str) -> str:
    """统一答案写法，例如 1,080、$1080 和 1080 都视为同一个数。"""
    return re.sub(r"[\s,$]", "", str(answer)).strip()


def correctness_reward_func(
    prompts: list[Any],
    completions: list[Any],
    answer: list[str],
    **kwargs: Any,
) -> list[float]:
    """最终答案完全正确奖励 2 分；错误或没写 <answer> 得 0 分。"""
    del prompts, kwargs
    predictions = [extract_xml_answer(text) for text in get_completion_texts(completions)]
    return [
        2.0 if normalize_answer(prediction) == normalize_answer(target) else 0.0
        for prediction, target in zip(predictions, answer)
    ]


def integer_reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
    """最终答案是一个完整整数奖励 0.5 分，引导模型不要输出多余文字。"""
    del kwargs
    predictions = [extract_xml_answer(text) for text in get_completion_texts(completions)]
    return [0.5 if re.fullmatch(r"[+-]?\d+", normalize_answer(x)) else 0.0 for x in predictions]


def strict_format_reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
    """整个回答严格符合 reasoning + answer XML 结构，奖励 0.5 分。"""
    del kwargs
    return [0.5 if STRICT_PATTERN.fullmatch(text) else 0.0 for text in get_completion_texts(completions)]


def soft_format_reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
    """回答里包含完整 XML 结构就奖励 0.5 分，要求比 strict 更宽松。"""
    del kwargs
    return [0.5 if SOFT_PATTERN.search(text) else 0.0 for text in get_completion_texts(completions)]


def xml_count_reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
    """四个 XML 标签每出现且只出现一次奖励 0.125 分，最多 0.5 分。"""
    del kwargs
    tags = ("<reasoning>", "</reasoning>", "<answer>", "</answer>")
    rewards = []
    for text in get_completion_texts(completions):
        score = sum(0.125 for tag in tags if text.count(tag) == 1)
        rewards.append(score)
    return rewards


# 训练脚本会按这个顺序调用全部奖励函数，并把分数加起来。
REWARD_FUNCTIONS = [
    correctness_reward_func,
    integer_reward_func,
    strict_format_reward_func,
    soft_format_reward_func,
    xml_count_reward_func,
]

