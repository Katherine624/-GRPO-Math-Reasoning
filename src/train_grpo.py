"""第二、三步：加载 Qwen 基座模型，并使用处理后的 GSM8K 做 GRPO 训练。"""

import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from rewards import REWARD_FUNCTIONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "Qwen2.5-0.5B-Instruct"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "gsm8k_grpo"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qwen2.5-0.5b-grpo-lora"


def parse_args() -> argparse.Namespace:
    """命令行参数让我们先短训，再用同一份代码做正式训练。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=10, help="参数更新次数")
    parser.add_argument(
        "--train-samples",
        type=int,
        default=256,
        help="使用多少条训练数据；传 0 表示使用全部数据",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"找不到基座模型：{MODEL_DIR}")
    if not DATA_DIR.exists():
        raise FileNotFoundError("找不到处理后的数据，请先运行 src/prepare_data.py")

    dataset = load_from_disk(DATA_DIR)
    train_dataset = dataset["train"]
    if args.train_samples > 0:
        train_dataset = train_dataset.select(range(min(args.train_samples, len(train_dataset))))

    # tokenizer 负责把文字变成 token；左侧填充更适合批量生成。
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 这里才是真正加载基座模型。BF16 可以显著降低显存占用。
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    # 8GB 显存不适合全参数 GRPO，所以只训练 LoRA 小矩阵。
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # 每道题生成 2 个回答，比较同组回答的奖励，这就是 GRPO 的核心。
    training_args = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        generation_batch_size=2,
        num_generations=2,
        max_completion_length=192,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        use_cache=False,
        beta=0.04,
        loss_type="grpo",
        temperature=0.9,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        use_vllm=False,
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCTIONS,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
    )

    trainer.train()

    # 保存的是体积很小的 LoRA 适配器，不会复制一份完整基座模型。
    final_dir = OUTPUT_DIR / "final_adapter"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"训练完成，LoRA 已保存到：{final_dir}")


if __name__ == "__main__":
    main()

