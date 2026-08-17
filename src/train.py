"""
Fine-tune Qwen2.5-3B-Instruct on Bengali patient-doctor dialogue with QLoRA.

Supports two modes:
  - quick, subsampled run for fast iteration (--sample_size N)
  - full-dataset run with checkpoint resume (--resume), suited to Colab
    sessions that may disconnect mid-training

Example (quick run):
    python src/train.py --data_dir data --output_dir outputs/quick \
        --sample_size 5000 --epochs 1

Example (full run, resumable):
    python src/train.py --data_dir data --output_dir outputs/full \
        --epochs 2 --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 --use_liger --resume
"""
import argparse
import gc
import os

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

SYSTEM_PROMPT = (
    "আপনি একজন অভিজ্ঞ ও সহানুভূতিশীল বাংলা চিকিৎসক। "
    "রোগীর প্রশ্ন মনোযোগ দিয়ে বিশ্লেষণ করুন এবং প্রাসঙ্গিক চিকিৎসা জ্ঞান ব্যবহার করে "
    "যুক্তিসংগত, নির্ভুল ও সহজ ভাষায় উত্তর দিন। প্রথমে রোগীর সমস্যার সম্ভাব্য কারণ বা "
    "প্রেক্ষাপট সংক্ষেপে ব্যাখ্যা করুন, তারপর প্রয়োজনীয় পরামর্শ দিন। অপ্রয়োজনীয় তথ্য, "
    "পুনরাবৃত্তি বা অনুমানভিত্তিক দাবি করবেন না। উত্তর হবে স্বাভাবিক, আশ্বস্তকারী, "
    "পেশাদার এবং রোগী-বান্ধব বাংলায়।"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default="data", help="Folder containing train.csv")
    p.add_argument("--train_csv", default=None, help="Override path to train.csv")
    p.add_argument("--output_dir", default="outputs/qwen_medical_bengali")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--sample_size", type=int, default=None,
                    help="Subsample this many training rows for a quick run. "
                         "Omit to use the full dataset.")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=768)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--use_liger", action="store_true",
                    help="Apply Liger fused kernels to Qwen2 (faster/lower memory "
                         "on longer runs; requires `pip install liger-kernel`)")
    p.add_argument("--resume", action="store_true",
                    help="Resume from the latest checkpoint-* in output_dir if present")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_data(args):
    train_csv = args.train_csv or os.path.join(args.data_dir, "train.csv")
    df = pd.read_csv(train_csv)
    df = df[["id", "input", "output"]].dropna(subset=["input", "output"]).reset_index(drop=True)

    if args.sample_size is not None:
        df = df.sample(n=min(args.sample_size, len(df)), random_state=args.seed).reset_index(drop=True)

    train_df, val_df = train_test_split(df, test_size=args.val_fraction, random_state=args.seed)
    print(f"Train: {train_df.shape}, Val: {val_df.shape}")
    return train_df, val_df


def build_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def make_format_and_tokenize(tokenizer, max_length):
    """Builds a tokenize fn that masks the prompt tokens with -100 so loss
    is only computed on the assistant's response."""

    def format_and_tokenize(example):
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
        ]
        full_messages = prompt_messages + [{"role": "assistant", "content": example["output"]}]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(
            full_text, truncation=True, max_length=max_length, add_special_tokens=False
        )["input_ids"]

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = full_ids.copy()
        for i in range(prompt_len):
            labels[i] = -100

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    return format_and_tokenize


def has_supervised_tokens(example):
    return any(l != -100 for l in example["labels"])


def custom_data_collator(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            n_pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * n_pad)
            attention_mask.append(f["attention_mask"] + [0] * n_pad)
            labels.append(f["labels"] + [-100] * n_pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def build_model(args):
    if args.use_liger:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2
        apply_liger_kernel_to_qwen2()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def find_last_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not ckpts:
        return None
    return os.path.join(output_dir, sorted(ckpts, key=lambda x: int(x.split("-")[1]))[-1])


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_df, val_df = load_data(args)

    tokenizer = build_tokenizer(args.model_name)
    tokenize_fn = make_format_and_tokenize(tokenizer, args.max_length)

    train_dataset = Dataset.from_pandas(train_df).map(tokenize_fn, remove_columns=["id", "input", "output"])
    val_dataset = Dataset.from_pandas(val_df).map(tokenize_fn, remove_columns=["id", "input", "output"])

    before = len(train_dataset)
    train_dataset = train_dataset.filter(has_supervised_tokens)
    val_dataset = val_dataset.filter(has_supervised_tokens)
    print(f"Train: {before} -> {len(train_dataset)} (dropped fully-truncated examples)")

    model = build_model(args)

    training_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_strategy="steps",
        logging_steps=20,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    try:
        training_args = TrainingArguments(eval_strategy="steps", **training_kwargs)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="steps", **training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=custom_data_collator(tokenizer),
    )

    resume_from = find_last_checkpoint(args.output_dir) if args.resume else None
    if args.resume and resume_from:
        print("Resuming from:", resume_from)
    trainer.train(resume_from_checkpoint=resume_from)

    final_dir = args.output_dir.rstrip("/") + "_final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    print("Model saved to", final_dir)


if __name__ == "__main__":
    main()
