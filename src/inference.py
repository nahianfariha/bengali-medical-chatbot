"""
Merge a trained LoRA adapter into the base model and batch-generate doctor
responses for a test set, writing a submission.csv in the competition's
expected format (id, output).

Example:
    python src/inference.py \
        --adapter_dir outputs/qwen_medical_bengali_final \
        --test_csv data/test.csv \
        --out_csv submission.csv
"""
import argparse

import pandas as pd
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from train import SYSTEM_PROMPT  # reuse the same system prompt used in training


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter_dir", required=True, help="Directory with the saved LoRA adapter + tokenizer")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--test_csv", required=True)
    p.add_argument("--out_csv", default="submission.csv")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=768)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--no_repeat_ngram_size", type=int, default=0)
    return p.parse_args()


def build_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model = model.merge_and_unload()
    model.config.use_cache = True
    model.eval()
    return model, tokenizer


def generate_batch(model, tokenizer, questions, args):
    predictions = []
    pbar = tqdm(total=len(questions), desc="Generating")
    for i in range(0, len(questions), args.batch_size):
        batch = questions[i:i + args.batch_size]
        prompts = [build_prompt(tokenizer, q) for q in batch]

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=args.max_length,
        ).to(model.device)

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        if args.no_repeat_ngram_size:
            gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)

        input_len = inputs.input_ids.shape[1]
        for out in outputs:
            predictions.append(tokenizer.decode(out[input_len:], skip_special_tokens=True).strip())

        pbar.update(len(batch))
    pbar.close()
    return predictions


def main():
    args = parse_args()
    model, tokenizer = load_model(args)

    test_df = pd.read_csv(args.test_csv)
    print(test_df.shape)

    predictions = generate_batch(model, tokenizer, test_df["input"].tolist(), args)

    submission = pd.DataFrame({"id": test_df["id"], "output": predictions})
    submission["output"] = submission["output"].fillna("").astype(str).str.strip()

    empty_count = (submission["output"] == "").sum()
    print(f"Empty predictions: {empty_count}")
    submission.loc[submission["output"] == "", "output"] = "দুঃখিত, উত্তর তৈরি করা সম্ভব হয়নি।"

    submission.to_csv(args.out_csv, index=False, encoding="utf-8")
    print("Saved:", args.out_csv)


if __name__ == "__main__":
    main()
