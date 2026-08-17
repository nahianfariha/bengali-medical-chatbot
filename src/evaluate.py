"""
Score model generations on a held-out validation split using the same
composite metric as the original competition:

    score = 0.5 * BERTScore_F1 + 0.3 * token-level F1 + 0.2 * ROUGE-L F1

Requires: pip install bert-score rouge-score

Example:
    python src/evaluate.py \
        --adapter_dir outputs/qwen_medical_bengali_final \
        --val_csv data/val.csv
"""
import argparse

import pandas as pd
from bert_score import score as bert_score
from rouge_score import rouge_scorer

from inference import build_prompt, generate_batch, load_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--val_csv", required=True, help="CSV with columns id, input, output (reference)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=768)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--no_repeat_ngram_size", type=int, default=0)
    p.add_argument("--bert_score_lang", default="bn", help="Language code for BERTScore's default model")
    return p.parse_args()


def token_f1(pred, ref):
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    for t, c in common.items():
        overlap += min(c, ref_counts.get(t, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def main():
    args = parse_args()
    model, tokenizer = load_model(args)

    val_df = pd.read_csv(args.val_csv)
    references = val_df["output"].astype(str).tolist()

    predictions = generate_batch(model, tokenizer, val_df["input"].tolist(), args)

    # BERTScore F1
    _, _, bert_f1 = bert_score(predictions, references, lang=args.bert_score_lang, verbose=False)
    bert_f1 = bert_f1.tolist()

    # Token-level F1
    tok_f1_scores = [token_f1(p, r) for p, r in zip(predictions, references)]

    # ROUGE-L F1
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l_scores = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)]

    composite = [
        0.5 * b + 0.3 * t + 0.2 * r
        for b, t, r in zip(bert_f1, tok_f1_scores, rouge_l_scores)
    ]

    n = len(composite)
    print(f"N examples:        {n}")
    print(f"BERTScore F1 (avg): {sum(bert_f1) / n:.4f}")
    print(f"Token F1 (avg):     {sum(tok_f1_scores) / n:.4f}")
    print(f"ROUGE-L F1 (avg):   {sum(rouge_l_scores) / n:.4f}")
    print(f"Composite (avg):    {sum(composite) / n:.4f}")


if __name__ == "__main__":
    main()
