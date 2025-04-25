import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
)
from sacrebleu.metrics import CHRF

def main():
    # 1) Загрузка JSONL: по умолчанию создаётся split 'train'
    ds = load_dataset("json", data_files="meladze.jsonl")
    # 2) Делим на train/test
    split = ds["train"].train_test_split(test_size=0.1, seed=42)
    raw_train = split["train"]
    raw_test  = split["test"]

    # 3) Превью первых трёх примеров
    print("=== Dataset preview (first 3 examples) ===")
    for i, ex in enumerate(raw_train.select(range(3)), 1):
        print(f"{i}) prompt: {ex['prompt']!r}")
        print(f"   text : {ex['text']!r}\n")

    # 4) Токенизатор и модель
    BASE = "cointegrated/rut5-base"
    tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    model     = AutoModelForSeq2SeqLM.from_pretrained(BASE)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 5) Функция препроцессинга
    def preprocess(batch):
        # убираем ведущие/замыкающие кавычки, если они есть
        batch["text"] = [t.strip('"') for t in batch["text"]]
        enc = tokenizer(
            batch["prompt"],
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        tgt = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )
        enc["labels"] = tgt["input_ids"]
        return enc

    tokenized = split.map(
        preprocess,
        batched=True,
        remove_columns=["prompt", "text"]
    )

    # 6) Обучение
    training_args = TrainingArguments(
        output_dir="./rut5-meladze",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        learning_rate=2e-5,
        warmup_steps=50,
        logging_dir="./logs",
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
    )
    trainer.train()

    # 7) Сохраняем
    model.save_pretrained("./rut5-meladze")
    tokenizer.save_pretrained("./rut5-meladze")

    # 8) Генератор
    def generate(prompt: str, max_new_tokens: int = 120) -> str:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=64
        ).to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            repetition_penalty=1.1,
        )
        return tokenizer.decode(out[0], skip_special_tokens=True)

    # 9) Пара примеров
    for title in ["Иностранец", "Сэра"]:
        print(f"\n=== Пример для «{title}» ===")
        print(generate(f"generate_song: {title}"))

    # 10) Метрики на test
    refs, hyps = [], []
    for ex in raw_test:
        hyp = generate(ex["prompt"])
        refs.append([ex["text"]])
        hyps.append(hyp)

    print("\n=== Evaluation on test set ===")
    chrf = CHRF()
    print("chrF++      ", chrf.corpus_score(hyps, refs).score)

    def distinct_n(corpus, n):
        grams, total = set(), 0
        for txt in corpus:
            toks = txt.split()
            total += max(len(toks)-n+1, 0)
            for i in range(len(toks)-n+1):
                grams.add(tuple(toks[i:i+n]))
        return len(grams)/total if total>0 else 0

    print("Distinct-1  ", distinct_n(hyps, 1))
    print("Distinct-2  ", distinct_n(hyps, 2))

    train_tok = set()
    for t in raw_train["text"]:
        train_tok.update(t.split())
    gen_tok = set(w for hyp in hyps for w in hyp.split())
    novelty = 100 * len(gen_tok - train_tok) / len(gen_tok) if gen_tok else 0
    print("Novelty (%) ", novelty)

    model.eval()
    losses = []
    for ex in raw_test:
        with torch.no_grad():
            enc = tokenizer(
                ex["prompt"],
                return_tensors="pt",
                truncation=True,
                max_length=64
            ).to(device)
            lbls = tokenizer(
                ex["text"],
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).input_ids.to(device)
            out = model(**enc, labels=lbls)
            losses.append(out.loss.item())
    ppl = float(np.exp(np.mean(losses)))
    print("Perplexity  ", ppl)

if __name__ == "__main__":
    main()
