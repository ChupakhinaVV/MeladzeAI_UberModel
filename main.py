import torch
import numpy as np
from datasets import load_dataset
from transformers import (GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments)
from sacrebleu.metrics import CHRF

def main():
    # === Загружаем датасет ===
    ds = load_dataset("json", data_files="meladze.jsonl")
    split = ds["train"].train_test_split(test_size=0.1, seed=42)
    raw_train, raw_test = split["train"], split["test"]

    # === Загружаем модель и токенизатор ===
    MODEL = "ai-forever/rugpt3medium_based_on_gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # === Подготовка данных ===
    def preprocess(example):
        full = example["prompt"] + "\n" + example["text"]
        tokens = tokenizer(
            full,
            truncation=True,
            padding="max_length",
            max_length=768
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = split.map(preprocess, remove_columns=["prompt", "text"])

    # === Аргументы обучения ===
    args = TrainingArguments(
        output_dir="./gpt3-meladze",
        per_device_train_batch_size=2,
        num_train_epochs=50,
        learning_rate=8e-5,
        logging_dir="./logs",
        logging_steps=20,
        save_steps=500,
        save_total_limit=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        overwrite_output_dir=True
    )

    # === Trainer ===
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
    )

    trainer.train()

    # === Сохраняем ===
    model.save_pretrained("./gpt3-meladze")
    tokenizer.save_pretrained("./gpt3-meladze")

    # === Генерация коротких текстов ===
    def generate(prompt, max_new_tokens=40):
        inputs = tokenizer(prompt + "\n", return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=40,
            top_p=0.92,
            temperature=1.1,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(output[0], skip_special_tokens=True)

    # === Примеры ===
    for title in ["Полюбил", "Сэра", "Иностранец"]:
        print(f"\n=== Пример для '{title}' ===")
        print(generate(f"generate_song: {title}"))

    # === Метрики ===
    refs, hyps = [], []
    for ex in raw_test:
        hyp = generate(ex["prompt"])
        refs.append([ex["text"]])
        hyps.append(hyp)

    print("\nchrF++     ", CHRF().corpus_score(hyps, refs).score)

    def distinct_n(corpus, n):
        grams, total = set(), 0
        for txt in corpus:
            toks = txt.split()
            total += max(len(toks) - n + 1, 0)
            for i in range(len(toks) - n + 1):
                grams.add(tuple(toks[i:i + n]))
        return len(grams) / total if total > 0 else 0

    print("Distinct-1 ", distinct_n(hyps, 1))
    print("Distinct-2 ", distinct_n(hyps, 2))

    train_tokens = set(" ".join(raw_train["text"]).split())
    gen_tokens   = set(" ".join(hyps).split())
    novelty = 100 * len(gen_tokens - train_tokens) / len(gen_tokens)
    print("Novelty %  ", novelty)

    # Perplexity
    model.eval()
    losses = []
    for ex in raw_test:
        with torch.no_grad():
            full   = ex["prompt"] + "\n" + ex["text"]
            tokens = tokenizer(full, return_tensors="pt", truncation=True, max_length=512).to(device)
            out    = model(**tokens, labels=tokens["input_ids"])
            losses.append(out.loss.item())
    print("Perplexity ", float(np.exp(np.mean(losses))))

if __name__ == "__main__":
    main()
