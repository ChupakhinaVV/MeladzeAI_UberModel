import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from sacrebleu.metrics import CHRF

MODEL_NAME = "ai-forever/rugpt3medium_based_on_gpt2"
MODEL_PATH = "./gpt3-meladze"

def load_tokenizer_model(device):
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    model.to(device)
    return tokenizer, model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_dataset("json", data_files="meladze_seq.jsonl")
    split = ds["train"].train_test_split(test_size=0.1, seed=42)
    raw_train, raw_test = split["train"], split["test"]

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    def preprocess(example):
        full = example["prompt"] + "\n" + example["text"]
        tokens = tokenizer(
            full,
            truncation=True,
            padding="max_length",
            max_length=64
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = split.map(preprocess, remove_columns=["prompt", "text"])

    args = TrainingArguments(
        output_dir=MODEL_PATH,
        num_train_epochs=6,
        learning_rate=8e-5,
        logging_dir="./logs",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        overwrite_output_dir=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

    # === Метрики ===
    refs, hyps = [], []
    for ex in raw_test:
        hyp = generate(tokenizer,device,model,ex["prompt"])
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
    gen_tokens = set(" ".join(hyps).split())
    novelty = 100 * len(gen_tokens - train_tokens) / len(gen_tokens)
    print("Novelty %  ", novelty)

    # Perplexity
    model.eval()
    losses = []
    for ex in raw_test:
        with torch.no_grad():
            full = ex["prompt"] + "\n" + ex["text"]
            tokens = tokenizer(full, return_tensors="pt", truncation=True, max_length=512).to(device)
            out = model(**tokens, labels=tokens["input_ids"])
            losses.append(out.loss.item())
    print("Perplexity ", float(np.exp(np.mean(losses))))

def generate_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_tokenizer_model(device)



    while True:
        user_input = input("\nВведите заголовок песни через generate_song: или строку которую продолжить через continue: (или 'exit' для выхода): ").strip()
        if user_input.lower() == "exit":
            break
        prompt = f"{user_input}"
        result = generate(tokenizer,device,model,prompt)

        print(f"\n📝 Запрос:\n{user_input}")
        print(f"\n📝 Сгенерировано:\n{result}")

def main():
    print("Выберите режим работы:")
    print("1 — Обучить модель")
    print("2 — Генерация (использовать сохранённую модель)")

    choice = input("Ваш выбор: ").strip()

    if choice == "1":
        train()
    elif choice == "2":
        if not os.path.exists(MODEL_PATH):
            print("❌ Модель не найдена. Сначала обучите её.")
        else:
            generate_loop()
    else:
        print("❗ Неверный выбор")


def generate(tokenizer,device,model,prompt, max_new_tokens=40):
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

if __name__ == "__main__":
    main()
