import json

input_path = "meladze.jsonl"
output_path = "meladze_seq.jsonl"

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        base_prompt = data["prompt"].strip()

        # Разбиваем на строки и фильтруем "припевы"
        lines = [l.strip() for l in data["text"].split("\n") if l.strip()]
        lines = [l for l in lines if not l.lower().startswith("припев")]

        if len(lines) < 2:
            continue  # не с чем работать

        # Первая строка — генерация от названия
        outfile.write(json.dumps({
            "prompt": f"generate_song: {base_prompt}",
            "text": lines[0]
        }, ensure_ascii=False) + "\n")

        # Остальные строки — как продолжения (кроме последней)
        for i in range(1, len(lines) - 1):
            outfile.write(json.dumps({
                "prompt": f"continue: {lines[i - 1]}",
                "text": lines[i]
            }, ensure_ascii=False) + "\n")
