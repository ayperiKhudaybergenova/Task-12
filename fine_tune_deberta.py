import json
import jsonlines
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DebertaV3Tokenizer,
    DebertaV3ForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# -------------------------
# Config
# -------------------------
DOCS_PATH = "docs_train.json"
TRAIN_Q_PATH = "questions_train.jsonl"
DEV_Q_PATH = "questions_dev.jsonl"
MODEL_NAME = "microsoft/deberta-v3-large"

# -------------------------
# Load Docs → dict[(topic_id, uuid)] = combined context text
# -------------------------
with open(DOCS_PATH, "r") as f:
    docs_data = json.load(f)

context_lookup = {}
for item in docs_data:
    tid = item["topic_id"]
    topic = item["topic"]
    for d in item["docs"]:
        uid = d["uuid"]
        pieces = [
            d.get("title", ""),
            d.get("snippet", ""),
            d.get("content", "")
        ]
        combined = topic + "\n" + "\n".join(pieces)
        context_lookup[(tid, uid)] = combined.strip()

# -------------------------
# Load Questions
# -------------------------
def load_questions(path):
    samples = []
    with jsonlines.open(path, "r") as reader:
        for q in reader:
            tid = q["topic_id"]
            uid = q["uuid"]
            context = context_lookup.get((tid, uid), "")
            question = q["target_event"]

            options = {
                "A": q["option_A"],
                "B": q["option_B"],
                "C": q["option_C"],
                "D": q["option_D"],
            }

            input_text = (
                f"Question: {question}\n"
                f"Context: {context}\n"
                f"Options:\n"
                f"A: {options['A']}\n"
                f"B: {options['B']}\n"
                f"C: {options['C']}\n"
                f"D: {options['D']}"
            )

            golden = q["golden_answer"].replace(" ", "").split(",")
            label = [
                1 if letter in golden else 0
                for letter in ["A", "B", "C", "D"]
            ]

            samples.append((input_text, label))
    return samples

train_data = load_questions(TRAIN_Q_PATH)
dev_data = load_questions(DEV_Q_PATH)

# -------------------------
# Dataset Class
# -------------------------
tokenizer = DebertaV3Tokenizer.from_pretrained(MODEL_NAME)

class TaskDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        enc = tokenizer(text, truncation=True, padding="max_length", max_length=512)
        enc = {k: torch.tensor(v) for k,v in enc.items()}
        enc["labels"] = torch.tensor(label, dtype=torch.float)
        return enc

train_ds = TaskDataset(train_data)
dev_ds = TaskDataset(dev_data)

# -------------------------
# Model
# -------------------------
model = DebertaV3ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4,
    problem_type="multi_label_classification"
)

# -------------------------
# Training Arguments
# -------------------------
args = TrainingArguments(
    output_dir="./deberta_out",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    num_train_epochs=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    weight_decay=0.01,
    load_best_model_at_end=True,
    fp16=True,
)

# -------------------------
# Trainer (with Early Stopping)
# -------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# -------------------------
# Run Training
# -------------------------
trainer.train()
trainer.save_model("./deberta_finetuned")
