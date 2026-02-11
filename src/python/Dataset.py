import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch import nn

# -------- Build CSV dataset --------
NUM_CLASSES = 256
EXAMPLES_PER_CLASS = 2  # keep small for CPU demo

texts = []
labels = []

for label in range(NUM_CLASSES):
    for i in range(EXAMPLES_PER_CLASS):
        texts.append(f"This is example {i} for class {label}")
        labels.append(label)

df = pd.DataFrame({"text": texts, "label": labels})
df.to_csv("train.csv", index=False)
df.to_csv("test.csv", index=False)  # simple split for demo

# -------- Load dataset --------
dataset = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})

# -------- Tokenization --------
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
MAX_LENGTH = 128

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

# -------- Load model --------
model = AutoModelForSequenceClassification.from_pretrained(
    "distilgpt2",
    num_labels=NUM_CLASSES
)
model.config.pad_token_id = model.config.eos_token

# Smarter classifier head
model.classifier = nn.Sequential(
    nn.Linear(model.config.hidden_size, 512),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, NUM_CLASSES)
)

device = "cpu"
model.to(device)
