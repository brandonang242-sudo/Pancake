from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import nn

# Load dataset
dataset = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})

# Load smaller GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize texts
MAX_LENGTH = 256  # manageable for CPU

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load smaller GPT-2 for classification
NUM_LABELS = 1024
model = AutoModelForSequenceClassification.from_pretrained(
    "distilgpt2",
    num_labels=NUM_LABELS
)
model.config.pad_token_id = model.config.eos_token_id

# Optional: smarter classifier
model.classifier = nn.Sequential(
    nn.Linear(model.config.hidden_size, 1024),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(1024, NUM_LABELS)
)

# Use CPU
device = "cpu"
model.to(device)
