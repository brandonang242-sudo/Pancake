from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch import nn

NUM_LABELS = 1024
MAX_LENGTH = 256  # long enough to capture context, not too long for CPU

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    "distilgpt2",
    num_labels=NUM_LABELS
)
model.config.pad_token_id = model.config.eos_token_id

model.classifier = nn.Sequential(
    nn.Linear(model.config.hidden_size, 1024),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(1024, NUM_LABELS)
)
