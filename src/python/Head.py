import torch
from torch import nn
from transformers import AutoModelForSequenceClassification

NUM_LABELS = 1024

# Load smaller GPT-2
model = AutoModelForSequenceClassification.from_pretrained(
    "distilgpt2",
    num_labels=NUM_LABELS
)

# Fix pad token
model.config.pad_token_id = model.config.eos_token_id

# Smarter classifier with hidden layer
model.classifier = nn.Sequential(
    nn.Linear(model.config.hidden_size, 1024),  # hidden layer
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(1024, NUM_LABELS)
)

# Use CPU
device = "cpu"
model.to(device)
