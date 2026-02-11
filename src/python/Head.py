import torch
from torch import nn
from transformers import AutoModelForSequenceClassification

NUM_CLASSES = 256

# Load smaller GPT-2
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
