import torch
from torch import nn
from transformers import GPT2ForSequenceClassification

model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=1024)
model.config.pad_token_id = model.config.eos_token_id

model.classifier = nn.Sequential(
    nn.Linear(model.config.n_embd, 2048),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(2048, 1024)
)
