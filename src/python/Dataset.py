from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
from torch import nn
dataset = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token 

# Increase max_length for tokenization
MAX_LENGTH = 2048 

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load GPT-2 model for classification
NUM_LABELS = 1024
model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=NUM_LABELS)

# Resize positional embeddings to support longer sequences
model.resize_position_embeddings(MAX_LENGTH)

# Fix pad token
model.config.pad_token_id = model.config.eos_token_id

model.classifier = nn.Sequential(
    nn.Linear(model.config.n_embd, 2048),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(2048, NUM_LABELS)
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
