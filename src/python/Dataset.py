import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

NUM_CLASSES = 256
EXAMPLES_PER_CLASS = 2  # small for CPU
MAX_LENGTH = 128

def build_csv():
    texts = []
    labels = []

    for label in range(NUM_CLASSES):
        for i in range(EXAMPLES_PER_CLASS):
            texts.append(f"This is example {i} for class {label}")
            labels.append(label)

    df = pd.DataFrame({"text": texts, "label": labels})
    df.to_csv("train.csv", index=False)
    df.to_csv("test.csv", index=False)

def load_tokenized_dataset():
    build_csv()
    dataset = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

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
    return tokenized_datasets
