from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from Dataset import load_tokenized_dataset
from Head import create_classifier

NUM_CLASSES = 256

# Load dataset
tokenized_datasets = load_tokenized_dataset()

# Load GPT-2 model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilgpt2",
    num_labels=NUM_CLASSES
)
model.config.pad_token_id = model.config.eos_token_id

# Add custom classifier head
model = create_classifier(model, model.config.hidden_size, NUM_CLASSES)

# Use CPU
device = "cpu"
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    evaluation_strategy="steps"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

trainer.train()
