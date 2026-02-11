from torch import nn

def create_classifier(model, hidden_size, num_classes):
    model.classifier = nn.Sequential(
        nn.Linear(hidden_size, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, num_classes)
    )
    return model
