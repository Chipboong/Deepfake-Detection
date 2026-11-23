
import torch.nn as nn
from torchvision import models


def create_model(pretrained=True):

    # Load ResNet-50
    weights = (
        models.ResNet50_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.resnet50(weights=weights)

    # Replace final classification layer
    input_features = model.fc.in_features

    model.fc = nn.Linear(input_features, 2)

    return model
