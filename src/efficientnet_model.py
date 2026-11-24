import torch.nn as nn
from torchvision import models


def create_model(pretrained=True):

    # Load EfficientNet-B7
    weights = (
        models.EfficientNet_B7_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.efficientnet_b7(weights=weights)

    # Replace final classification layer (classifier[0] is Dropout, classifier[1] is Linear)
    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(input_features, 2)

    return model