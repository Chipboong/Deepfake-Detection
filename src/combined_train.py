
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from pathlib import Path

from src.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS_FROZEN,
    EPOCHS_FINETUNE,
    HEAD_LR,
    BACKBONE_LR,
    WEIGHT_DECAY,
    USE_AMP
)

from src.model import create_model
from src.combined_dataset import get_combined_dataloaders


ORIGINAL_CHECKPOINT = "checkpoints/resnet50_original_best.pth"
NEW_CHECKPOINT = "checkpoints/resnet50_combined_best.pth"

TOTAL_EPOCHS = EPOCHS_FROZEN + EPOCHS_FINETUNE


def calculate_class_weights(dataset):
    from collections import Counter

    counts = Counter()

    for child in dataset.datasets:

        # Original Hugging Face dataset
        if hasattr(child, "dataset") and hasattr(child.dataset, "column_names"):
            labels = child.dataset["Label_A"]
            counts.update(int(label) for label in labels)

        # Local ImageFolder dataset
        elif hasattr(child, "samples"):
            counts.update(
                label for _, label in child.samples
            )

        else:
            raise ValueError(
                f"Unknown dataset type: {type(child).__name__}"
            )

    class_counts = torch.tensor(
        [counts[0], counts[1]],
        dtype=torch.float32
    )

    weights = class_counts.sum() / (2 * class_counts)

    print("\nClass distribution:")
    print(f"Real: {int(class_counts[0])}")
    print(f"Fake: {int(class_counts[1])}")

    print("Class weights:", weights.tolist())

    return weights.to(DEVICE)

def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    training = optimizer is not None

    model.train() if training else model.eval()

    all_predictions = []
    all_labels = []
    total_loss = 0.0

    for images, labels in tqdm(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    return total_loss / len(loader), accuracy, f1


def train():
    Path("checkpoints").mkdir(exist_ok=True)

    train_loader, val_loader = get_combined_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    model = create_model(pretrained=False).to(DEVICE)

    print("\nLoading original best checkpoint...")

    checkpoint = torch.load(
        ORIGINAL_CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)

    print("Original model loaded successfully.")

    class_weights = calculate_class_weights(
        train_loader.dataset
    )

    print("Class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("fc.")

    optimizer = Adam(
        [
            {
                "params": model.fc.parameters(),
                "lr": HEAD_LR
            },
            {
                "params": [
                    p for name, p in model.named_parameters()
                    if not name.startswith("fc.")
                ],
                "lr": BACKBONE_LR
            }
        ],
        weight_decay=WEIGHT_DECAY
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    best_f1 = 0.0

    for epoch in range(TOTAL_EPOCHS):

        if epoch == EPOCHS_FROZEN:
            print("\nUnfreezing ResNet-50 backbone...")

            for parameter in model.parameters():
                parameter.requires_grad = True

        print(f"\nEpoch {epoch + 1}/{TOTAL_EPOCHS}")

        train_loss, train_acc, train_f1 = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler
        )

        val_loss, val_acc, val_f1 = run_epoch(
            model,
            val_loader,
            criterion
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Train F1: {train_f1:.4f}")

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Validation F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "validation_f1": val_f1
                },
                NEW_CHECKPOINT
            )

            print("Combined model checkpoint saved!")


if __name__ == "__main__":
    train()
