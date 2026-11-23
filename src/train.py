
import os

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model import create_model
from src.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS_FROZEN,
    EPOCHS_FINETUNE,
    HEAD_LR,
    BACKBONE_LR,
    WEIGHT_DECAY,
    CHECKPOINT_PATH,
    USE_AMP
)


def calculate_class_weights(train_loader):
    counts = torch.zeros(2)

    for _, labels in tqdm(train_loader, desc="Counting classes"):
        counts += torch.bincount(labels, minlength=2)

    # Inverse-frequency class weights
    weights = counts.sum() / (2 * counts)

    return weights.to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer, scaler):

    model.train()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Training"):

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_AMP
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    return avg_loss, accuracy, f1


@torch.no_grad()
def validate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Validation"):

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_AMP
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    return avg_loss, accuracy, f1


def main():

    os.makedirs("checkpoints", exist_ok=True)

    print("Using device:", DEVICE)

    # Load datasets
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Class weights
    class_weights = calculate_class_weights(train_loader)

    print("Class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Load model
    model = create_model(pretrained=True).to(DEVICE)

    # Freeze backbone initially
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    optimizer = optim.Adam(
        model.fc.parameters(),
        lr=HEAD_LR,
        weight_decay=WEIGHT_DECAY
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    best_f1 = -1.0

    total_epochs = EPOCHS_FROZEN + EPOCHS_FINETUNE

    for epoch in range(total_epochs):

        # Unfreeze backbone after initial frozen training
        if epoch == EPOCHS_FROZEN:

            print("\nUnfreezing ResNet-50 backbone...")

            for parameter in model.parameters():
                parameter.requires_grad = True

            optimizer = optim.Adam(
                [
                    {
                        "params": [
                            p for name, p in model.named_parameters()
                            if not name.startswith("fc.")
                        ],
                        "lr": BACKBONE_LR
                    },
                    {
                        "params": model.fc.parameters(),
                        "lr": HEAD_LR
                    }
                ],
                weight_decay=WEIGHT_DECAY
            )

        print(f"\nEpoch {epoch + 1}/{total_epochs}")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler
        )

        val_loss, val_acc, val_f1 = validate(
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

        # Save best model
        if val_f1 > best_f1:

            best_f1 = val_f1

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                },
                CHECKPOINT_PATH
            )

            print("Best model checkpoint saved!")

    print("\nTraining completed!")
    print("Best validation F1:", best_f1)


if __name__ == "__main__":
    main()
