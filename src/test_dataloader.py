
import torch
from dataset import get_dataloaders


def main():

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=32,
        num_workers=0
    )

    images, labels = next(iter(train_loader))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Labels:", labels)

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Testing batches:", len(test_loader))

    print("GPU available:", torch.cuda.is_available())


if __name__ == "__main__":
    main()
