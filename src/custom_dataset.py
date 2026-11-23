
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CUSTOM_DATA_PATH = Path("picture")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_custom_dataloaders(batch_size=32, num_workers=0):

    train_dataset = datasets.ImageFolder(
        CUSTOM_DATA_PATH / "train",
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        CUSTOM_DATA_PATH / "test",
        transform=eval_transform
    )

    # ImageFolder sorts alphabetically:
    # fake = 0, real = 1
    # We need to convert to:
    # real = 0, fake = 1

    print("Original folder mapping:", train_dataset.class_to_idx)

    train_dataset.targets = [
        1 if label == 0 else 0
        for label in train_dataset.targets
    ]

    test_dataset.targets = [
        1 if label == 0 else 0
        for label in test_dataset.targets
    ]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader
