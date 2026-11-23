
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset
from torchvision import datasets, transforms
from pathlib import Path

DATASET_NAME = "Rajarshi-Roy-research/Defactify_Image_Dataset"
CUSTOM_TRAIN_PATH = Path("picture/train")

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


class DefactifyDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        image = sample["Image"].convert("RGB")
        label = int(sample["Label_A"])

        if self.transform:
            image = self.transform(image)

        return image, label


class LocalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform

        # ImageFolder mapping: fake=0, real=1
        # Required mapping: real=0, fake=1
        self.samples = [
            (path, 1 if label == 0 else 0)
            for path, label in self.dataset.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_combined_dataloaders(batch_size=32, num_workers=0):

    print("Loading original training dataset...")

    original_data = load_dataset(
        DATASET_NAME,
        split="train"
    )

    original_train = DefactifyDataset(
        original_data,
        transform=train_transform
    )

    print("Loading local training dataset...")

    local_train = LocalImageDataset(
        CUSTOM_TRAIN_PATH,
        transform=train_transform
    )

    combined_train = ConcatDataset([
        original_train,
        local_train
    ])

    print("\nDataset statistics:")
    print(f"Original training images: {len(original_train)}")
    print(f"Local training images: {len(local_train)}")
    print(f"Combined training images: {len(combined_train)}")

    print("\nLoading original validation dataset...")

    val_data = load_dataset(
        DATASET_NAME,
        split="validation"
    )

    val_dataset = DefactifyDataset(
        val_data,
        transform=eval_transform
    )

    train_loader = DataLoader(
        combined_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
