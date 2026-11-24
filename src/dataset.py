import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DATASET_NAME = "Rajarshi-Roy-research/Defactify_Image_Dataset"


# Image preprocessing
train_transform = transforms.Compose(
    [
        transforms.Resize((380, 380)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


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


def get_dataloaders(batch_size=32, num_workers=4):

    train_data = load_dataset(DATASET_NAME, split="train")

    val_data = load_dataset(DATASET_NAME, split="validation")

    test_data = load_dataset(DATASET_NAME, split="test")

    train_dataset = DefactifyDataset(train_data, transform=train_transform)

    val_dataset = DefactifyDataset(val_data, transform=eval_transform)

    test_dataset = DefactifyDataset(test_data, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
