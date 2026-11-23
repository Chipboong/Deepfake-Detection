
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from src.config import DEVICE, BATCH_SIZE, CHECKPOINT_PATH
from src.model import create_model


CUSTOM_TEST_PATH = Path("picture/test")

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.classes = {
            "Real": 0,
            "Fake": 1
        }

        self.samples = []

        for class_name, label in self.classes.items():
            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Missing folder: {class_dir}"
                )

            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() in [
                    ".jpg", ".jpeg", ".png", ".webp"
                ]:
                    self.samples.append((image_path, label))

        if not self.samples:
            raise ValueError("No images found in custom_test.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def evaluate_custom():
    print("Loading custom dataset...")

    dataset = CustomImageDataset(
        CUSTOM_TEST_PATH,
        transform=eval_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"Total images: {len(dataset)}")
    print(f"Real images: {sum(label == 0 for _, label in dataset.samples)}")
    print(f"Fake images: {sum(label == 1 for _, label in dataset.samples)}")

    print("\nLoading best model checkpoint...")

    model = create_model(pretrained=False).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    all_predictions = []
    all_labels = []

    print("\nEvaluating custom dataset...")

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images = images.to(DEVICE)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    print("\n========== CUSTOM TEST RESULTS ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1],
            target_names=["Real", "Fake"],
            digits=4,
            zero_division=0
        )
    )

    print("\n========== CONFUSION MATRIX ==========")

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1]
    )

    print("Rows = Actual, Columns = Predicted")
    print(cm)


if __name__ == "__main__":
    evaluate_custom()
