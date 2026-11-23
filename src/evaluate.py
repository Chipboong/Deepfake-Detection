
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from tqdm import tqdm

from src.config import DEVICE, BATCH_SIZE, CHECKPOINT_PATH, NUM_WORKERS
from src.dataset import get_dataloaders
from src.model import create_model


def evaluate():
    print("Loading test dataset...")

    _, _, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    print("Loading best model checkpoint...")

    model = create_model(pretrained=False).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    # Support checkpoints saved as either a state dict
    # or a dictionary containing model_state_dict.
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    all_predictions = []
    all_labels = []

    print("Evaluating model...")

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):

            images = images.to(DEVICE)
            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    # Overall metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    print("\n========== TEST RESULTS ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1],
            target_names=["Real", "Fake"],
            digits=4
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
    evaluate()
