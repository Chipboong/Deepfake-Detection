
import torch
from src.model import create_model


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = create_model(pretrained=True)
    model = model.to(device)

    print("Model loaded successfully!")
    print("Device:", device)

    # Test with one batch
    dummy_images = torch.randn(
        4, 3, 224, 224
    ).to(device)

    with torch.no_grad():
        outputs = model(dummy_images)

    print("Input shape:", dummy_images.shape)
    print("Output shape:", outputs.shape)

    predictions = torch.argmax(outputs, dim=1)

    print("Predictions:", predictions)


if __name__ == "__main__":
    main()
