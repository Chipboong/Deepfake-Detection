from collections import Counter

from datasets import load_dataset

# Dataset name
DATASET_NAME = "Rajarshi-Roy-research/Defactify_Image_Dataset"

# Load official splits
train_data = load_dataset(DATASET_NAME, split="train")
val_data = load_dataset(DATASET_NAME, split="validation")
test_data = load_dataset(DATASET_NAME, split="test")

# Display dataset information
print("\n========== DATASET INFORMATION ==========")

print("Training samples:", len(train_data))
print("Validation samples:", len(val_data))
print("Testing samples:", len(test_data))

print("\nColumns:", train_data.column_names)


# Inspect label distribution
def inspect_labels(dataset, split_name):
    labels = Counter(dataset["Label_A"])

    print(f"\n{split_name} label distribution:")
    print("Real (0):", labels.get(0, 0))
    print("Fake (1):", labels.get(1, 0))

    total = len(dataset)

    for label, count in sorted(labels.items()):
        print(f"Label {label} percentage: {count / total * 100:.2f}%")


inspect_labels(train_data, "Training")
inspect_labels(val_data, "Validation")
inspect_labels(test_data, "Testing")

# Inspect one sample
print("\n========== SAMPLE INSPECTION ==========")

sample = train_data[0]

print("Available fields:", sample.keys())
print("Label_A:", sample["Label_A"])
print("Image:", sample["Image"])
